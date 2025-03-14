import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Layer
from keras import Model
import tensorflow_probability as tfp
import keras
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
import matplotlib.pyplot as plt

from src.CANN.cont_mech import modelArchitecture
from src.utils import *
import sympy
from sympy import Symbol, lambdify, diff, exp

from src.CANN.models import SingleInvNet_symbolic, SingleInvNetI4_symbolic, ortho_cann_3ff
import re

from src.CANN.models import I4_theta

from src.CANN.util_functions import reshape_input_output_mesh, traindata, makeDIR, Compile_and_fit, get_model_id
from src.CANN.models import get_max_inv_mesh, calculate_I4theta_max


# Takes raw data as input, outputs means and variances of latent variables
class Encoder(Layer):
    def __init__(self, latent_dim):
        super().__init__()
        self.fcnet = Sequential()
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(128, activation='relu')
        self.dense3 = Dense(128, activation='relu')
        self.dense4 = Dense(latent_dim * 2) # mean and log std dev
        self.fcnet.add(self.batchnorm)
        self.fcnet.add(self.dense1)
        self.fcnet.add(self.dense2)
        self.fcnet.add(self.dense3)
        self.fcnet.add(self.dense4)
        self.fcnet.add(self.batchnorm2)

    def call(self, inputs, *args, **kwargs):
        return self.fcnet(inputs)

    def get_config(self):
        return super(Encoder, self).get_config()

# Takes latent variables as input, outputs parameters for BCANN
class Decoder(Layer):
    def __init__(self, n_params):
        super().__init__()
        self.fcnet = Sequential()
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(256, activation='relu')
        self.dense3 = Dense(128, activation='relu')
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.dense4 = Dense(n_params, activation=tf.math.exp)
        self.fcnet.add(self.batchnorm)
        self.fcnet.add(self.dense1)
        self.fcnet.add(self.dense2)
        self.fcnet.add(self.dense3)
        self.fcnet.add(self.batchnorm2)
        self.fcnet.add(self.dense4)


    def call(self, inputs, *args, **kwargs):
        return self.fcnet(inputs)

    def get_config(self):
        return super(Encoder, self).get_config()

symb_wstar = Symbol("wstar")
symb_inv = Symbol("inv")
initializer_1 = tf.keras.initializers.RandomUniform(minval=0., maxval=1)

# Function for converting symbolic expression of Psi to stress (with regularization)
def get_stress_expression(Psi_expr, I1s_max, should_normalize=True):
    # Compute max I1
    I1s_max = tf.cast(I1s_max, tf.float32)
    I1maxmax = np.max(I1s_max) if should_normalize else 1.0
    # Differentiate strain energy to get stress
    P_expr = diff(Psi_expr, symb_inv)
    # Convert strain energy and stress to tf lambda expressions
    Psi_func = lambdify([symb_inv, symb_wstar], Psi_expr, "tensorflow")
    P_func = lambdify([symb_inv, symb_wstar], P_expr, "tensorflow")
    # Compute normalized stress lambda expression
    epsilon = 1e-8
    P_out = lambda inv, wstar : 1 / I1maxmax * P_func(inv / I1maxmax, wstar) / (tf.reduce_sum(Psi_func(I1s_max / I1maxmax, wstar)) + epsilon)
    out_func = P_out if should_normalize else P_func
    return out_func

# Custom layer that maps invariant I to derivative dPsi / dI, where Psi is specified by a sympy expression
class SingleTermStress(keras.layers.Layer):
    def __init__(self, kernel_initializer, Psi_expr, weight_name, should_normalize, p, alpha, I1s_max):
        super().__init__()
        self.P_func = get_stress_expression(Psi_expr, I1s_max, should_normalize) # Compute function for normalized stress from symbolic expression
        self.weight_name = weight_name
        self.p = p
        self.kernel_initializer = kernel_initializer
        self.alpha = alpha
        self.I1maxmax = np.max(I1s_max)
    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_initializer": self.kernel_initializer,
            "P_func": self.P_func,
            "weight_name": self.weight_name,
            "p": self.p,
            "alpha": self.alpha
        })
        return config

    def build(self, input_shape):
        # Wstar is nonnegative and initialized as specified in constructior
        self.wstar = self.add_weight(
            shape=(1, ),
            initializer=self.kernel_initializer,
            constraint=keras.constraints.NonNeg(),
            trainable=True,
            name=self.weight_name + '1'
        )
        # W is nonnegative and has lp regularization applied with weight alpha, initialized with uniform distribution
        self.w = self.add_weight(
            shape=(1,),
            initializer=initializer_1,
            constraint= keras.constraints.NonNeg(),
            regularizer=keras.regularizers.l1(self.alpha),
            trainable=True,
            name=self.weight_name + '2'
        )
    def call(self, inputs):
        mean = self.w ** (1 / self.p) * self.P_func(inputs, self.wstar)
        return mean

def single_term_stress_vae(Psi_expr, invs, I1s_max, should_normalize=True, p=1.0, w=None, wstar=None, wsigma=None):
    if w is None:
        w = tf.keras.Input(shape=(1,))
        wstar = tf.keras.Input(shape=(1,))
        wsigma = tf.keras.Input(shape=(1,))

    P_func = get_stress_expression(Psi_expr, I1s_max, should_normalize)
    invs = tf.cast(invs, tf.float32)
    mean =  w ** (1 / p) * (P_func(invs, wstar) + invs * 0.0)
    stddev = (mean * wsigma)
    return mean, stddev, [wstar, w, wsigma]

# Define activation functions for use in strain energy symbolic expressions
def identity(x):
    return x

def activation_exp(x):
    return exp(x) - 1.0
def activation_exp_I4(x):
    return exp(x) - x - 1.0

#
def flatten(arr):
    return [x for y in arr for x in y]
def single_inv_stress_vae(invs, I1s_max, output_grad, should_normalize, p):
    activations = [identity, activation_exp]
    n_exps = 2
    Psi_funcs = [activations[j](symb_wstar * symb_inv ** (i + 1)) for i in range(n_exps) for j in range(len(activations))]
    terms = [single_term_stress_vae(Psi_func, invs, I1s_max, should_normalize=should_normalize, p=p) for Psi_func in Psi_funcs]
    means = get_stress_mean([x[0] for x in terms], output_grad)
    variance = get_stress_var([x[1] for x in terms], output_grad)
    params = flatten([x[2] for x in terms]) # List of all inputs
    return means, variance, params

def single_inv_I4_stress_vae(invs, I1s_max, output_grad, should_normalize, p):
    activations = [activation_exp_I4, identity, activation_exp]
    exps = [1., 2., 2.]
    Psi_funcs = [activations[i](symb_wstar * symb_inv ** exps[i]) for i in range(len(activations))]
    terms = [single_term_stress_vae(Psi_func, invs, I1s_max, should_normalize=should_normalize, p=p) for Psi_func in Psi_funcs]
    means = get_stress_mean([x[0] for x in terms], output_grad)
    variance = get_stress_var([x[1] for x in terms], output_grad)
    params = flatten([x[2] for x in terms]) # List of all inputs
    return means, variance, params

def single_inv_I4_theta_stress_vae(I4_plus, I4_minus, I1s_max, output_grad_plus, output_grad_minus, should_normalize, p):
    activations = [activation_exp_I4, identity, activation_exp]
    exps = [1., 2., 2.]
    Psi_funcs = [activations[i](symb_wstar * symb_inv ** exps[i]) for i in range(len(activations))]
    terms_plus = [single_term_stress_vae(Psi_func, I4_plus, I1s_max, should_normalize=should_normalize, p=p) for Psi_func in Psi_funcs]
    params = [x[2] for x in terms_plus] # List of all inputs
    terms_minus = [single_term_stress_vae(Psi_funcs[i], I4_minus, I1s_max, should_normalize=should_normalize, p=p,
                                          wstar=params[i][0], w=params[i][1], wsigma=params[i][2]) for i in range(len(Psi_funcs))]
    means = get_stress_mean([x[0] for x in terms_plus], output_grad_plus) + \
                get_stress_mean([x[0] for x in terms_minus], output_grad_minus)
    # means = get_stress_mean([x[0] for x in terms_plus], output_grad_plus)

    # Add together before squaring for contributions from same term
    stddevs = [terms_plus[i][1][:, :, np.newaxis] * output_grad_plus +
               terms_minus[i][1][:, :, np.newaxis] * output_grad_minus for i in range(len(terms_plus))]
    variance = tf.reduce_sum(tf.stack(stddevs, axis=-1) ** 2, axis=-1)
    return means, variance, flatten(params)

def get_I4_theta(I4f, I4n, I8):
    theta = np.pi / 3
    I4theta = I4f * (np.cos(theta)) ** 2 + I4n * (np.sin(theta)) ** 2 + I8 * np.sin(2 * theta)
    I4negtheta = I4f * (np.cos(theta)) ** 2 + I4n * (np.sin(theta)) ** 2 - I8 * np.sin(2 * theta)
    return I4theta, I4negtheta

def get_stress_mean(arr, output_grad):
    return tf.reduce_sum(tf.stack(arr, axis=-1), axis=-1, keepdims=True) * output_grad

def get_stress_var(arr, output_grad):
    return tf.reduce_sum(tf.stack(arr, axis=-1) ** 2, axis=-1, keepdims=True) * output_grad ** 2
def ortho_cann_3ff_model(lam_ut_all, should_normalize=True, p=1.0):
    # Compute normalizing constants
    modelFit_mode = "012" # Only 3 experiments
    # Is_max = get_max_inv_mesh(lam_ut_norepeats, modelFit_mode)
    Is_max = get_max_inv_mesh(reshape_input_output_mesh([lam_ut_all, lam_ut_all], n_exps=3), modelFit_mode)
    I4theta_max = calculate_I4theta_max(Is_max)
    # lam_ut_all = np.array(lam_ut_all).transpose((0, 2, 1))

    lam_ut_all = tf.constant(np.array(lam_ut_all).transpose())
    print(lam_ut_all.shape)
    invs = get_invs(lam_ut_all) # 300 x 2
    invs = tf.constant(invs) # convert to TF for compatibility

    I1_in = invs[:, 0] # shape is 1000 x 1
    I2_in = invs[:, 1]
    I4f_in = invs[:, 2]
    I4n_in = invs[:, 3]
    I8fn_in = invs[:, 4]
    I4theta, I4negtheta = get_I4_theta(I4f_in, I4n_in, I8fn_in)

    # Put invariants in the reference configuration (substrct 3)
    I1_ref = I1_in - 3
    I2_ref = I2_in - 3
    I4f_ref = I4f_in - 1
    I4n_ref = I4n_in - 1
    I4theta_ref = I4theta - 1
    I4negtheta_ref = I4negtheta - 1


    # Compute stress contribution of each gradient
    output_grads = get_output_grads(lam_ut_all)  # 1000 x 5 x 2
    theta = np.pi / 3
    grad_I4theta = output_grads[:, 2, :] * (np.cos(theta)) ** 2 \
              + output_grads[:, 3, :] * (np.sin(theta)) ** 2 \
              + output_grads[:, 4, :] * np.sin(2 * theta)
    grad_I4negtheta = output_grads[:, 2, :] * (np.cos(theta)) ** 2 \
              + output_grads[:, 3, :] * (np.sin(theta)) ** 2 \
              - output_grads[:, 4, :] * np.sin(2 * theta)


    I1_mean, I1_var, params1 = single_inv_stress_vae(I1_ref, Is_max[:, 0], output_grads[:, 0, :], should_normalize, p)
    I2_mean, I2_var, params2 = single_inv_stress_vae(I2_ref, Is_max[:, 1], output_grads[:, 0, :], should_normalize, p)
    I4f_mean, I4f_var, params4f = single_inv_I4_stress_vae(I4f_ref, Is_max[:, 2], output_grads[:, 0, :], should_normalize, p)
    I4n_mean, I4n_var, params4n = single_inv_I4_stress_vae(I4n_ref, Is_max[:, 4], output_grads[:, 0, :], should_normalize, p)
    I4theta_mean, I4theta_var, params4theta = single_inv_I4_theta_stress_vae(I4theta_ref, I4negtheta_ref, I4theta_max,
                                                                    grad_I4theta, grad_I4negtheta, should_normalize, p)

    extra_variance = tf.keras.Input(shape=(1,), name="extra_variance")
    params_all = params1 + params2 + params4f + params4n + params4theta + [extra_variance]

    out_mean = I1_mean + I2_mean + I4f_mean + I4n_mean + I4theta_mean
    out_var = I1_var + I2_var + I4f_var + I4n_var + I4theta_var + extra_variance[:, :, np.newaxis]

    # final shape is None x 1000 x 2
    model = keras.models.Model(inputs=params_all, outputs=[out_mean, out_var])
    return model

class CANN_VAE(Model):
    def __init__(self, n_params, latent_dim, lam_ut_all):
        super().__init__()
        self.n_params = n_params
        self.latent_dim = latent_dim
        self.lam_ut_all = lam_ut_all
        self.enc = Encoder(latent_dim)
        self.dec = Decoder(n_params + 1)
        self.flatten = Flatten()
        self.cann_model = ortho_cann_3ff_model(lam_ut_all)


    def decode(self, latent):
        params = self.dec(latent)
        means, variances = self.cann_model(tf.split(params, self.n_params, axis=1))
        return means, variances, params


    def call(self, inputs):
        flat_in = self.flatten(inputs)
        batch = tf.shape(inputs)[0]
        normal_rand = tf.random.normal(shape=(batch, self.latent_dim), dtype=tf.float32)
        enc_out = self.enc(flat_in)
        latent = enc_out[:, 0:self.latent_dim] + normal_rand * tf.exp(enc_out[:, self.latent_dim:]) # Reparameterization trick
        # print(latent)
        params = self.dec(latent) * 0.01
        # print(params)
        params_split = tf.split(params, self.n_params + 1, axis=1)
        means, variances = self.cann_model(params_split)
        epsilon = 1e-6

        rec, kl = nelbo(inputs, means, variances + epsilon, enc_out)
        # self.add_loss(tf.reduce_min(variances))
        self.add_loss(tf.reduce_mean(rec + kl))
        self.add_metric(tf.reduce_mean(rec), name='rec')
        self.add_metric(tf.reduce_mean(kl), name='kl')
        return means, variances, params





    def train(self, xs, ys, epochs=10, batch_size=32, validation_split=0):
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
        self.fit(xs, ys, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=callback)

class CANN_VAE_Pretraining(Model):
    def __init__(self, n_params, lam_ut_all):
        super().__init__()
        self.n_params = n_params
        self.lam_ut_all = lam_ut_all
        self.dense = Dense(n_params + 1, kernel_constraint=keras.constraints.NonNeg(), kernel_initializer=initializer_1, use_bias=False)
        self.flatten = Flatten()
        self.cann_model = ortho_cann_3ff_model(lam_ut_all)

    def call(self, inputs):

        params = self.dense(inputs)
        params_split = tf.split(params, self.n_params + 1, axis=1)
        means, variances = self.cann_model(params_split)
        epsilon = 1e-6

        # rec = rec_loss(inputs, means, variances + epsilon)
        # # self.add_loss(tf.reduce_min(variances))
        # self.add_loss(tf.reduce_mean(rec))
        # self.add_metric(tf.reduce_mean(rec), name='rec')
        # self.add_metric(tf.reduce_mean(kl), name='kl')
        return tf.stack([means, variances], axis=-1)

class CANN_VAE_Indirect(Model):
    def __init__(self, n_params, latent_dim, lam_ut_all):
        super().__init__()
        self.n_params = n_params
        self.latent_dim = latent_dim
        self.lam_ut_all = lam_ut_all
        self.enc = Encoder(latent_dim)
        self.dec = Decoder(n_params + 1)
        self.flatten = Flatten()


    def call(self, inputs):
        flat_in = self.flatten(inputs)
        batch = tf.shape(inputs)[0]
        normal_rand = tf.random.normal(shape=(batch, self.latent_dim), dtype=tf.float32)
        enc_out = self.enc(flat_in)
        latent = enc_out[:, 0:self.latent_dim] + normal_rand * tf.exp(
            enc_out[:, self.latent_dim:])  # Reparameterization trick
        # print(latent)
        params = self.dec(latent) * 0.01
        return params


def nelbo(inputs, means, variances, enc_out):
    return rec_loss(inputs, means, variances), 0 * kl_loss(enc_out) / tf.reduce_sum(tf.ones_like(inputs), axis=(1, 2))

def rec_loss(inputs, means, variances):
    squared_diff = tf.square(inputs - means) / variances
    return 0.5 * tf.reduce_mean(squared_diff + tf.math.log(2 * np.pi * variances), axis=(1, 2))

def rec_loss_trainable(y_true, y_pred):
    print(y_pred.shape)
    eps = 1e-6
    return rec_loss(y_true, y_pred[:, :, :, 0], y_pred[:, :, :, 1] + eps)
def kl_loss(enc_out):
    latent_dim = int(enc_out.shape[1] / 2)
    means =  enc_out[:, 0:latent_dim]
    log_sigma = enc_out[:, latent_dim:]
    kls = log_sigma + (1 + means ** 2) / (2 * tf.math.exp(2 * log_sigma)) - 0.5
    return tf.reduce_sum(kls, axis=-1)


# Train BCANN based on stretch and stress data provided, or load from file
# Returns trained model as well as reshaped stretch and stress inputs
def train_bcann_vae(stretches, stresses, should_train=False, independent=True):
    # Reshape stretch and stress inputs
    stretches = np.float64(stretches) # (3 x 100) x 2
    stresses = np.float64(stresses) # n x 5 x (3 x 100) x 2
    lam_ut_all = [stretches[:, k].flatten() for k in range(2)]
    # P_ut_all = [[stresses.reshape((2, -1, 500, 2))[i, :,:, k].flatten() for k in range(2)] for i in range(2)]
    P_ut_all = stresses.reshape((-1, 300, 2))

    # Define hyperparameters
    modelFit_mode = "0123456789"
    epochs = 20000
    batch_size = 100  # may want to increase? also may not matter since so many close by data points
    gamma_ss = []
    P_ss = []
    last_weights = []
    path2saveResults = '../Results'
    id = "independent" if independent else "covariance"
    Save_path = path2saveResults + f'/model_vae_{id}.h5'
    Save_weights = path2saveResults + f'/weights_vae_{id}'
    path_checkpoint = path2saveResults + f'/best_weights_vae_{id}'

    # Pretrain
    # for i in range(P_ut_all.shape[0]):
    path_checkpoint = path2saveResults + f'/best_weights_pretrain_{id}'
    model_pretrain = CANN_VAE_Pretraining(3 * 17, lam_ut_all)
    input = np.eye(P_ut_all.shape[0])

    if should_train:
        opti1 = tf.optimizers.Adam(learning_rate=0.5)
        # Note custom loss (negative log likelihood) is used
        model_pretrain.compile(optimizer=opti1, loss=rec_loss_trainable)  # don't add losses sincen already added
        # Stop early if loss doesn't decrease for 2000 epochs
        es_callback = keras.callbacks.EarlyStopping(monitor="loss", min_delta=1e-6, patience=2000,
                                                    restore_best_weights=True)

        modelckpt_callback = keras.callbacks.ModelCheckpoint(
            monitor="loss",
            filepath=path_checkpoint,
            verbose=0,
            save_weights_only=True,
            save_best_only=True,
        )

        history = model_pretrain.fit(input,
                            P_ut_all,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.0,
                            callbacks=[es_callback, modelckpt_callback],
                            shuffle=True,
                            verbose=1)

        model_pretrain.load_weights(path_checkpoint, by_name=False, skip_mismatch=False)
        model_pretrain.save_weights(Save_weights, overwrite=True)
    else:
        model_pretrain.load_weights(Save_weights)

    print(rec_loss_trainable(np.float64(P_ut_all), np.float64(model_pretrain.predict(np.float64(input)))))
    print("LOSS")
    return model_pretrain, input
    # ground_truth_params = model_pretrain.get_weights()[0] # nx52 matrix
    # CANN_VAE_Pretraining







    # Build model
    model = CANN_VAE(3 * 17, 128, lam_ut_all)

    stress_old = model.predict(P_ut_all)

    # Train model
    if should_train:
        opti1 = tf.optimizers.Adam(learning_rate=0.001)
        # Note custom loss (negative log likelihood) is used
        model.compile(optimizer=opti1) # don't add losses sincen already added
        # Stop early if loss doesn't decrease for 2000 epochs
        es_callback = keras.callbacks.EarlyStopping(monitor="loss", min_delta=1e-6, patience=2000,
                                                    restore_best_weights=True)

        modelckpt_callback = keras.callbacks.ModelCheckpoint(
            monitor="loss",
            filepath=path_checkpoint,
            verbose=0,
            save_weights_only=True,
            save_best_only=True,
        )

        # Is this data format correct ?? TODO
        history = model.fit(P_ut_all,
                                  P_ut_all,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_split=0.0,
                                  callbacks=[es_callback, modelckpt_callback],
                                  shuffle=True,
                                  verbose=1)


        model.load_weights(path_checkpoint, by_name=False, skip_mismatch=False)
        # tf.keras.models.save_model(model, Save_path, overwrite=True)
        model.save_weights(Save_weights, overwrite=True)
    else:
        model.load_weights(Save_weights)


    stress_new = model.predict(P_ut_all)
    print(stress_old)
    print(stress_new)
    return model



########## Symbolic BCANN ##############

# Compute terms for I1 or I2
def single_inv_stress_bcann(inv, term_idx, I1s_max, output_grad, should_normalize, p, alpha):
    # Iterate over 2 activations and 2 exponents (1st / 2nd power)
    activations = [identity, activation_exp]
    n_exps = 2
    # Create symbolic expressions for strain energy
    Psi_funcs = [activations[j](symb_wstar * symb_inv ** (i + 1)) for i in range(n_exps) for j in range(len(activations))]
    initializers = [initializer_1] * 4 # Initialize all inner weights with uniform distribution
    # Compute term as product of output_grad (dI/dlambda) and SingleTermStress (dPsi/dI), +0.0 ensures proper weight order
    terms = [output_grad * SingleTermStress(initializers[i], Psi_funcs[i], f"w_{term_idx + i}_", should_normalize, p, alpha, I1s_max)(inv) + 0.0 for i in range(len(Psi_funcs))]

    return terms

def single_inv_I4_stress_bcann(inv, term_idx, I1s_max, output_grad, should_normalize, p, alpha):
    # Iterate over 3 activations
    activations = [activation_exp_I4, identity, activation_exp]
    exps = [1., 2., 2.]
    # Create symbolic expressions for strain energy
    Psi_funcs = [activations[i](symb_wstar * symb_inv ** exps[i]) for i in range(len(activations))]
    initializers = [initializer_1] * 3# Initialize all inner weights with uniform distribution
    # Compute term as product of output_grad (dI/dlambda) and SingleTermStress (dPsi/dI), +0.0 ensures proper weight order
    terms = [output_grad * SingleTermStress(initializers[i], Psi_funcs[i], f"w_{term_idx + i}_", should_normalize, p, alpha, I1s_max)(inv) + 0.0 for i in range(len(Psi_funcs))]
    return terms

def single_inv_I4_theta_stress_bcann(I4_plus, I4_minus, term_idx, I1s_max, output_grad_plus, output_grad_minus, should_normalize, p, alpha):
    # Iterate over 3 activations
    activations = [activation_exp_I4, identity, activation_exp]
    exps = [1., 2., 2.]
    # Create symbolic expressions for strain energy
    Psi_funcs = [activations[i](symb_wstar * symb_inv ** exps[i]) for i in range(len(activations))]
    initializers = [initializer_1] * 3 # Initialize all inner weights with uniform distribution
    layers = [SingleTermStress(initializers[i], Psi_funcs[i], f"w_{term_idx + i}_", should_normalize, p, alpha, I1s_max) for i in range(len(Psi_funcs))]
    # Compute term as product of output_grad (dI/dlambda) and SingleTermStress (dPsi/dI), +0.0 ensures proper weight order
    # Then sum contribution from I4thetaplus and I4thetaminus
    terms = [layer(I4_plus) * output_grad_plus + layer(I4_minus) * output_grad_minus for layer in layers]
    return terms # list of pairs

# Create BCANN model
def ortho_cann_3ff_bcann(lam_ut_all, gamma_ss, P_ut_all, P_ss, modelFit_mode, alpha, should_normalize, p, independent, regularization):
    terms = 17 # 4 x I1, 4 x I2, 3 x I4w, 3 x I4s, 3 x I4theta
    # Compute scale_factor (number to multiply all outputs by to help training stability)
    # and Is_max (maximum value of each invariant, incorporated into strain energy to ensure comparable initial stability)
    lam_ut_norepeats = reshape_input_output_mesh([[x[0:500] for x in y] for y in lam_ut_all]) # stretch values for first sample
    Is_max = get_max_inv_mesh(lam_ut_norepeats, modelFit_mode) # compute maximum invariants
    P_ut_reshaped = np.array(reshape_input_output_mesh(P_ut_all)) # Reshape stress
    lam_ut_reshaped = np.array(reshape_input_output_mesh(lam_ut_all)) # Reshape stretch
    scale_factors = np.mean(P_ut_reshaped, axis=-1) * (np.max(lam_ut_reshaped, axis=-1) - np.min(lam_ut_reshaped, axis=-1)) # compute the integral of the measured stress
    scale_factor = (np.sum(scale_factors) / terms * 2)
    # Divide stress integral by number of terms and multiply by 2 so if each term has an integral of 0.5 in expectation then the stress will be the correct magnitude
    I4theta_max = calculate_I4theta_max(Is_max)


    # Create model with inputs as invariants and derivative of invariants wrt x and y stretches
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
    I4f_in = tf.keras.Input(shape=(1,), name='I4f')
    I4n_in = tf.keras.Input(shape=(1,), name='I4n')
    I8fn_in = tf.keras.Input(shape=(1,), name='I8fn')
    dI1_in = tf.keras.Input(shape=(2,), name='dI1')
    dI2_in = tf.keras.Input(shape=(2,), name='dI2')
    dI4f_in = tf.keras.Input(shape=(2,), name='dI4f')
    dI4n_in = tf.keras.Input(shape=(2,), name='dI4n')
    dI8fn_in = tf.keras.Input(shape=(2,), name='dI8fn')

    # Put invariants in the reference configuration
    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)
    I4f_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4f_in)
    I4n_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4n_in)

    # Compute I4 theta
    I4theta, I4negtheta = I4_theta()([I4f_in, I4n_in, I8fn_in])
    I4theta_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4theta)
    I4negtheta_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4negtheta)

    # Compute derivative of I4theta wrt x and y stretches
    theta = np.pi / 3
    dI4theta = dI4f_in * (np.cos(theta)) ** 2 + dI4n_in * (np.sin(theta)) ** 2  + dI8fn_in * np.sin(2 * theta)
    dI4negtheta = dI4f_in * (np.cos(theta)) ** 2 + dI4n_in * (np.sin(theta)) ** 2 - dI8fn_in * np.sin(2 * theta)

    # Compute terms
    I1_out = single_inv_stress_bcann(I1_ref, 0, Is_max[:, 0], dI1_in, should_normalize, p, alpha)
    I2_out = single_inv_stress_bcann(I2_ref, 4, Is_max[:, 1], dI2_in, should_normalize, p, alpha)
    I4f_out = single_inv_I4_stress_bcann(I4f_ref, 8, Is_max[:, 2], dI4f_in, should_normalize, p, alpha)
    I4n_out = single_inv_I4_stress_bcann(I4n_ref, 11, Is_max[:, 4], dI4n_in, should_normalize, p, alpha)
    I4theta_out = single_inv_I4_theta_stress_bcann(I4theta_ref, I4negtheta_ref, 14, I4theta_max, dI4theta, dI4negtheta, should_normalize, p, alpha)

    # Concatenate terms
    All_I_out = I1_out + I2_out + I4f_out + I4n_out + I4theta_out
    # Mean is sum of terms
    all_terms = tf.stack(All_I_out, axis=-1) * scale_factor
    mean_out = tf.reduce_sum(all_terms, axis=-1)
    # Multiply the terms by L (lower triangular) then take sum of squares, where Sigma = LL^T is the covariance matrix
    # If independent then L must be diagonal
    var_constraint = (DiagonalNonnegativeLessThanOne() if regularization else DiagonalNonnegative()) if independent else ValidCovariance()
    var_out = tf.reduce_sum(Dense(all_terms.shape[2], kernel_initializer=initializer_1, kernel_constraint=var_constraint, use_bias=False)(all_terms) ** 2, axis=-1)
    # Create model to map from invariants and invariant derivatives to stress means and variances
    P_model = keras.models.Model(inputs=[I1_in, I2_in, I4f_in, I4n_in, I8fn_in, dI1_in, dI2_in, dI4f_in, dI4n_in, dI8fn_in],
                                   outputs=[mean_out[:, 0], var_out[:, 0], mean_out[:, 1], var_out[:, 1]], name='P_model')


    # Create 0-90 model
    # Inputs are warp and shute stretch
    Stretch_w = keras.layers.Input(shape=(1,),
                                   name='Stretch_w')
    Stretch_s = keras.layers.Input(shape=(1,),
                                   name='Stretch_s')
    # Compute invariants
    I1 = keras.layers.Lambda(lambda x: x[0] ** 2 + x[1] ** 2 + 1. / (x[0] * x[1]) ** 2)([Stretch_w, Stretch_s])
    I2 = keras.layers.Lambda(lambda x: 1 / x[0] ** 2 + 1 / x[1] ** 2 + x[0] ** 2 * x[1] ** 2)(
        [Stretch_w, Stretch_s])
    I4w = keras.layers.Lambda(lambda x: x ** 2)(Stretch_w)
    I4s = keras.layers.Lambda(lambda x: x ** 2)(Stretch_s)
    I8ws = keras.layers.Lambda(lambda x: x ** 0 - 1)(Stretch_s)

    # Compute derivatives of invariants wrt warp and shute stretch
    Stretch_z = keras.layers.Lambda(lambda x: 1 / (x[0] * x[1]))([Stretch_w, Stretch_s])
    dI1 = tf.keras.layers.concatenate([2 * (Stretch_w - Stretch_z * Stretch_z / Stretch_w), 2 * (Stretch_s - Stretch_z*Stretch_z / Stretch_s)], axis=-1)
    dI2 = tf.keras.layers.concatenate([2 * (Stretch_w*Stretch_s*Stretch_s - 1/(Stretch_w*Stretch_w*Stretch_w)), 2 * (Stretch_w*Stretch_w*Stretch_s - 1/(Stretch_s*Stretch_s*Stretch_s))], axis=-1)
    dI4w = tf.keras.layers.concatenate([2 * Stretch_w, 0 * Stretch_w], axis=-1)
    dI4s = tf.keras.layers.concatenate([0 * Stretch_s, 2 * Stretch_s], axis=-1)
    dI8ws = tf.keras.layers.concatenate([0 * Stretch_w, 0 * Stretch_w], axis=-1)

    # Apply P_model to invariants and derivatives
    outputs = P_model([I1, I2, I4w, I4s, I8ws, dI1, dI2, dI4w, dI4s, dI8ws])
    model_90 = keras.models.Model(inputs=[Stretch_w, Stretch_s], outputs=outputs)

    # Create 0-90 model
    # Inputs are x and y stretch
    Stretch_x = keras.layers.Input(shape=(1,),
                                   name='Stretch_x')
    Stretch_y = keras.layers.Input(shape=(1,),
                                   name='Stretch_y')
    Stretch_z = keras.layers.Lambda(lambda x: 1 / (x[0] * x[1]))([Stretch_x, Stretch_y])

    # Compute invariants
    I1 = keras.layers.Lambda(lambda x: x[0] ** 2 + x[1] ** 2 + 1. / (x[0] * x[1]) ** 2)([Stretch_x, Stretch_y])
    I2 = keras.layers.Lambda(lambda x: 1 / x[0] ** 2 + 1 / x[1] ** 2 + x[0] ** 2 * x[1] ** 2)(
        [Stretch_x, Stretch_y])
    I4w = keras.layers.Lambda(lambda x: (x[0] ** 2 + x[1] ** 2) / 2)([Stretch_x, Stretch_y])
    I4s = keras.layers.Lambda(lambda x: (x[0] ** 2 + x[1] ** 2) / 2)([Stretch_x, Stretch_y])
    I8ws = keras.layers.Lambda(lambda x: (x[0] ** 2 - x[1] ** 2) / 2)([Stretch_x, Stretch_y])

    # Compute derivatives of invariants wrt warp and shute stretch
    dI1 = tf.keras.layers.concatenate(
        [2 * (Stretch_x - Stretch_z*Stretch_z / Stretch_x), 2 * (Stretch_y - Stretch_z * Stretch_z / Stretch_y)],
        axis=-1)
    dI2 = tf.keras.layers.concatenate([2 * (Stretch_x * Stretch_y * Stretch_y - 1 / (Stretch_x * Stretch_x * Stretch_x)),
                                 2 * (Stretch_x * Stretch_x * Stretch_y - 1 / (Stretch_y * Stretch_y * Stretch_y))],
                                axis=-1)
    dI4w = tf.keras.layers.concatenate([Stretch_x, Stretch_y], axis=-1)
    dI4s = tf.keras.layers.concatenate([Stretch_x, Stretch_y], axis=-1)
    dI8ws = tf.keras.layers.concatenate([Stretch_x, -Stretch_y], axis=-1)

    # Apply P_model to invariants and derivatives
    outputs = P_model([I1, I2, I4w, I4s, I8ws, dI1, dI2, dI4w, dI4s, dI8ws])
    model_45 = keras.models.Model(inputs=[Stretch_x, Stretch_y], outputs=outputs)

    # Combine two models
    models = [model_90, model_45]
    inputs = [model.inputs for model in models]
    outputs = [model.outputs for model in models]
    outputs = tf_stack(flatten(outputs),axis=1) # make output a single tensor so we can apply our custom loss
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    return model


# Train BCANN based on stretch and stress data provided
# Returns trained model as well as reshaped stretch and stress inputs
def train_bcanns(stretches, stresses, modelFit_mode = "0123456789abcde", should_train=False, model_type="independent", alpha_in=0):
    model_id = get_model_id(model_type, alpha_in)

    # Reshape stretch and stress inputs
    stretches = np.float64(stretches)
    stresses = np.float64(stresses) # 2 x (ns*500) x 2
    lam_ut_all = [[stretches.reshape((2, -1, 2))[i, :, k].flatten() for k in range(2)] for i in range(2)]
    P_ut_all = [[stresses.reshape((2, -1, 2))[i, :, k].flatten() for k in range(2)] for i in range(2)]

    # Define hyperparameters
    alphas =  [0] if model_type == "unregularized" else [0, alpha_in]
    ps = [1.0] if model_type == "unregularized" else [1.0 , 0.5]
    epochs = 2000
    batch_size = 1000
    gamma_ss = []
    P_ss = []
    last_weights = []

    # Iterate over regularization values
    for i in range(len(alphas)):
        path2saveResults = '../Results/' + modelFit_mode
        makeDIR(path2saveResults)
        Save_path = path2saveResults + f'/model_{model_id}.h5'
        Save_weights = path2saveResults + f'/weights_{model_id}'
        path_checkpoint = path2saveResults + f'/best_weights_{model_id}'

        if i < len(alphas) - 1 and not should_train: # If not training, skip to last alpha and load model
            continue

        # Build model
        model = ortho_cann_3ff_bcann(lam_ut_all, gamma_ss, P_ut_all, P_ss, "0123456789", alphas[i],
                                     True, ps[i], not (model_type == "correlated"), i > 0)


        # If not the first iteration, set initial weights to be final weights from previous iteration
        if i > 0 and should_train:
            model.set_weights(last_weights)

        # Load training data
        model_given, input_train, output_train, sample_weights = traindata(modelFit_mode, model, lam_ut_all, P_ut_all,
                                                                           model, gamma_ss, P_ss, model, 0)

        if should_train:
            # Train model
            model_given, history, weight_hist_arr = Compile_and_fit_bcann(model_given, input_train, output_train, epochs,
                                                                    path_checkpoint,
                                                                    sample_weights, batch_size)

            model_given.load_weights(path_checkpoint, by_name=False, skip_mismatch=False)
            tf.keras.models.save_model(model, Save_path, overwrite=True)
            model_given.save_weights(Save_weights, overwrite=True)
            last_weights = model_given.get_weights()

            if i < len(ps) - 1: # If not last iteration, update last_weights so it is correct for the next value of p
                p_ratio = ps[i + 1] / ps[i]
                last_weights = [last_weights[i] ** (p_ratio if (i % 2 == 1) else 1.0) for i in range(len(last_weights))]

        else:
            model.load_weights(Save_weights)

    # Uncomment this to display names for each weight
    # names = [weight.name for layer in model_given.layers for weight in layer.weights]
    # print(names)


    model_weights = model.get_weights()
    nonzero_terms = sum([x > 0 for x in model_weights[1::2]])
    print(f"Nonzero Terms: {nonzero_terms}") # Print number of nonzero terms to determine effect of regularization

    return model_given, lam_ut_all, P_ut_all

def train_ensemble(stretches, stresses, modelFit_mode = "0123456789abcde", should_train=False):
    # Reshape stretch and stress inputs
    stretches = np.float64(stretches)
    stresses = np.float64(stresses) # 2 x (ns x 5 x 100) x 2
    n_samples = 5
    lam_ut_all = [[[stretches.reshape((n_samples, -1, 2))[j, :, k].flatten() for k in range(2)] for i in range(2)] for j in range(n_samples)]
    P_ut_all = [[[stresses.reshape((2, n_samples, -1, 2))[i, j, :, k].flatten() for k in range(2)] for i in range(2)] for j in range(n_samples)]

    # Define hyperparameters
    regularization = False
    alphas = [0, 0.3] if regularization else [0]
    ps = [1.0 , 0.5] if regularization else [1.0]
    epochs = 2000 # TODO change back
    batch_size = 1000 # may want to increase? also may not matter since so many close by data points
    gamma_ss = []
    P_ss = []
    last_weights = []
    Stress_predict_UT = []

    for sample_idx in range(n_samples):
        # Iterate over regularization values
        for i in range(len(alphas)):
            ##########
            # Use ogden model with exponents and gains as weights
            if i < len(alphas) - 1 and not should_train: # If not training, skip to last alpha and load model
                continue

            # Build model
            Psi_model, terms = ortho_cann_3ff(lam_ut_all[sample_idx], gamma_ss, P_ut_all[sample_idx], P_ss, modelFit_mode, alphas[i], True, ps[i])


            # If not the first iteration, set initial weights to be final weights from previous iteration
            if i > 0 and should_train:
                Psi_model.set_weights(last_weights)

            # Create complete model architecture
            model_UT, model_SS, Psi_model, model = modelArchitecture("mesh", Psi_model)

            # Load training data
            model_given, input_train, output_train, sample_weights = traindata(modelFit_mode, model_UT, lam_ut_all[sample_idx], P_ut_all[sample_idx],
                                                                               model_SS, gamma_ss, P_ss, model, 0, is_bcann=True, should_normalize=True)


            # model_given.summary(print_fn=print)
            path2saveResults = '../Results/' + modelFit_mode + f"/ensemble_{sample_idx}"
            makeDIR(path2saveResults)
            Save_path = path2saveResults + f'/model.h5'
            Save_weights = path2saveResults + f'/weights'
            path_checkpoint = path2saveResults + f'/best_weights'

            # Train model
            if should_train:

                model_given, history, weight_hist_arr = Compile_and_fit(model_given, input_train, output_train, epochs,
                                                                        path_checkpoint,
                                                                        sample_weights, batch_size)

                model_given.load_weights(path_checkpoint, by_name=False, skip_mismatch=False)
                tf.keras.models.save_model(Psi_model, Save_path, overwrite=True)
                Psi_model.save_weights(Save_weights, overwrite=True)
                last_weights = model_given.get_weights()

                if i < len(ps) - 1:  # If not last iteration, update last_weights so it is correct for the next value of p
                    p_ratio = ps[i + 1] / ps[i]
                    last_weights = [last_weights[i] ** (p_ratio if (i % 2 == 1) else 1.0) for i in
                                    range(len(last_weights))]
            else:
                Psi_model.load_weights(Save_weights)


        Stress_predict_UT.append(model_UT.predict(lam_ut_all))

    Stress_predict_all = np.squeeze(np.array(Stress_predict_UT)).transpose((0, 1, 3, 2)).reshape((5, 2, 5, -1, 2))
    Stress_predict_mean = np.mean(Stress_predict_all, axis=0)
    Stress_predict_std = np.std(Stress_predict_all, axis=0)

    return Stress_predict_mean, Stress_predict_std, lam_ut_all, P_ut_all

def format_2sigfigs(x):
    return f"{x:.0f}" if x >= 100 else f"{x:.2g}"
# Display strain energy equation and weights (with variances)
def disp_equation_weights_bcann(weights, lam_ut_all, P_ut_all, modelFit_mode):
    sf, terms, weights_var = ortho_bcann_4ff_symbolic(weights, lam_ut_all, P_ut_all, modelFit_mode)

    n_mus = 0
    n_abs = 0
    eqn = ""
    # Iterate over terms
    for term, w_var in zip(terms, weights_var):
        term = term.item()
        if term == 0: # Skip any zero terms
            continue
        gain = term.args[0] * sf * 2 ## * 2 accounts for 1/2 we add later
        if term.args[1].func == sympy.core.add.Add: # Print exponential weights
            try:
                exponent = term.args[1].args[-1].args[0][0].args[0]
            except:
                # print(term)
                continue
            n_abs += 1
            gain *= exponent
            std_dev = float(gain * np.sqrt(w_var))
            print(f"$a_{n_abs} = {format_2sigfigs(gain)}\\pm{format_2sigfigs(std_dev)}" + "\\text{ kPa}$\\newline")
            print(f"$b_{n_abs} = {format_2sigfigs(exponent)}$\\newline")
            term_str = f"a_{n_abs}(" + str(term.args[1]) + f") / b_{n_abs}"
            term_str = re.sub(r'\d+\.\d\d\d+', f"b_{n_abs}", term_str)

        else: # Print non-exponential weights
            n_mus += 1
            term_str = f"\mu_{n_mus}" + str(term.args[1])
            std_dev = float(gain * np.sqrt(w_var))
            print(f"$\mu_{n_mus} = {format_2sigfigs(gain)}\\pm{format_2sigfigs(std_dev)}" +  "\\text{ kPa}$\\newline")


        # Perform substitutions so it renders correctly
        term_str = term_str.replace("**", "^")
        term_str = term_str.replace("I1", "(I_1 - 3)")
        term_str = term_str.replace("I2", "(I_2 - 3)")
        term_str = term_str.replace("I4w", "(I_{4w} - 1)")
        term_str = term_str.replace("I4s", "(I_{4s} - 1)")
        term_str = term_str.replace("1.0", "1")
        term_str = term_str.replace("exp", "\exp")
        term_str = term_str.replace("*", "")
        term_str = term_str.replace("[", "")
        term_str = term_str.replace("]", "")
        term_str = "&+&\\frac{1}{2}" + term_str
        term_str += "\n"
        # Append term to equation
        if "I4theta" in term_str:
            eqn += term_str.replace("I4theta", "(I_{4s_I} - 1)")
            eqn += term_str.replace("I4theta", "(I_{4s_{II}} - 1)")

        else:
            eqn += term_str

    # Print equation
    print("\psi &=& " + eqn[3:])

# Construct symbolic expression for BCANN
def ortho_bcann_4ff_symbolic(weights, lam_ut_all, P_ut_all, modelFit_mode):
    terms = len(weights) // 2

    # Compute scale factor and Imaxes
    Is_max = get_max_inv_mesh(reshape_input_output_mesh(lam_ut_all), modelFit_mode)
    P_ut_reshaped = np.array(reshape_input_output_mesh(P_ut_all))
    lam_ut_reshaped = np.array(reshape_input_output_mesh(lam_ut_all))
    scale_factors = np.mean(P_ut_reshaped, axis=-1) * (np.max(lam_ut_reshaped, axis=-1) - np.min(lam_ut_reshaped, axis=-1)) # * 10 because there are 5 loading configurations and 2
    scale_factor = np.sum(scale_factors) / terms * 2
    I4theta_max = calculate_I4theta_max(Is_max)

    # Separate weights into CANN weights and variances
    weights_mean = weights[:-1]# get only wstar and w
    weights_var = tf.reduce_sum(tf.square(weights[-1]), axis=-1).numpy().tolist() # get only wsigma
    print(weights)
    # Create list of all terms
    output = SingleInvNet_symbolic(Symbol("I1"), weights_mean[0:8],  I1s_max=(Is_max[:, 0]))
    output += SingleInvNet_symbolic(Symbol("I2"), weights_mean[8:16],  I1s_max=(Is_max[:, 1]))
    output += SingleInvNetI4_symbolic(Symbol("I4w"), weights_mean[16:22],  I1s_max=(Is_max[:, 2]))
    output += SingleInvNetI4_symbolic(Symbol("I4s"), weights_mean[22:28],  I1s_max=(Is_max[:, 4]))
    output += SingleInvNetI4_symbolic(Symbol("I4theta"), weights_mean[28:],  I1s_max=(I4theta_max))

    # Output scale factor, list of terms, and list of wsigmas
    return scale_factor, output, weights_var



# Perform training of model, return fit model, training history, and weight history
def Compile_and_fit_bcann(model_given, input_train, output_train, epochs, path_checkpoint, sample_weights, batch_size):

    opti1 = tf.optimizers.Adam(learning_rate=0.001)
    # Note custom loss (negative log likelihood) is used
    model_given.compile(loss=NLL,
                        optimizer=opti1,
                        metrics=[NLL])
    # Stop early if loss doesn't decrease for 2000 epochs
    es_callback = keras.callbacks.EarlyStopping(monitor="loss", min_delta=1e-6, patience=2000,
                                                restore_best_weights=True)

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="loss",
        filepath=path_checkpoint,
        verbose=0,
        save_weights_only=True,
        save_best_only=True,
    )
    # Create array to store weight history
    weight_hist_arr = []
    # Create callback to append model weights to weight_hist_arr every epoch
    weight_hist_callback = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs:
        weight_hist_arr.append(model_given.get_weights()))

    # Reshape output_train to be a single tf array
    output_temp = tf.keras.backend.stack(flatten(output_train) + flatten(sample_weights),axis=1) #
    output_temp = tf.cast(output_temp, tf.float32)

    history = model_given.fit(input_train,
                              output_temp,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_split=0.0,
                              callbacks=[es_callback, modelckpt_callback, weight_hist_callback],
                              shuffle=True,
                              verbose=1)

    return model_given, history, weight_hist_arr

# Compute negative log likelihood
def NLL(y_true, y_pred):
    eps = 1e-6
    # Separate mean and variance
    means = y_pred[:, 0::2]
    vars = y_pred[:, 1::2]

    y_wgts = y_true[:, 4:]
    y_true = y_true[:, 0:4]
    # Compute negative log likelihood for a normal distribution
    errors = 0.5 * (tf.math.log(2 * np.pi * (vars + eps)) + tf.math.square(y_true - means) / (vars + eps)) * y_wgts
    return tf.reduce_sum(errors, axis=1)

# Constrains weights to be between 0.0 and 1.0
class NonNegLessThanOne(keras.constraints.Constraint):
    def __call__(self, w):
        return tf.clip_by_value(w, 0.0, 1.0)


class DiagonalNonnegativeLessThanOne(keras.constraints.Constraint):
    """Constrains the weights to be diagonal and nonnegative
    """
    def __call__(self, w):
        N = tf.keras.backend.int_shape(w)[-1]
        m = tf.eye(N)
        return m * tf.clip_by_value(w, 0, 1.0)

class DiagonalNonnegative(keras.constraints.Constraint):
    """Constrains the weights to be diagonal and nonnegative
    """
    def __call__(self, w):
        N = tf.keras.backend.int_shape(w)[-1]
        m = tf.eye(N)
        return m * tf.clip_by_value(w, 0, np.inf)

class ValidCovariance(keras.constraints.Constraint):
    def __call__(self, w):
        lower = tf.linalg.band_part(w, -1, 0)
        norms = tf.math.sqrt(
            tf.reduce_sum(tf.square(lower), axis=-1, keepdims=True)
        )
        desired_norms = tf.clip_by_value(norms, 0, 1.0)
        return lower * (desired_norms / (tf.keras.backend.epsilon() + norms))


