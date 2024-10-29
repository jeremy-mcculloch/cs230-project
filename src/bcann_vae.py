import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Layer
from keras import Model
import tensorflow_probability as tfp
import keras
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
import matplotlib.pyplot as plt

from src.utils import *
import sympy
from sympy import Symbol, lambdify, diff, exp


# Takes raw data as input, outputs means and variances of latent variables
class Encoder(Layer):
    def __init__(self, latent_dim):
        super().__init__()
        self.fcnet = Sequential()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(128, activation='relu')
        self.dense3 = Dense(128, activation='relu')
        self.dense4 = Dense(latent_dim * 2) # mean and log std dev
        self.fcnet.add(self.dense1)
        self.fcnet.add(self.dense2)
        self.fcnet.add(self.dense3)
        self.fcnet.add(self.dense4)

    def call(self, inputs, *args, **kwargs):
        return self.fcnet(inputs)

    def get_config(self):
        return super(Encoder, self).get_config()

# Takes latent variables as input, outputs parameters for BCANN
class Decoder(Layer):
    def __init__(self, n_params):
        super().__init__()
        self.fcnet = Sequential()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(256, activation='relu')
        self.dense3 = Dense(128, activation='relu')
        self.dense4 = Dense(n_params)
        self.fcnet.add(self.dense1)
        self.fcnet.add(self.dense2)
        self.fcnet.add(self.dense3)
        self.fcnet.add(self.dense4)


    def call(self, inputs, *args, **kwargs):
        return self.fcnet(inputs)

    def get_config(self):
        return super(Encoder, self).get_config()

symb_wstar = Symbol("wstar")
symb_inv = Symbol("inv")
initializer_1 = tf.keras.initializers.RandomUniform(minval=0., maxval=1)

def get_stress_expression(Psi_expr, I1s_max, should_normalize=True):
    I1s_max = tf.cast(I1s_max, tf.float32)

    I1maxmax = np.max(I1s_max) if should_normalize else 1.0
    P_expr = diff(Psi_expr, symb_inv)
    Psi_func = lambdify([symb_inv, symb_wstar], Psi_expr, "tensorflow")
    P_func = lambdify([symb_inv, symb_wstar], P_expr, "tensorflow")
    epsilon = 1e-8
    P_out = lambda inv, wstar : 1 / I1maxmax * P_func(inv / I1maxmax, wstar) / (tf.reduce_sum(Psi_func(I1s_max / I1maxmax, wstar)) + epsilon)
    out_func = P_out if should_normalize else P_func
    return out_func

class SingleTermStress(keras.layers.Layer):
    def __init__(self, kernel_initializer, Psi_expr, weight_name, should_normalize, p, alpha, I1s_max):
        super().__init__()
        self.P_func = get_stress_expression(Psi_expr, I1s_max, should_normalize)
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
        # print(input_shape)
        # Create weight for power
        self.wstar = self.add_weight(
            shape=(1, ),
            initializer=self.kernel_initializer,
            constraint=keras.constraints.NonNeg(),
            trainable=True,
            name=self.weight_name + '1'
        )
        self.w = self.add_weight(
            shape=(1,),
            initializer=initializer_1,
            constraint= keras.constraints.NonNeg(),
            regularizer=keras.regularizers.l1(self.alpha),
            trainable=True,
            name=self.weight_name + '2'
        )
        self.wsigma = self.add_weight(
            shape=(1,),
            initializer=initializer_1,
            constraint=keras.constraints.NonNeg(),
            trainable=True,
            name=self.weight_name + '3'
        )
    def call(self, inputs):
        mean = self.w ** (1 / self.p) * self.P_func(inputs, self.wstar)
        stddev = (mean * self.wsigma)
        return mean, stddev

def single_term_stress_vae(Psi_expr, invs, I1s_max, should_normalize=True, p=1.0, w=tf.keras.Input(shape=(1,)), wstar=tf.keras.Input(shape=(1,)), wsigma=tf.keras.Input(shape=(1,))):
    P_func = get_stress_expression(Psi_expr, I1s_max, should_normalize)
    mean =  w ** (1 / p) * P_func(invs, wstar)
    stddev = (mean * wsigma)
    return mean, stddev, [wstar, w, wsigma]

def identity(x):
    return x

def activation_exp(x):
    return exp(x) - 1.0
def activation_exp_I4(x):
    return exp(x) - x - 1.0

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
    terms_minus = [single_term_stress_vae(Psi_funcs[i], I4_plus, I1s_max, should_normalize=should_normalize, p=p,
                                          wstar=params[i][0], w=params[i][1], wsigma=params[i][2]) for i in range(len(Psi_funcs))]
    means = get_stress_mean([x[0] for x in terms_plus], output_grad_plus) + \
                get_stress_mean([x[0] for x in terms_minus], output_grad_minus)

    # Add together before squaring for contributions from same term
    stddevs = [terms_plus[i][1][:, :, np.newaxis] * output_grad_plus +
               terms_minus[i][1][:, :, np.newaxis] * output_grad_minus for i in range(len(terms_plus))]
    variance = tf.reduce_sum(tf.stack(stddevs, axis=-1) ** 2, axis=-1)
    return means, variance, params

def get_I4_theta(I4f, I4n, I8):
    theta = np.pi / 3
    I4theta = I4f * (np.cos(theta)) ** 2 + I4n * (np.sin(theta)) ** 2 + I8 * np.sin(2 * theta)
    I4negtheta = I4f * (np.cos(theta)) ** 2 + I4n * (np.sin(theta)) ** 2 - I8 * np.sin(2 * theta)
    return I4theta, I4negtheta

from src.CANN.util_functions import reshape_input_output_mesh
from src.CANN.models import get_max_inv_mesh, calculate_I4theta_max

def get_stress_mean(arr, output_grad):
    return tf.reduce_sum(tf.stack(arr, axis=-1), axis=-1, keepdims=True) * output_grad

def get_stress_var(arr, output_grad):
    return tf.reduce_sum(tf.stack(arr, axis=-1) ** 2, axis=-1, keepdims=True) * output_grad ** 2
def ortho_cann_3ff_model(lam_ut_all, should_normalize=True, p=1.0):
    # Compute normalizing constants
    modelFit_mode = "0123456789"
    Is_max = get_max_inv_mesh(reshape_input_output_mesh(lam_ut_all), modelFit_mode)
    I4theta_max = calculate_I4theta_max(Is_max)



    invs = get_invs(lam_ut_all)
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
    grad_I4theta = output_grads[:, 2, np.newaxis, :] * (np.cos(theta)) ** 2 \
              + output_grads[:, 3, np.newaxis, :] * (np.sin(theta)) ** 2 \
              + output_grads[:, 4, np.newaxis, :] * np.sin(2 * theta)
    grad_I4negtheta = output_grads[:, 2, np.newaxis, :] * (np.cos(theta)) ** 2 \
              + output_grads[:, 3, np.newaxis, :] * (np.sin(theta)) ** 2 \
              - output_grads[:, 4, np.newaxis, :] * np.sin(2 * theta)

    I1_mean, I1_var, params1 = single_inv_stress_vae(I1_ref, Is_max[:, 0], output_grads[:, 0, :], should_normalize, p)
    I2_mean, I2_var, params2 = single_inv_stress_vae(I2_ref, Is_max[:, 1], output_grads[:, 0, :], should_normalize, p)
    I4f_mean, I4f_var, params4f = single_inv_I4_stress_vae(I4f_ref, Is_max[:, 2], output_grads[:, 0, :], should_normalize, p)
    I4n_mean, I4n_var, params4n = single_inv_I4_stress_vae(I4n_ref, Is_max[:, 4], output_grads[:, 0, :], should_normalize, p)
    I4theta_mean, I4theta_var, params4theta = single_inv_I4_theta_stress_vae(I4theta_ref, I4negtheta_ref, Is_max[:, 4],
                                                                    grad_I4theta, grad_I4negtheta, should_normalize, p)

    params_all = params1 + params2 + params4f + params4n + params4theta
    out_mean = I1_mean + I2_mean + I4f_mean + I4n_mean + I4theta_mean
    out_var = I1_var + I2_var + I4f_var + I4n_var + I4theta_var

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
        self.dec = Decoder(n_params)
        self.flatten = Flatten()
        self.cann_model = ortho_cann_3ff_model(self.lam_ut_all)


    def decode(self, latent):
        params = self.dec(latent)
        means, variances = self.cann_model(tf.split(params, self.n_params, axis=1))
        return means, variances, params


    def call(self, inputs):
        flat_in = self.flatten(inputs)
        batch = tf.shape(inputs)[0]
        normal_rand = tf.random.normal(shape=(batch, self.latent_dim), dtype=tf.float64)
        enc_out = self.enc(flat_in)
        latent = enc_out[:, 0:self.latent_dim] + normal_rand * tf.exp(enc_out[:, self.latent_dim:]) # Reparameterization trick
        params = self.dec(latent)
        means, variances = self.cann_model(tf.split(params, self.n_params, axis=1))
        rec, kl = nelbo(inputs, means, variances, enc_out)
        self.add_loss(tf.reduce_mean(rec + kl))
        self.add_metric(tf.reduce_mean(rec), name='rec')
        self.add_metric(tf.reduce_mean(kl), name='kl')
        return means, variances, params





    def train(self, xs, ys, epochs=10, batch_size=32, validation_split=0):
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
        self.fit(xs, ys, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=callback)


def nelbo(inputs, means, variances, enc_out):
    return rec_loss(inputs, means, variances), kl_loss(enc_out) / tf.reduce_sum(tf.ones_like(inputs), axis=(1, 2))

def rec_loss(inputs, means, variances):
    squared_diff = tf.square(inputs - means) / variances
    return 0.5 * tf.reduce_mean(squared_diff + tf.math.log(variances), axis=(1, 2))

def kl_loss(enc_out):
    latent_dim = int(enc_out.shape[1] / 2)
    means =  enc_out[:, 0:latent_dim]
    log_sigma = enc_out[:, latent_dim:]
    kls = log_sigma + (1 + means ** 2) / (2 * tf.math.exp(2 * log_sigma)) - 0.5
    return tf.reduce_sum(kls, axis=-1)



########## Symbolic BCANN ##############

#(I1_ref, 0, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 0]))
def single_inv_stress_bcann(inv, term_idx, I1s_max, output_grad, should_normalize, p, alpha):
    activations = [identity, activation_exp]
    n_exps = 2
    Psi_funcs = [activations[j](symb_wstar * symb_inv ** (i + 1)) for i in range(n_exps) for j in range(len(activations))]
    initializers = [initializer_1] * 4
    terms = [[output_grad * x + 0.0 for x in SingleTermStress(initializers[i], Psi_funcs[i], f"w_{term_idx + i}_", should_normalize, p, alpha, I1s_max)(inv)] for i in range(len(Psi_funcs))]

    return terms

def single_inv_I4_stress_bcann(inv, term_idx, I1s_max, output_grad, should_normalize, p, alpha):
    activations = [activation_exp_I4, identity, activation_exp]
    exps = [1., 2., 2.]
    Psi_funcs = [activations[i](symb_wstar * symb_inv ** exps[i]) for i in range(len(activations))]
    initializers = [initializer_1] * 3
    terms = [[output_grad * x + 0.0 for x in SingleTermStress(initializers[i], Psi_funcs[i], f"w_{term_idx + i}_", should_normalize, p, alpha, I1s_max)(inv)] for i in range(len(Psi_funcs))]
    return terms

def single_inv_I4_theta_stress_bcann(I4_plus, I4_minus, term_idx, I1s_max, output_grad_plus, output_grad_minus, should_normalize, p, alpha):
    activations = [activation_exp_I4, identity, activation_exp]
    exps = [1., 2., 2.]
    Psi_funcs = [activations[i](symb_wstar * symb_inv ** exps[i]) for i in range(len(activations))]
    initializers = [initializer_1] * 3
    layers = [SingleTermStress(initializers[i], Psi_funcs[i], f"w_{term_idx + i}_", should_normalize, p, alpha, I1s_max) for i in range(len(Psi_funcs))]
    terms = [[x[0] * output_grad_plus + x[1] * output_grad_minus for x in zip(layer(I4_plus),layer(I4_minus))] for layer in layers]
    return terms # list of pairs

from src.CANN.models import I4_theta
def ortho_cann_3ff_bcann(lam_ut_all, gamma_ss, P_ut_all, P_ss, modelFit_mode, alpha, should_normalize, p, kevins_version, two_term=False, terms=[]):
    # This stuff seems fishy, make sure its working right / robustly to different input formats
    Is_max = get_max_inv_mesh(reshape_input_output_mesh(lam_ut_all), modelFit_mode)
    P_ut_reshaped = np.array(reshape_input_output_mesh(P_ut_all))
    lam_ut_reshaped = np.array(reshape_input_output_mesh(lam_ut_all))
    scale_factors = np.mean(P_ut_reshaped, axis=-1) * (np.max(lam_ut_reshaped, axis=-1) - np.min(lam_ut_reshaped, axis=-1)) # * 10 because there are 5 loading configurations and 2
    scale_factor = 1.0 if kevins_version else np.sum(scale_factors)
    I4theta_max = calculate_I4theta_max(Is_max)


    # Inputs defined
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
    I4theta, I4negtheta = I4_theta()([I4f_in, I4n_in, I8fn_in])

    # Put invariants in the reference configuration (substrct 3)
    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)
    I4f_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4f_in)
    I4n_ref = keras.layers.Lambda(lambda x: (abs(x)-1.0))(I4n_in)
    I4theta_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4theta)
    I4negtheta_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4negtheta)

    # output_grads = get_output_grads_batch(lam_ut_all)  # None x 5 x 2
    # Compute gradients
    theta = np.pi / 3
    dI4theta = dI4f_in * (np.cos(theta)) ** 2 + dI4n_in * (np.sin(theta)) ** 2  + dI8fn_in * np.sin(2 * theta)
    dI4negtheta = dI4f_in * (np.cos(theta)) ** 2 + dI4n_in * (np.sin(theta)) ** 2 - dI8fn_in * np.sin(2 * theta)

    I1_out = single_inv_stress_bcann(I1_ref, 0, Is_max[:, 0], dI1_in, should_normalize, p, alpha)
    I2_out = single_inv_stress_bcann(I2_ref, 4, Is_max[:, 1], dI2_in, should_normalize, p, alpha)
    I4f_out = single_inv_I4_stress_bcann(I4f_ref, 8, Is_max[:, 2], dI4f_in, should_normalize, p, alpha)
    I4n_out = single_inv_I4_stress_bcann(I4n_ref, 11, Is_max[:, 4], dI4n_in, should_normalize, p, alpha)
    I4theta_out = single_inv_I4_theta_stress_bcann(I4theta_ref, I4negtheta_ref, 14, I4theta_max, dI4theta, dI4negtheta, should_normalize, p, alpha)

    All_I_out = I1_out + I2_out + I4f_out + I4n_out + I4theta_out
    mean_out = tf.reduce_sum(tf.stack([x[0] for x in All_I_out], axis=-1), axis=-1) * scale_factor
    var_out = tf.reduce_sum(tf.stack([x[1] for x in All_I_out], axis=-1) ** 2, axis=-1) * scale_factor ** 2
    P_model = keras.models.Model(inputs=[I1_in, I2_in, I4f_in, I4n_in, I8fn_in, dI1_in, dI2_in, dI4f_in, dI4n_in, dI8fn_in],
                                   outputs=[mean_out[:, 0], var_out[:, 0], mean_out[:, 1], var_out[:, 1]], name='P_model')


    Stretch_w = keras.layers.Input(shape=(1,),
                                   name='Stretch_w')
    Stretch_s = keras.layers.Input(shape=(1,),
                                   name='Stretch_s')
    I1 = keras.layers.Lambda(lambda x: x[0] ** 2 + x[1] ** 2 + 1. / (x[0] * x[1]) ** 2)([Stretch_w, Stretch_s])
    I2 = keras.layers.Lambda(lambda x: 1 / x[0] ** 2 + 1 / x[1] ** 2 + x[0] ** 2 * x[1] ** 2)(
        [Stretch_w, Stretch_s])
    I4w = keras.layers.Lambda(lambda x: x ** 2)(Stretch_w)
    I4s = keras.layers.Lambda(lambda x: x ** 2)(Stretch_s)
    I8ws = keras.layers.Lambda(lambda x: x ** 0 - 1)(Stretch_s)
    Stretch_z = keras.layers.Lambda(lambda x: 1 / (x[0] * x[1]))([Stretch_w, Stretch_s])
    # Need to make these 2d
    dI1 = tf.keras.layers.concatenate([2 * (Stretch_w - Stretch_z * Stretch_z / Stretch_w), 2 * (Stretch_s - Stretch_z*Stretch_z / Stretch_s)], axis=-1)
    dI2 = tf.keras.layers.concatenate([2 * (Stretch_w*Stretch_s*Stretch_s - 1/(Stretch_w*Stretch_w*Stretch_w)), 2 * (Stretch_w*Stretch_w*Stretch_s - 1/(Stretch_s*Stretch_s*Stretch_s))], axis=-1)
    dI4w = tf.keras.layers.concatenate([2 * Stretch_w, 0 * Stretch_w], axis=-1)
    dI4s = tf.keras.layers.concatenate([0 * Stretch_s, 2 * Stretch_s], axis=-1)
    dI8ws = tf.keras.layers.concatenate([0 * Stretch_w, 0 * Stretch_w], axis=-1)

    outputs = P_model([I1, I2, I4w, I4s, I8ws, dI1, dI2, dI4w, dI4s, dI8ws])
    model_90 = keras.models.Model(inputs=[Stretch_w, Stretch_s], outputs=outputs)

    # 45 degree offset

    Stretch_x = keras.layers.Input(shape=(1,),
                                   name='Stretch_x')
    Stretch_y = keras.layers.Input(shape=(1,),
                                   name='Stretch_y')
    Stretch_z = keras.layers.Lambda(lambda x: 1 / (x[0] * x[1]))([Stretch_x, Stretch_y])

    I1 = keras.layers.Lambda(lambda x: x[0] ** 2 + x[1] ** 2 + 1. / (x[0] * x[1]) ** 2)([Stretch_x, Stretch_y])
    I2 = keras.layers.Lambda(lambda x: 1 / x[0] ** 2 + 1 / x[1] ** 2 + x[0] ** 2 * x[1] ** 2)(
        [Stretch_x, Stretch_y])
    I4w = keras.layers.Lambda(lambda x: (x[0] ** 2 + x[1] ** 2) / 2)([Stretch_x, Stretch_y])
    I4s = keras.layers.Lambda(lambda x: (x[0] ** 2 + x[1] ** 2) / 2)([Stretch_x, Stretch_y])
    I8ws = keras.layers.Lambda(lambda x: (x[0] ** 2 - x[1] ** 2) / 2)([Stretch_x, Stretch_y])

    dI1 = tf.keras.layers.concatenate(
        [2 * (Stretch_x - Stretch_z*Stretch_z / Stretch_x), 2 * (Stretch_y - Stretch_z * Stretch_z / Stretch_y)],
        axis=-1)
    dI2 = tf.keras.layers.concatenate([2 * (Stretch_x * Stretch_y * Stretch_y - 1 / (Stretch_x * Stretch_x * Stretch_x)),
                                 2 * (Stretch_x * Stretch_x * Stretch_y - 1 / (Stretch_y * Stretch_y * Stretch_y))],
                                axis=-1)
    dI4w = tf.keras.layers.concatenate([Stretch_x, Stretch_y], axis=-1)
    dI4s = tf.keras.layers.concatenate([Stretch_x, Stretch_y], axis=-1)
    dI8ws = tf.keras.layers.concatenate([Stretch_x, -Stretch_y], axis=-1)

    outputs = P_model([I1, I2, I4w, I4s, I8ws, dI1, dI2, dI4w, dI4s, dI8ws])

    model_45 = keras.models.Model(inputs=[Stretch_x, Stretch_y], outputs=outputs)

    models = [model_90, model_45]
    inputs = [model.inputs for model in models]
    outputs = [model.outputs for model in models]

    outputs = tf_stack(flatten(outputs),axis=1) # change shape so loss actually works
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    return model

