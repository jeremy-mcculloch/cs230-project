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
from src.CANN.util_functions import traindata, Compile_and_fit
from src.utils import *
from src.CANN.models import ortho_cann_3ff

global_scale_factor = 1e6

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

class Decoder(Layer):
    def __init__(self, n_params, stddev_output):
        super().__init__()
        self.stddev_ouput = stddev_output
        self.fcnet = Sequential()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(256, activation='relu')
        self.dense3 = Dense(128, activation='relu')
        self.dense4 = Dense(n_params + (1 if stddev_output else 0), activation='exponential')
        self.fcnet.add(self.dense1)
        self.fcnet.add(self.dense2)
        self.fcnet.add(self.dense3)
        self.fcnet.add(self.dense4)


    def call(self, inputs, *args, **kwargs):
        return self.fcnet(inputs)

    def get_config(self):
        return super(Encoder, self).get_config()


class CANN_VAE(Model):
    def __init__(self, n_params, latent_dim, lam_ut_all, stddev=-1):
        super().__init__()
        self.n_params = n_params
        self.latent_dim = latent_dim
        self.lam_ut_all = lam_ut_all
        self.enc = Encoder(latent_dim)
        self.stddev_output = stddev < 0
        self.stddev = stddev
        self.dec = Decoder(n_params, self.stddev_output)
        self.flatten = Flatten()
        self.cann_model = ortho_cann_3ff_model(self.lam_ut_all)
        self.stdev_scaling = 10.0


    def decode(self, latent):
        dec_out = self.dec(latent)
        params = dec_out[:, 0:self.n_params]
        stdev = self.stdev_scaling * dec_out[:, -1] if self.stddev_output else self.stddev
        stresses = self.cann_model(tf.split(params, self.n_params, axis=1))
        stresses = tf.reduce_sum(stresses, axis=2)
        return stresses, params, stdev


    def call(self, inputs):
        flat_in = self.flatten(inputs)
        batch = tf.shape(inputs)[0]
        normal_rand = tf.random.normal(shape=(batch, self.latent_dim), dtype=tf.float64)
        enc_out = self.enc(flat_in)
        latent = enc_out[:, 0:self.latent_dim] + normal_rand * tf.exp(enc_out[:, self.latent_dim:]) # Reparameterization trick
        dec_out = self.dec(latent)
        params = dec_out[:, 0:self.n_params]
        stdev = self.stdev_scaling * dec_out[:, -1] if self.stddev_output else self.stddev
        stresses = self.cann_model(tf.split(params, self.n_params, axis=1))

        stresses = tf.reduce_sum(stresses, axis=2)
        rec, kl = nelbo(inputs, stresses, stdev, enc_out)
        self.add_loss(tf.reduce_mean(rec + kl))
        self.add_metric(tf.reduce_mean(rec), name='rec')
        self.add_metric(tf.reduce_mean(kl), name='kl')

        return stresses, params, stdev





    def train(self, xs, ys, epochs=10, batch_size=32, validation_split=0):
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
        self.fit(xs, ys, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=callback)

def ortho_cann_3ff_model(lam_ut_all):
    # tf.keras.Input(shape=(1,), name='I1')
    # Inputs defined

    invs = get_invs(lam_ut_all)
    invs = tf.constant(invs)

    I1_in = invs[:, 0]
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


    I1_out, params1 = SingleInvNet(I1_ref, 0)
    I2_out, params2 = SingleInvNet(I2_ref, 4)
    I4f_out, params3 = SingleInvNet_I4(I4f_ref, 8)
    I4n_out, params4 = SingleInvNet_I4(I4n_ref, 11)
    I4theta_out, I4neg_out, params5 = SingleInvNet_I4theta(I4theta_ref, I4negtheta_ref, 14)
    params = params1 + params2 + params3 + params4 + params5

    output_grads = get_output_grads(lam_ut_all)  # 1000 x 5 x 2

    theta = np.pi / 3
    grad_I4theta = output_grads[:, 2, np.newaxis, :] * (np.cos(theta)) ** 2 \
              + output_grads[:, 3, np.newaxis, :] * (np.sin(theta)) ** 2 \
              + output_grads[:, 4, np.newaxis, :] * np.sin(2 * theta)
    grad_I4negtheta = output_grads[:, 2, np.newaxis, :] * (np.cos(theta)) ** 2 \
              + output_grads[:, 3, np.newaxis, :] * (np.sin(theta)) ** 2 \
              - output_grads[:, 4, np.newaxis, :] * np.sin(2 * theta)
    ALL_I_out = [I1_out[:, :, :, np.newaxis] * output_grads[:, 0, np.newaxis, :],
                 I2_out[:, :, :, np.newaxis] * output_grads[:, 1, np.newaxis, :],
                 I4f_out[:, :, :, np.newaxis] * output_grads[:, 2, np.newaxis, :],
                 I4n_out[:, :, :, np.newaxis] * output_grads[:, 3, np.newaxis, :],
                 I4theta_out[:, :, :, np.newaxis] * grad_I4theta
                 + I4neg_out[:, :, :, np.newaxis] * grad_I4negtheta]
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out,axis=2)
    terms = ALL_I_out.get_shape().as_list()[2]
    model = keras.models.Model(inputs=params, outputs=ALL_I_out)
    return model


def activation_Exp(x):
    return 1.0 * tf.math.exp(x)
def activation_Exp_I4(x):
    return 1.0 * (tf.math.exp(x) - 1)
def identity_deriv(x):
    return x * 0.0 + 1.0

# Define network block
def SingleInvNet(I1_ref, idi):
    # Layer 1. order
    I_1_w11, params_1 = cann_dense(I1_ref, activation=None, weight_name='w' + str(1 + idi))
    I_1_w21, params_2 = cann_dense(I1_ref, activation=activation_Exp, weight_name='w' + str(2 + idi))
    I_1_w41, params_3 = cann_dense(I1_ref**2, activation=None, weight_name='w' + str(3 + idi))
    I_1_w51, params_4 = cann_dense(I1_ref**2, activation=activation_Exp, weight_name='w' + str(4 + idi))
    I_1_w41 = I_1_w41 * 2 * I1_ref
    I_1_w51 = I_1_w51 * 2 * I1_ref

    collect = [I_1_w11, I_1_w21, I_1_w41, I_1_w51]
    collect_out = tf.stack(collect, axis=-1)

    return collect_out, params_1 + params_2 + params_3 + params_4

def SingleInvNet_I4(I1_ref, idi):
    I_1_w21, params1 = cann_dense(I1_ref, activation=activation_Exp_I4, weight_name='w' + str(1 + idi))
    I_1_w41, params2 = cann_dense(I1_ref ** 2, activation=None, weight_name='w' + str(2 + idi))
    I_1_w51, params3 = cann_dense(I1_ref ** 2, activation=activation_Exp, weight_name='w' + str(3 + idi))
    I_1_w41 = I_1_w41 * 2 * I1_ref
    I_1_w51 = I_1_w51 * 2 * I1_ref
    collect = [I_1_w21, I_1_w41, I_1_w51]
    collect_out = tf.stack(collect, axis=-1)

    return collect_out, params1 + params2 + params3

def SingleInvNet_I4theta(I4theta, I4negtheta, idi):
    exp_weights = [tf.keras.Input(shape=(1,), name='w' + str(1 + idi) + "_1"),
                   tf.keras.Input(shape=(1,), name='w' + str(1 + idi) + "_2")]
    exp_term = cann_dense(I4theta, activation=activation_Exp_I4, weights=exp_weights)
    exp_term_neg = cann_dense(I4negtheta, activation=activation_Exp_I4, weights=exp_weights)
    quad_weights = [tf.keras.Input(shape=(1,), name='w' + str(2 + idi) + "_1"),
                    tf.keras.Input(shape=(1,), name='w' + str(2 + idi) + "_2")]
    quad_term = cann_dense(I4theta ** 2, activation=None, weights=quad_weights)
    quad_term_neg = cann_dense(I4negtheta ** 2, activation=None, weights=quad_weights)
    exp_quad_weights = [tf.keras.Input(shape=(1,), name='w' + str(3 + idi) + "_1"),
                    tf.keras.Input(shape=(1,), name='w' + str(3 + idi) + "_2")]
    exp_quad_term = cann_dense(I4theta ** 2, activation=activation_Exp, weights=exp_quad_weights)
    exp_quad_term_neg = cann_dense(I4negtheta ** 2, activation=activation_Exp, weights=exp_quad_weights)

    collect = [exp_term, quad_term  * 2 * I4theta, exp_quad_term  * 2 * I4theta]
    collect_neg = [exp_term_neg, quad_term_neg * 2 * I4negtheta, exp_quad_term_neg * 2 * I4negtheta]


    return tf.stack(collect, axis=-1), tf.stack(collect_neg, axis=-1), exp_weights + quad_weights + exp_quad_weights

def get_invs(stretches): # 2 x 500 x 2
    ## 0-90
    w_stretch_090 = stretches[0, :, 0]
    s_stretch_090 = stretches[0, :, 1]
    invs_090 = np.stack([w_stretch_090 ** 2 + s_stretch_090 ** 2 + w_stretch_090 ** -2 * s_stretch_090 ** -2,
                         w_stretch_090 ** -2 + s_stretch_090 ** -2 + w_stretch_090 ** 2 * s_stretch_090 ** 2,
                         w_stretch_090 ** 2, s_stretch_090 ** 2, 0 * w_stretch_090],  axis=-1)

    ## 45-135
    x_stretch_45 = stretches[1, :, 0]
    y_stretch_45 = stretches[1, :, 1]
    invs_45 = np.stack([x_stretch_45 ** 2 + y_stretch_45 ** 2 + x_stretch_45 ** -2 * y_stretch_45 ** -2,
                         x_stretch_45 ** -2 + y_stretch_45 ** -2 + x_stretch_45 ** 2 * y_stretch_45 ** 2,
                         (x_stretch_45 ** 2 + y_stretch_45 ** 2) / 2, (x_stretch_45 ** 2 + y_stretch_45 ** 2) / 2,
                         (x_stretch_45 ** 2 - y_stretch_45 ** 2) / 2], axis=-1)
    return np.concatenate([invs_090, invs_45], axis=0).reshape((-1, 5))


def cann_dense(I1ref, activation=None, weights=None, weight_name=None):
    if activation is None:
        activation = identity_deriv
    if weights is None:
        weights = [tf.keras.Input(shape=(1,), name=weight_name + "_1"),
                   tf.keras.Input(shape=(1,), name=weight_name + "_2")]
        return weights[1] * weights[0] * activation(I1ref * weights[0]), weights
    else:
        return weights[1] * weights[0] * activation(I1ref * weights[0])


def get_I4_theta(I4f, I4n, I8):
    theta = np.pi / 3
    I4theta = I4f * (np.cos(theta)) ** 2 + I4n * (np.sin(theta)) ** 2 + I8 * np.sin(2 * theta)
    I4negtheta = I4f * (np.cos(theta)) ** 2 + I4n * (np.sin(theta)) ** 2 - I8 * np.sin(2 * theta)
    return I4theta, I4negtheta

def nelbo(stress_in, stress_out, stdev, enc_out):
    return rec_loss(stress_in, stress_out, stdev), kl_loss(enc_out) / 2000

def rec_loss(stress_in, stress_out, stdev):
    stdev_reshaped = stdev[:, np.newaxis, np.newaxis] if len(stdev.shape) > 0 else stdev
    squared_diff = tf.square((stress_out - stress_in) / stdev_reshaped)
    return 0.5 * tf.reduce_mean(squared_diff, axis=(1, 2)) + tf.math.log(stdev_reshaped)

def kl_loss(enc_out):
    latent_dim = int(enc_out.shape[1] / 2)
    means =  enc_out[:, 0:latent_dim]
    log_sigma = enc_out[:, latent_dim:]
    kls = log_sigma + (1 + means ** 2) / (2 * tf.math.exp(2 * log_sigma)) - 0.5
    return tf.reduce_sum(kls, axis=-1)


def train_vae(stretches, stresses, epochs=5000, should_save=True):
    inputs = stresses[:, :, :] / global_scale_factor
    stdev = np.std(inputs)
    model = CANN_VAE(34, 64, stretches)
    optimizer = keras.optimizers.Adam(lr=0.0001)

    model.compile(optimizer=optimizer)
    test, params, stdev = model(inputs)
    print(test)
    print(stdev)
    print(params)
    model.summary()

    model.train(inputs, inputs, epochs=epochs, batch_size=128, validation_split=0)
    if should_save:
        model.save_weights('./Results/vae_weights')
        np.save( './Results/stddev.npy', stdev)
    return model

def load_vae(stretches, stresses):
    model = CANN_VAE(34, 64, stretches)
    inputs = stresses[:, :, :] / global_scale_factor
    test, params, stdev = model(inputs)
    model.load_weights('./Results/vae_weights')
    return model

def test_vae(model, stretches, stresses):
    inputs = stresses[:, :, :] / global_scale_factor

    output, params, stdev = model(inputs)
    # np_config.enable_numpy_behavior()

    stress_out = output[:, :, :].numpy().reshape((10, 10, 100, 2)) * global_scale_factor
    stress_in = stresses[:, :, :].reshape((10, 10, 100, 2))
    stretch_plot = stretches[0, :, :].reshape((5, 100, 2))
    stretch_plot_delta = stretch_plot[2, :, 0] * 1e-6

    fig, axes = plt.subplots(4, 5)
    for i in range(4):
        for j in range(5):
            # Row i, column j
            for k in range(2):
                axes[i][j].plot(stretch_plot[j, :, i % 2] + stretch_plot_delta,
                                stress_in[k, int(i / 2) * 5 + j, :, i % 2], color="black")
                axes[i][j].plot(stretch_plot[j, :, i % 2] + stretch_plot_delta,
                                stress_out[k, int(i / 2) * 5 + j, :, i % 2], color="red")

    plt.show()

    ## Meta Test
    # train_cann(stretches[0, :, :], stress_out[0, :, :, :].reshape((-1, 2)))
    validate_cann(stretches[0, :, :], stress_out[0, :, :, :].reshape((-1, 2)), params[0, :])



def nuts_sample(model, stretches, stress_map, stresses):
    stresses_in = stresses / global_scale_factor
    def log_p(latent):
        stresses_out, params_out, stdev_out = model.decode(latent)
        prior = -0.5 * tf.norm(latent) ** 2
        log_var = - tf.math.log(stdev_out) * np.sum(stress_map, axis=(0, 1))
        mse = - 0.5 * tf.reduce_sum(((stresses_in - stresses_out) / stdev_out) ** 2, axis=(1,2))
        result = prior + log_var + mse
        # result = mse
        return result

    num_burnin_steps = 50
    n_samples = 100
    initial_state = np.random.randn(1, model.latent_dim)
    sampler = tfp.mcmc.NoUTurnSampler(log_p, step_size=tf.cast(0.1, tf.float64))
    adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=sampler,
        num_adaptation_steps=int(0.8 * num_burnin_steps),
        target_accept_prob=tf.cast(0.75, tf.float64))

    # Speed up sampling by tracing with `tf.function`.
    @tf.function(autograph=False, jit_compile=False)
    def do_sampling(init_state):
        return tfp.mcmc.sample_chain(
            kernel=adaptive_sampler,
            current_state=init_state,
            num_results=n_samples,
            num_burnin_steps=num_burnin_steps,
            trace_fn=None)


    samples = do_sampling(initial_state).numpy() # 100 x 1 x 64

    stress_out_avg = np.zeros_like(stresses)
    for i in range(n_samples):
        latent = samples[i, :, :] # 1 x 64
        stress_out, params_out, stdev_out = model.decode(latent)
        stress_out_avg += stress_out
    stress_out_avg /= n_samples

    stretch_plot = stretches[0, :, :].reshape((5, 100, 2))
    stretch_plot_delta = stretch_plot[2, :, 0] * 1e-6

    stress_out_plot = stress_out_avg.numpy() .reshape((10, 100, 2)) * global_scale_factor
    stress_in_plot = stresses.reshape((10, 100, 2))

    fig, axes = plt.subplots(4, 5)
    for i in range(4):
        for j in range(5):
            axes[i][j].plot(stretch_plot[j, :, i % 2] + stretch_plot_delta,
                            stress_in_plot[int(i / 2) * 5 + j, :, i % 2], color="black")
            axes[i][j].plot(stretch_plot[j, :, i % 2] + stretch_plot_delta,
                            stress_out_plot[int(i / 2) * 5 + j, :, i % 2], color="red")

    plt.show()

# 10 x 100 x 2
def train_cann(stretches, stresses):
    stretches = np.float64(stretches)
    stresses = np.float64(stresses)
    lam_ut_all = [[stretches.reshape((-1, 2))[:, k].flatten() for k in range(2)] for i in range(2)]
    P_ut_all = [[stresses.reshape((2, -1, 2))[i, :, k].flatten() for k in range(2)] for i in range(2)]
    modelFit_mode = "0123456789"
    alpha = 0
    p = 1
    epochs = 10000
    batch_size = 64
    gamma_ss = []
    P_ss = []
    Psi_model, terms = ortho_cann_3ff(lam_ut_all, gamma_ss, P_ut_all, P_ss, modelFit_mode, alpha, False, p)
    model_UT, model_SS, Psi_model, model = modelArchitecture("mesh", Psi_model)
    # Load training data

    model_given, input_train, output_train, sample_weights = traindata(modelFit_mode, model_UT, lam_ut_all, P_ut_all,
                                                                               model_SS, gamma_ss, P_ss, model, 0)
    # # model_given.summary(print_fn=print)
    path2saveResults = '../Results'
    Save_path = path2saveResults + '/model.h5'
    Save_weights = path2saveResults + '/weights'
    path_checkpoint = path2saveResults + '/best_weights'

    # Train model
    model_given, history, weight_hist_arr = Compile_and_fit(model_given, input_train, output_train, epochs,
                                                            path_checkpoint,
                                                                    sample_weights, batch_size)

    model_given.load_weights(path_checkpoint, by_name=False, skip_mismatch=False)
    tf.keras.models.save_model(Psi_model, Save_path, overwrite=True)
    Psi_model.save_weights(Save_weights, overwrite=True)
            #
            # # Add final weights to model history
            # threshold = 1e-3
    model_weights_0 = Psi_model.get_weights()
            # model_weights_0 = [model_weights_0[i] if i%2 == 0 or model_weights_0[i] > threshold ** p else 0.0 * model_weights_0[i] for i in range(len(model_weights_0))]
            # weight_hist_arr.append(model_weights_0)
            # Psi_model.set_weights(model_weights_0)

    Stress_predict_UT = model_UT.predict(lam_ut_all)

    stretch_plot = stretches.reshape((5, -1, 2))
    stress_in_plot = stresses.reshape((10, -1, 2))

    stress_out_plot = np.array(Stress_predict_UT).squeeze().transpose((0, 2, 1)).reshape((10, -1, 2)) # 2 x 500 x 2
    fig, axes = plt.subplots(4, 5)
    stretch_plot_delta = stretch_plot[2, :, 0] * 1e-6
    for i in range(4):
        for j in range(5):
            axes[i][j].plot(stretch_plot[j, :, i % 2] + stretch_plot_delta,
                            stress_in_plot[int(i / 2) * 5 + j, :, i % 2], color="black")
            axes[i][j].plot(stretch_plot[j, :, i % 2] + stretch_plot_delta,
                            stress_out_plot[int(i / 2) * 5 + j, :, i % 2], color="red")

    plt.show()


# 10 x 100 x 2
def train_cann(stretches, stresses):
    stretches = np.float64(stretches)
    stresses = np.float64(stresses)
    lam_ut_all = [[stretches.reshape((-1, 2))[:, k].flatten() for k in range(2)] for i in range(2)]
    P_ut_all = [[stresses.reshape((2, -1, 2))[i, :, k].flatten() for k in range(2)] for i in range(2)]
    modelFit_mode = "0123456789"
    alpha = 0
    p = 1
    epochs = 10000
    batch_size = 64
    gamma_ss = []
    P_ss = []
    Psi_model, terms = ortho_cann_3ff(lam_ut_all, gamma_ss, P_ut_all, P_ss, modelFit_mode, alpha, False, p)
    model_UT, model_SS, Psi_model, model = modelArchitecture("mesh", Psi_model)
    # Load training data

    model_given, input_train, output_train, sample_weights = traindata(modelFit_mode, model_UT, lam_ut_all, P_ut_all,
                                                                               model_SS, gamma_ss, P_ss, model, 0)
    # # model_given.summary(print_fn=print)
    path2saveResults = '../Results'
    Save_path = path2saveResults + '/model.h5'
    Save_weights = path2saveResults + '/weights'
    path_checkpoint = path2saveResults + '/best_weights'

    # Train model
    model_given, history, weight_hist_arr = Compile_and_fit(model_given, input_train, output_train, epochs,
                                                            path_checkpoint,
                                                                    sample_weights, batch_size)

    model_given.load_weights(path_checkpoint, by_name=False, skip_mismatch=False)
    tf.keras.models.save_model(Psi_model, Save_path, overwrite=True)
    Psi_model.save_weights(Save_weights, overwrite=True)
            #
            # # Add final weights to model history
            # threshold = 1e-3
    model_weights_0 = Psi_model.get_weights()
            # model_weights_0 = [model_weights_0[i] if i%2 == 0 or model_weights_0[i] > threshold ** p else 0.0 * model_weights_0[i] for i in range(len(model_weights_0))]
            # weight_hist_arr.append(model_weights_0)
            # Psi_model.set_weights(model_weights_0)

    Stress_predict_UT = model_UT.predict(lam_ut_all)

    stretch_plot = stretches.reshape((5, -1, 2))
    stress_in_plot = stresses.reshape((10, -1, 2))

    stress_out_plot = np.array(Stress_predict_UT).squeeze().transpose((0, 2, 1)).reshape((10, -1, 2)) # 2 x 500 x 2
    fig, axes = plt.subplots(4, 5)
    stretch_plot_delta = stretch_plot[2, :, 0] * 1e-6
    for i in range(4):
        for j in range(5):
            axes[i][j].plot(stretch_plot[j, :, i % 2] + stretch_plot_delta,
                            stress_in_plot[int(i / 2) * 5 + j, :, i % 2], color="black")
            axes[i][j].plot(stretch_plot[j, :, i % 2] + stretch_plot_delta,
                            stress_out_plot[int(i / 2) * 5 + j, :, i % 2], color="red")

    plt.show()

# params is len 34
def validate_cann(stretches, stresses, params):
    stretches = np.float64(stretches)
    stresses = np.float64(stresses)
    params = np.float64(params)
    lam_ut_all = [[stretches.reshape((-1, 2))[:, k].flatten() for k in range(2)] for i in range(2)]
    P_ut_all = [[stresses.reshape((2, -1, 2))[i, :, k].flatten() for k in range(2)] for i in range(2)]
    modelFit_mode = "0123456789"
    alpha = 0
    p = 1
    gamma_ss = []
    P_ss = []
    Psi_model, terms = ortho_cann_3ff(lam_ut_all, gamma_ss, P_ut_all, P_ss, modelFit_mode, alpha, False, p)
    model_UT, model_SS, Psi_model, model = modelArchitecture("mesh", Psi_model)
    # Load training data

    model_weights_0 = np.array_split(params, params.shape[0])
    Psi_model.set_weights(model_weights_0)

    Stress_predict_UT = model_UT.predict(lam_ut_all)

    stretch_plot = stretches.reshape((5, -1, 2))
    stress_in_plot = stresses.reshape((10, -1, 2))

    stress_out_plot = np.array(Stress_predict_UT).squeeze().transpose((0, 2, 1)).reshape((10, -1, 2)) * global_scale_factor  # 2 x 500 x 2
    fig, axes = plt.subplots(4, 5)
    stretch_plot_delta = stretch_plot[2, :, 0] * 1e-6
    for i in range(4):
        for j in range(5):
            axes[i][j].plot(stretch_plot[j, :, i % 2] + stretch_plot_delta,
                            stress_in_plot[int(i / 2) * 5 + j, :, i % 2], color="black")
            axes[i][j].plot(stretch_plot[j, :, i % 2] + stretch_plot_delta,
                            stress_out_plot[int(i / 2) * 5 + j, :, i % 2], color="red")

    plt.show()


# params is len 34
def test_cann(stretches, params):
    # Generate stretch / stress
    cann_model = ortho_cann_3ff_model(stretches)
    stresses = np.float64(tf.reduce_sum(cann_model(tf.split(params, params.shape[0], axis=0)), axis=2)) * global_scale_factor
    validate_cann(stretches[0, :, :], stresses, params)


