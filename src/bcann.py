#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:40:23 2022

@author: kevinlinka
"""
import copy
import csv

import sympy.core.add
from tensorflow_probability.python.layers import DenseFlipout

# All the models that can be used for CANN training.
from src.CANN.util_functions import *
import numpy as np
from tensorflow import keras
from src.CANN.cont_mech import *
from src.CANN.models import *
import matplotlib.pyplot as plt
from plotting import *
import tensorflow_probability as tfp
from tensorflow.python import ops
import sympy as sp
import re
from bcann_vae import ortho_cann_3ff_bcann

# Orthotropic CANN with fibers in the Warp, theta, and negative theta directions
def ortho_cann_3ff_bcann_old(lam_ut_all, gamma_ss, P_ut_all, P_ss, modelFit_mode, alpha, should_normalize, p, kevins_version, two_term=False, terms=[]):
    # This stuff seems fishy, make sure its working right / robustly to different input formats
    Is_max = get_max_inv_mesh(reshape_input_output_mesh(lam_ut_all), modelFit_mode)
    P_ut_reshaped = np.array(reshape_input_output_mesh(P_ut_all))
    lam_ut_reshaped = np.array(reshape_input_output_mesh(lam_ut_all))
    scale_factors = np.mean(P_ut_reshaped, axis=-1) * (np.max(lam_ut_reshaped, axis=-1) - np.min(lam_ut_reshaped, axis=-1)) # * 10 because there are 5 loading configurations and 2
    scale_factor = 1.0 if kevins_version else np.sum(scale_factors)


    # Inputs defined
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
    I4f_in = tf.keras.Input(shape=(1,), name='I4f')
    I4n_in = tf.keras.Input(shape=(1,), name='I4n')
    I8fn_in = tf.keras.Input(shape=(1,), name='I8fn')
    I4theta, I4negtheta = I4_theta()([I4f_in, I4n_in, I8fn_in])

    # Put invariants in the reference configuration (substrct 3)
    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)
    I4f_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4f_in)
    I4n_ref = keras.layers.Lambda(lambda x: (abs(x)-1.0))(I4n_in)
    I4theta_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4theta)
    I4negtheta_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4negtheta)


    I1_out = SingleInvNet(I1_ref, 0, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 0]))
    I2_out = SingleInvNet(I2_ref, 6, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 1]))
    I4f_out = SingleInvNet_I4(I4f_ref, 12, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 2]))
    I4n_out = SingleInvNet_I4(I4n_ref, 18, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=(Is_max[:, 4]))

    I4theta_max = calculate_I4theta_max(Is_max)
    I4theta_out = SingleInvNet_I4theta(I4theta_ref, I4negtheta_ref, 24, should_normalize=should_normalize, alpha=alpha, p=p, I1s_max=I4theta_max)


    ALL_I_out = [I1_out, I2_out, I4f_out,I4n_out, I4theta_out]
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out,axis=1)  * scale_factor
    terms = ALL_I_out.get_shape().as_list()[1]
    if kevins_version:
        outputs = []
        kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (lam_ut_all[0][0].shape[0] * 1.0)
        for ii in range(terms):
            output = tfp.layers.DenseFlipout(2, bias_posterior_fn=None,
                                            bias_prior_fn=None,
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=None)(ALL_I_out[:, ii:ii+1])
            outputs.append(output)
        ALL_I_out_mean = tf.keras.layers.concatenate([x[:, 0:1] for x in outputs], axis=1)
        All_I_out_variances = tf.keras.layers.concatenate([x[:, 1:] for x in outputs], axis=1)

    else:
        All_I_out_variances = keras.layers.Dense(terms, kernel_constraint=DiagonalNonnegative(), use_bias=False, kernel_initializer=initializer_1)(ALL_I_out)
        ALL_I_out_mean = ALL_I_out

    Psi_model = keras.models.Model(inputs=[I1_in, I2_in, I4f_in, I4n_in, I8fn_in], outputs=[ALL_I_out_mean, All_I_out_variances], name='Psi')

    return Psi_model, terms  # 32 terms

class DiagonalNonnegative(keras.constraints.Constraint):
    """Constrains the weights to be diagonal and nonnegative
    """
    def __call__(self, w):
        N = K.int_shape(w)[-1]
        m = tf.eye(N)
        return m * tf.clip_by_value(w, 0, np.inf)

def myGradientSquared(a, b): # TODO find way to vectorize this
    a_unstacked = tf.unstack(a, axis=1)
    der_unstacked = [tf.gradients(x, b, unconnected_gradients="zero")[0] for x in a_unstacked]
    der = tf.stack(der_unstacked, axis=1)
    return tf.reduce_sum(der ** 2, axis=1)
    # return myGradient(a, b)

def myJacobian(a, b): # TODO find way to vectorize this
    a_unstacked = tf.unstack(a, axis=1)
    der_unstacked = [tf.gradients(x, b, unconnected_gradients="zero")[0] for x in a_unstacked]
    der = tf.stack(der_unstacked, axis=1)
    return der

# Complete model architecture definition given strain energy model
def get_stresses_090(Psi, I1, I2, I4w, I4s, I8ws, Stretch_w, Stretch_s):
    dWI1 = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I1])
    dWdI2 = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I2])
    dWdI4w = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I4w])
    dWdI4s = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I4s])
    Stress_w = keras.layers.Lambda(function=Stress_cal_w)([dWI1, dWdI2, dWdI4w, Stretch_w, Stretch_s])
    Stress_s = keras.layers.Lambda(function=Stress_cal_s)([dWI1, dWdI2, dWdI4s, Stretch_w, Stretch_s])
    return Stress_w, Stress_s
def get_stresses_090_var(Psi, I1, I2, I4w, I4s, I8ws, Stretch_w, Stretch_s):
    dWI1 = keras.layers.Lambda(lambda x: myGradientSquared(x[0], x[1]))([Psi[:, 0:4], I1])
    dWdI2 = keras.layers.Lambda(lambda x: myGradientSquared(x[0], x[1]))([Psi[:, 4:8], I2])
    dWdI4w = keras.layers.Lambda(lambda x: myGradientSquared(x[0], x[1]))([Psi[:, 8:], I4w])
    dWdI4s = keras.layers.Lambda(lambda x: myGradientSquared(x[0], x[1]))([Psi[:, 11:], I4s])
    Stress_w = keras.layers.Lambda(function=Stress_cal_w_sq)([dWI1, dWdI2, dWdI4w, Stretch_w, Stretch_s])
    Stress_s = keras.layers.Lambda(function=Stress_cal_s_sq)([dWI1, dWdI2, dWdI4s, Stretch_w, Stretch_s])

    return Stress_w, Stress_s

def get_stresses_45(Psi, I1, I2, I4w, I4s, I8ws, Stretch_x, Stretch_y):
    dWI1 = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I1])
    dWdI2 = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I2])
    dWdI4w = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I4w])
    dWdI4s = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I4s])
    dWdI8ws = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([Psi, I8ws])

    Stress_x = keras.layers.Lambda(function=Stress_cal_x_45)([dWI1, dWdI2, dWdI4w, dWdI4s, dWdI8ws, Stretch_x, Stretch_y])
    Stress_y = keras.layers.Lambda(function=Stress_cal_y_45)([dWI1, dWdI2, dWdI4w, dWdI4s, dWdI8ws, Stretch_x, Stretch_y])
    return Stress_x, Stress_y

def get_stresses_45_var(Psi, I1, I2, I4w, I4s, I8ws, Stretch_x, Stretch_y):
    dWI1 = keras.layers.Lambda(lambda x: myGradientSquared(x[0], x[1]))([Psi[:, 0:4], I1])
    dWdI2 = keras.layers.Lambda(lambda x: myGradientSquared(x[0], x[1]))([Psi[:, 4:8], I2])
    dWdI4w = keras.layers.Lambda(lambda x: myGradientSquared(x[0], x[1]))([Psi[:, 8:11], I4w])
    dWdI4s = keras.layers.Lambda(lambda x: myGradientSquared(x[0], x[1]))([Psi[:, 11:14], I4s])
    dWdI8ws = dWdI4s * 0

    Stress_x = keras.layers.Lambda(function=Stress_cal_x_45_sq)([dWI1, dWdI2, dWdI4w, dWdI4s, dWdI8ws, Stretch_x, Stretch_y])
    Stress_y = keras.layers.Lambda(function=Stress_cal_y_45_sq)([dWI1, dWdI2, dWdI4w, dWdI4s, dWdI8ws, Stretch_x, Stretch_y])
    Stress_x_theta, Stress_y_theta = keras.layers.Lambda(function=Stress_cal_theta)([Psi[:, 14:], I4w, I4s, I8ws, Stretch_x, Stretch_y])
    return Stress_x + Stress_x_theta, Stress_y + Stress_y_theta

def Stress_cal_theta(inputs):
    (Psi, I4w, I4s, I8ws, Stretch_x, Stretch_y) = inputs

    dWI4w = myJacobian(Psi, I4w)
    dWI4s = myJacobian(Psi, I4s)
    dWI8ws = myJacobian(Psi, I8ws)

    stress_x = tf.reduce_sum((dWI4w + dWI4s + dWI8ws) ** 2, axis=1) * Stretch_x ** 2
    stress_y = tf.reduce_sum((dWI4w + dWI4s - dWI8ws) ** 2, axis=1) * Stretch_y ** 2
    return stress_x, stress_y

def variance_transform(x):
    Para_SD = 0.2
    return tf.square(1e-3 +  tf.nn.elu(Para_SD * x))
def modelArchitecture_bcann(Psi_model, kevins_version):
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
    Psi, Psi_sd = Psi_model([I1, I2, I4w, I4s, I8ws])
    Stress_w, Stress_s = get_stresses_090(Psi, I1, I2, I4w, I4s, I8ws, Stretch_w, Stretch_s)
    if kevins_version:
        Stress_w_sd, Stress_s_sd = get_stresses_090(Psi, I1, I2, I4w, I4s, I8ws, Stretch_w, Stretch_s)
        Stress_w_var = variance_transform(Stress_w_sd)
        Stress_s_var = variance_transform(Stress_s_sd)

    else:
        Stress_w_var, Stress_s_var = get_stresses_090_var(Psi_sd, I1, I2, I4w, I4s, I8ws, Stretch_w, Stretch_s)
    outputs_90 = [Stress_w, Stress_w_var, Stress_s, Stress_s_var]
    model_90 = keras.models.Model(inputs=[Stretch_w, Stretch_s], outputs= outputs_90)
    # [tf.reduce_sum(x, axis=1) for x in outputs_90])

    # 45 degree offset
    Stretch_x = keras.layers.Input(shape=(1,),
                                   name='Stretch_x')
    Stretch_y = keras.layers.Input(shape=(1,),
                                   name='Stretch_y')
    I1 = keras.layers.Lambda(lambda x: x[0] ** 2 + x[1] ** 2 + 1. / (x[0] * x[1]) ** 2)([Stretch_x, Stretch_y])
    I2 = keras.layers.Lambda(lambda x: 1 / x[0] ** 2 + 1 / x[1] ** 2 + x[0] ** 2 * x[1] ** 2)(
        [Stretch_x, Stretch_y])
    I4w = keras.layers.Lambda(lambda x: (x[0] ** 2 + x[1] ** 2) / 2)([Stretch_x, Stretch_y])
    I4s = keras.layers.Lambda(lambda x: (x[0] ** 2 + x[1] ** 2) / 2)([Stretch_x, Stretch_y])
    I8ws = keras.layers.Lambda(lambda x: (x[0] ** 2 - x[1] ** 2) / 2)([Stretch_x, Stretch_y])
    Psi, Psi_sd = Psi_model([I1, I2, I4w, I4s, I8ws])
    Stress_x, Stress_y = get_stresses_45(Psi, I1, I2, I4w, I4s, I8ws, Stretch_x, Stretch_y)
    if kevins_version:
        Stress_x_sd, Stress_y_sd = get_stresses_45(Psi_sd, I1, I2, I4w, I4s, I8ws, Stretch_x, Stretch_y)
        Stress_x_var = variance_transform(Stress_x_sd)
        Stress_y_var = variance_transform(Stress_y_sd)
    else:
        Stress_x_var, Stress_y_var = get_stresses_45_var(Psi_sd, I1, I2, I4w, I4s, I8ws, Stretch_x, Stretch_y)
    outputs_45 = [Stress_x, Stress_x_var, Stress_y, Stress_y_var]
    model_45 = keras.models.Model(inputs=[Stretch_x, Stretch_y], outputs= outputs_45)
    # [tf.reduce_sum(x, axis=1) for x in outputs_45])

    models = [model_90, model_45]
    inputs = [model.inputs for model in models]
    outputs = [model.outputs for model in models]
    outputs = tf.keras.layers.concatenate(flatten(outputs),axis=1) # change shape so loss actually works

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    #
    return model, model, Psi_model, model

def NLL(y_true, y_pred):
    eps = 1e-6
    means = y_pred[:, 0::2]
    vars = y_pred[:, 1::2]
    errors = 0.5 * (tf.math.log(2 * np.pi * (vars + eps)) + tf.math.square(y_true - means) / (vars + eps))
    return tf.reduce_sum(errors, axis=1)


# Perform training of model, return fit model, training history, and weight history
def Compile_and_fit_bcann(model_given, input_train, output_train, epochs, path_checkpoint, sample_weights, batch_size, kevins_version):

    opti1 = tf.optimizers.Adam(learning_rate=0.00002 if kevins_version else 0.001)

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

    output_temp = tf.keras.backend.stack(flatten(output_train),axis=1) #
    output_temp = tf.cast(output_temp, tf.float32)


    # pred = model_given.predict(input_train)
    # loss = NLL(output_temp, pred)
    # print(pred)
    # print(loss)
    # assert False
    history = model_given.fit(input_train,
                              output_temp, # Repeat output_train twice to match format of input (
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_split=0.0,
                              callbacks=[es_callback, modelckpt_callback, weight_hist_callback],
                              shuffle=True,
                              verbose=1)

    return model_given, history, weight_hist_arr
def make_nonneg(x):
    return tf.clip_by_value(x,0.0,np.inf)
def train_bcanns(stretches, stresses, should_train=False, kevins_version=False):
    stretches = np.float64(stretches)
    stresses = np.float64(stresses) # 2 x (ns x 5 x 100) x 2
    lam_ut_all = [[stretches.reshape((-1, 2))[:, k].flatten() for k in range(2)] for i in range(2)]
    P_ut_all = [[stresses.reshape((2, -1, 2))[i, :, k].flatten() for k in range(2)] for i in range(2)]
    modelFit_mode = "0123456789"
    alphas = [0, 0.1]
    ps = [1.0, 0.5]
    epochs = 2000
    batch_size = 1000 # may want to increase? also may not matter since so many close by data points
    gamma_ss = []
    P_ss = []
    old = False
    last_weights = []
    for i in range(len(alphas)):
        # # model_given.summary(print_fn=print)
        name = "kevin" if kevins_version else "jeremy"
        path2saveResults = '../Results'
        Save_path = path2saveResults + f'/model_{name}.h5'
        Save_weights = path2saveResults + f'/weights_{name}'
        path_checkpoint = path2saveResults + f'/best_weights_{name}'

        if i < len(alphas) - 1 and not should_train:
            continue
        if old:
            Psi_model, terms = ortho_cann_3ff_bcann_old(lam_ut_all, gamma_ss, P_ut_all, P_ss, modelFit_mode, alphas[i],
                                                    not kevins_version, ps[i], kevins_version)

            # Create complete model architecture
            model_UT, model_SS, Psi_model, model = modelArchitecture_bcann(Psi_model, kevins_version)
            model.load_weights(Save_weights)
        else:
            model = ortho_cann_3ff_bcann(lam_ut_all, gamma_ss, P_ut_all, P_ss, modelFit_mode, alphas[i],
                                                    not kevins_version, ps[i], kevins_version)


        if i > 0 and should_train:
            model.set_weights(last_weights)

        # Load training data
        model_given, input_train, output_train, sample_weights = traindata(modelFit_mode, model, lam_ut_all, P_ut_all,
                                                                           model, gamma_ss, P_ss, model, 0)

        # model_given(input_train) # Test how this works first



        if should_train:
            # Train model
            model_given, history, weight_hist_arr = Compile_and_fit_bcann(model_given, input_train, output_train, epochs,
                                                                    path_checkpoint,
                                                                    sample_weights, batch_size, kevins_version)

            model_given.load_weights(path_checkpoint, by_name=False, skip_mismatch=False)
            tf.keras.models.save_model(model, Save_path, overwrite=True)
            model_given.save_weights(Save_weights, overwrite=True)
            last_weights = model_given.get_weights()

            if i < len(ps) - 1:
                p_ratio = ps[i + 1] / ps[i]
                last_weights = [last_weights[i] ** (p_ratio if (i % 3 == 1) else 1.0) for i in range(len(last_weights))]

        else:
            model.load_weights(Save_weights)

    # names = [weight.name for layer in model_given.layers for weight in layer.weights]
    # print(names)
    # unit_test_bcann(model_given, lam_ut_all)
    # Stress_predict_UT_old = model_UT.predict(lam_ut_all).reshape((-1, 2, 2, 2))  # N x 8 (45, xy, mean/var)
    # print(Stress_predict_UT_old)
    model_weights = model.get_weights()
    nonzero_terms = sum([x > 0 for x in model_weights[1::3]])
    print(f"Nonzero Terms: {nonzero_terms}")

    Stress_predict_UT = model_given.predict(lam_ut_all).reshape((-1, 2, 2, 2)) # N x 8 (45, xy, mean/var)
    print(Stress_predict_UT)
    # print(Stress_predict_UT - Stress_predict_UT_old)
    # Stress_predict_UT = Stress_predict_UT_old
    stretch_plot = stretches.reshape((5, 5, -1, 2))[0, :, :, :] # 5 x 100 x 2
    stress_in_plot = stresses.reshape((2, 5, 5, -1, 2))

    stress_out_plot = np.array(Stress_predict_UT[:, :, :, 0]).transpose((1, 0, 2)).reshape((2, 5, 5, -1, 2))[:, 0, :, :, :]  # 2 x 500 x 2
    stress_out_plot_std = np.sqrt(np.array(Stress_predict_UT[:, :, :, 1]).transpose((1, 0, 2)).reshape((2, 5, 5, -1, 2)))[:, 0, :, :, :]  # 2 x 500 x 2

    plot_bcann(stretch_plot, stress_in_plot, stress_out_plot, stress_out_plot_std, kevins_version)



def unit_test_bcann(model_given, lam_ut_all):
    model_weights = model_given.get_weights()
    # names = [weight.name for layer in model_given.layers for weight in layer.weights]
    # print(names)
    preds = model_given.predict(lam_ut_all)
    preds_test = preds * 0.0
    terms = len(model_weights) // 2
    for i in range(terms):
        model_weights_temp = copy.deepcopy(model_weights)
        for j in range(2 * terms):
            if i != (j // 2):
                model_weights_temp[j] = 0.0 * model_weights_temp[j]
        model_given.set_weights(model_weights_temp)
        preds_test += model_given.predict(lam_ut_all)

    nonzero_terms = sum([x > 0 for x in model_weights[1::2]])
    print(f"Nonzero Terms: {nonzero_terms}")
    error = preds - preds_test
    model_given.set_weights(model_weights)
    # print(error)
    # print(np.max(np.abs(error)))

