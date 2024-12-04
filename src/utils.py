import numpy as np
import pandas as pd

# Load stress data from files
def load_data():
    # List of length 2 * Ns where Ns is the number of samples
    # List file and sheet names where data is stored, first 5 sheets are all 5 samples in the 0/90 orientation, next 5 sheets are the 5 samples in the +/-45 orientation
    file_names = ["../inputs/sample_stresses_90.xlsx"] * 5 +  ["../inputs/sample_stresses_45.xlsx"] * 5
    sheetnames = [f"Sheet{i+1}" for i in range(5)] * 2
    # Load from each sheet which should have dimensions 500 x 4
    # 5 experiments * 100 datapoints per experiment, 4 columns are x stretch, y stretch, x stress, y stress
    # Reshape to be 2 x (Ns*500) x 4
    loading_data_all = np.array([pd.read_excel(file_names[i], sheet_name=sheetnames[i], engine='openpyxl').to_numpy() for i in
                       range(len(file_names))]).reshape(2, -1, 4)
    # Seperate into stretch and stress (both 2 x (Ns*500) x 2)
    stretches = loading_data_all[:, :, 0:2]
    stresses = loading_data_all[:, :, 2:]
    return stretches, stresses

def load_data_vae():
    # List of length 2 * Ns where Ns is the number of samples
    # List file and sheet names where data is stored, first 5 sheets are all 5 samples in the 0/90 orientation, next 5 sheets are the 5 samples in the +/-45 orientation
    file_name = "../inputs/data_out.xlsx"
    xl = pd.ExcelFile(file_name)
    mesh_types = list(set(([x[:-2] for x in xl.sheet_names if not x[-3].isdigit()])))  # see all sheet names
    n_samples = 5
    print(xl.sheet_names)
    print(mesh_types)
    # assert False
    # sheetnames =
    # Load from each sheet which should have dimensions 500 x 4
    # 5 experiments * 100 datapoints per experiment, 4 columns are x stretch, y stretch, x stress, y stress
    # Reshape to be 2 x (Ns*500) x 4
    # loading_data_all = [[f"{mesh_type}_{i+1}" for i in range(n_samples) if xl.parse(f"{mesh_type}_{i+1}").to_numpy().shape[0] != 300] for mesh_type in mesh_types]

    loading_data_all = np.array([[xl.parse(f"{mesh_type}_{i+1}").to_numpy() for i in range(n_samples)] for mesh_type in mesh_types])
    print(loading_data_all.shape)
    # Seperate into stretch and stress (both 2 x (Ns*500) x 2)
    stretches = loading_data_all[:, :,:, 0:2]
    stresses = loading_data_all[:, :,:, 2:] * 1e3
    return stretches, stresses

def get_invs(stretches): # 2 x 500 x 2
    ## 0-90
    w_stretch_090 = stretches[:, 0]
    s_stretch_090 = stretches[:, 1]
    invs_090 = np.stack([w_stretch_090 ** 2 + s_stretch_090 ** 2 + w_stretch_090 ** -2 * s_stretch_090 ** -2,
                         w_stretch_090 ** -2 + s_stretch_090 ** -2 + w_stretch_090 ** 2 * s_stretch_090 ** 2,
                         w_stretch_090 ** 2, s_stretch_090 ** 2, 0 * w_stretch_090],  axis=-1)

    ## 45-135
    # x_stretch_45 = stretches[1, :, 0]
    # y_stretch_45 = stretches[1, :, 1]
    # invs_45 = np.stack([x_stretch_45 ** 2 + y_stretch_45 ** 2 + x_stretch_45 ** -2 * y_stretch_45 ** -2,
    #                      x_stretch_45 ** -2 + y_stretch_45 ** -2 + x_stretch_45 ** 2 * y_stretch_45 ** 2,
    #                      (x_stretch_45 ** 2 + y_stretch_45 ** 2) / 2, (x_stretch_45 ** 2 + y_stretch_45 ** 2) / 2,
    #                      (x_stretch_45 ** 2 - y_stretch_45 ** 2) / 2], axis=-1)
    # return np.concatenate([invs_090, invs_45], axis=0).reshape((-1, 5))
    return invs_090.reshape((-1, 5))

def get_invs_old(stretches): # 2 x 500 x 2
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
    # return invs_090.reshape((-1, 5))


def get_output_grads(lam_ut_all):  # 1000 x 5 x 2

    stretch_w = lam_ut_all[:, 0]
    stretch_s = lam_ut_all[:, 1]
    stretch_z_ws = 1 / (stretch_w * stretch_s)
    # stretch_x = lam_ut_all[1, :, 0]
    # stretch_y = lam_ut_all[1, :, 1]
    # stretch_z_xy = 1 / (stretch_x * stretch_y)

    ## Compute gradients of invariants wrt stretches
    output_grads_w = np.stack([2 * (stretch_w - stretch_z_ws * stretch_z_ws / stretch_w),
                               2 * (stretch_w * stretch_s * stretch_s - 1 / (stretch_w * stretch_w * stretch_w)),
                               2 * stretch_w, 0 * stretch_w, 0 * stretch_w], axis=-1)
    output_grads_s = np.stack([2 * (stretch_s - stretch_z_ws * stretch_z_ws / stretch_s),
                               2 * (stretch_w * stretch_w * stretch_s - 1 / (stretch_s * stretch_s * stretch_s)),
                               0 * stretch_w, 2 * stretch_s, 0 * stretch_w], axis=-1)

    # output_grads_x = np.stack([2 * (stretch_x - stretch_z_xy * stretch_z_xy / stretch_x),
    #                            2 * (stretch_x * stretch_y * stretch_y - 1 / (stretch_x * stretch_x * stretch_x)),
    #                            stretch_x, stretch_x, stretch_x], axis=-1)
    # output_grads_y = np.stack([2 * (stretch_y - stretch_z_xy * stretch_z_xy / stretch_y),
    #                            2 * (stretch_x * stretch_x * stretch_y - 1 / (stretch_y * stretch_y * stretch_y)),
    #                            stretch_y, stretch_y, -stretch_y], axis=-1)

    # output_grads_wx = np.concatenate([output_grads_w, output_grads_x], axis=0).reshape((-1, 5))
    # output_grads_sy = np.concatenate([output_grads_s, output_grads_y], axis=0).reshape((-1, 5))
    output_grads = np.stack([output_grads_w, output_grads_s], axis=-1) # N x 5 x 2
    return output_grads

def get_output_grads_old(lam_ut_all):  # 1000 x 5 x 2

    stretch_w = lam_ut_all[0, :, 0]
    stretch_s = lam_ut_all[0, :, 1]
    stretch_z_ws = 1 / (stretch_w * stretch_s)
    stretch_x = lam_ut_all[1, :, 0]
    stretch_y = lam_ut_all[1, :, 1]
    stretch_z_xy = 1 / (stretch_x * stretch_y)

    ## Compute gradients of invariants wrt stretches
    output_grads_w = np.stack([2 * (stretch_w - stretch_z_ws * stretch_z_ws / stretch_w),
                               2 * (stretch_w * stretch_s * stretch_s - 1 / (stretch_w * stretch_w * stretch_w)),
                               2 * stretch_w, 0 * stretch_w, 0 * stretch_w], axis=-1)
    output_grads_s = np.stack([2 * (stretch_s - stretch_z_ws * stretch_z_ws / stretch_s),
                               2 * (stretch_w * stretch_w * stretch_s - 1 / (stretch_s * stretch_s * stretch_s)),
                               0 * stretch_w, 2 * stretch_s, 0 * stretch_w], axis=-1)

    output_grads_x = np.stack([2 * (stretch_x - stretch_z_xy * stretch_z_xy / stretch_x),
                               2 * (stretch_x * stretch_y * stretch_y - 1 / (stretch_x * stretch_x * stretch_x)),
                               stretch_x, stretch_x, stretch_x], axis=-1)
    output_grads_y = np.stack([2 * (stretch_y - stretch_z_xy * stretch_z_xy / stretch_y),
                               2 * (stretch_x * stretch_x * stretch_y - 1 / (stretch_y * stretch_y * stretch_y)),
                               stretch_y, stretch_y, -stretch_y], axis=-1)

    output_grads_wx = np.concatenate([output_grads_w, output_grads_x], axis=0).reshape((-1, 5))
    output_grads_sy = np.concatenate([output_grads_s, output_grads_y], axis=0).reshape((-1, 5))
    output_grads = np.stack([output_grads_wx, output_grads_sy], axis=-1) # N x 5 x 2
    return output_grads

import tensorflow as tf


def tf_stack(array, axis=-1):
    return tf.keras.layers.concatenate([tf.expand_dims(x, axis=axis) for x in array], axis=axis)

def train_test_split(stresses):
    rng1 = np.random.default_rng(seed=25)
    rng1.shuffle(stresses)
    stress_dev = stresses[0:1, 0:3, :, :]
    stress_test = stresses[0:1, 3:, :, :]
    stress_train = stresses[1:, :, :, :]
    return stress_train, stress_dev, stress_test
