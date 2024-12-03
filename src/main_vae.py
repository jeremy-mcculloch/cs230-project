import numpy as np

from src.CANN.util_functions import disp_mat
from src.bcann_vae import disp_equation_weights_bcann, train_bcanns, train_bcann_vae, train_ensemble
from src.plotting import plot_bcann, plot_bcann_raw_data, plot_vae
from src.utils import *


if __name__ == "__main__":

    stretches, stresses = load_data_vae()
    stress_train, stress_dev, stress_test = train_test_split(stresses)
    vae_model, input = train_bcann_vae(stretches[0, 0, :, :], stress_train, should_train=True, independent=True)
    input = input.reshape((stress_train.shape[0], stress_train.shape[1], -1))
    # Training
    for i in range(stress_train.shape[0]):
        for j in range(stress_train.shape[1]):
            plot_vae(vae_model, stretches[0, 0, :, :], stress_train[i, j:(j+1), :, :], f"train_{i}_{j}", input[i, j:(j+1), :])
    # Dev set
    plot_vae(vae_model, stretches[0, 0, :, :], stress_dev[0, 0:1, :, :], "dev")
    # # Dev set
    # plot_vae(vae_model, stretches[0, 0, :, :], stress_dev[0, :, :, :])