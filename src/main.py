import numpy as np

from src.CANN.util_functions import disp_mat, get_model_id
from src.bcann_vae import disp_equation_weights_bcann, train_bcanns, train_bcann_vae, train_ensemble
from src.plotting import plot_bcann, plot_bcann_raw_data
from src.utils import *


if __name__ == "__main__":

    stretches, stresses = load_data()

    # Switch which of these is uncommented to determine which type of model is trained
    # id = "ensemble"
    model_type = "independent"
    # id = "correlated"
    # id = "unregularized"

    alpha = 0.1
    model_id = get_model_id(model_type, alpha)


    # Set to true to train model, false to load previously trained model
    should_train = False
    # String with list of which tests to use for training (0-9 are 0-90 orientation , a-e are +/-45 orientation)
    modelFit_mode = "012345678abce"
    # modelFit_mode = "0123456789abcde" # Uncomment this to train using all data

    # Ensemble trains one model per sample and then takes mean and variance of result
    if "ensemble" in model_id:
        stress_pred_mean, stress_pred_std, lam_ut_all, P_ut_all = train_ensemble(stretches[0, :, :], stresses, modelFit_mode=modelFit_mode, should_train=should_train) # Test if works the same as before
        plot_bcann_raw_data(stretches, stresses, stress_pred_mean, stress_pred_std, None, terms=False, id=model_id, modelFit_mode=modelFit_mode, blank=False, plot_dist=True)
        plot_bcann_raw_data(stretches, stresses, stress_pred_mean, stress_pred_std, None, terms=False, id=model_id, modelFit_mode=modelFit_mode, blank=False, plot_dist=False)

    else:
        # Train BCANN based on stretch and stress data provided
        model_given, lam_ut_all, P_ut_all = train_bcanns(stretches, stresses, modelFit_mode=modelFit_mode, should_train=should_train, model_type=model_type, alpha_in=alpha) # Test if works the same as before

        # Print covariance matrix
        model_weights = model_given.get_weights()
        nonzero_weights = [i for i in range(len(model_weights) // 2) if model_weights[2 * i + 1] > 0.0]
        cov_matrix = model_weights[-1] @ model_weights[-1].T
        cov_matrix_subset = cov_matrix[nonzero_weights, :][:, nonzero_weights]
        std_devs = np.sqrt(np.diag(cov_matrix_subset))
        corr_matrix = cov_matrix_subset / (std_devs[:, np.newaxis] @ std_devs[np.newaxis, :])
        disp_mat(corr_matrix)

        # Print model equation and weights (with variances)e
        disp_equation_weights_bcann(model_given.get_weights(), lam_ut_all, P_ut_all, "0123456789")

        # Create all plots
        plot_bcann(stretches, stresses, model_given, lam_ut_all, False, id, modelFit_mode, blank=True) # Blank plot of just data
        plot_bcann(stretches, stresses, model_given, lam_ut_all, True, id, modelFit_mode) # Plot of terms contributing to mean
        plot_bcann(stretches, stresses, model_given, lam_ut_all, False, id, modelFit_mode, plot_dist=True) # Plot of mean and standard deviation (plot data distribution vs predicted distribution)
        plot_bcann(stretches, stresses, model_given, lam_ut_all, False, id, modelFit_mode, plot_dist=False) # Plot of mean and standard deviation (plot data vs predicted distribution)

