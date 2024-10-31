from src.bcann_vae import disp_equation_weights_bcann, train_bcanns
from src.plotting import plot_bcann
from src.utils import *


if __name__ == "__main__":

    stretches, stresses = load_data()
    independent = False
    model_given, lam_ut_all, P_ut_all = train_bcanns(stretches[0, :, :], stresses, should_train=False, independent=independent) # Test if works the same as before


    # Print covariance matrix
    model_weights = model_given.get_weights()
    nonzero_weights = [i for i in range(len(model_weights) // 2) if model_weights[2 * i + 1] > 0.0]
    cov_matrix = model_weights[-1] @ model_weights[-1].T
    cov_matrix_subset = cov_matrix[nonzero_weights, :][:, nonzero_weights]
    print(cov_matrix_subset)


    modelFit_mode = "0123456789"
    # Print model equation and weights (with variances)
    disp_equation_weights_bcann(model_given.get_weights(), lam_ut_all, P_ut_all, modelFit_mode)


    plot_bcann(stretches, stresses, model_given, lam_ut_all, False, independent)
    plot_bcann(stretches, stresses, model_given, lam_ut_all, True, independent)
