from src.bcann_vae import disp_equation_weights_bcann, train_bcanns, train_bcann_vae, train_ensemble
from src.plotting import plot_bcann, plot_bcann_raw_data
from src.utils import *


if __name__ == "__main__":

    stretches, stresses = load_data()
    # id = "ensemble"
    # id = "independent0p01"
    id = "correlated0p1"
    # id = "unregularized"
    should_train = False
    modelFit_mode = "012345678abce"
    if id == "ensemble":
        stress_pred_mean, stress_pred_std, lam_ut_all, P_ut_all = train_ensemble(stretches[0, :, :], stresses, modelFit_mode=modelFit_mode, should_train=should_train) # Test if works the same as before
        plot_bcann_raw_data(stretches, stresses, stress_pred_mean, stress_pred_std, None, terms=False, id=id, modelFit_mode=modelFit_mode, blank=False)

    else:
        model_given, lam_ut_all, P_ut_all = train_bcanns(stretches[0, :, :], stresses, modelFit_mode=modelFit_mode, should_train=should_train, id=id) # Test if works the same as before

        # Print covariance matrix
        model_weights = model_given.get_weights()
        nonzero_weights = [i for i in range(len(model_weights) // 2) if model_weights[2 * i + 1] > 0.0]
        cov_matrix = model_weights[-1] @ model_weights[-1].T
        cov_matrix_subset = cov_matrix[nonzero_weights, :][:, nonzero_weights]
        print(cov_matrix_subset)

        # Print model equation and weights (with variances)e
        disp_equation_weights_bcann(model_given.get_weights(), lam_ut_all, P_ut_all, "0123456789")

        plot_bcann(stretches, stresses, model_given, lam_ut_all, False, id, modelFit_mode, blank=True)
        plot_bcann(stretches, stresses, model_given, lam_ut_all, True, id, modelFit_mode)
        plot_bcann(stretches, stresses, model_given, lam_ut_all, False, id, modelFit_mode)

