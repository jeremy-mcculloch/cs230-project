from src.bcann_vae import disp_equation_weights_bcann, train_bcanns
from src.plotting import plot_bcann
from src.utils import *


if __name__ == "__main__":

    stretches, stresses = load_data()
    model_given, lam_ut_all, P_ut_all = train_bcanns(stretches[0, :, :], stresses, should_train=True) # Test if works the same as before

    modelFit_mode = "0123456789"
    # Print model equation and weights (with variances)
    disp_equation_weights_bcann(model_given.get_weights(), lam_ut_all, P_ut_all, modelFit_mode)


    plot_bcann(stretches, stresses, model_given, lam_ut_all, False)
    plot_bcann(stretches, stresses, model_given, lam_ut_all, True)
