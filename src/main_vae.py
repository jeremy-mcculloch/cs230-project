import numpy as np

from src.CANN.util_functions import disp_mat
from src.bcann_vae import disp_equation_weights_bcann, train_bcanns, train_bcann_vae, train_ensemble
from src.plotting import plot_bcann, plot_bcann_raw_data
from src.utils import *


if __name__ == "__main__":

    stretches, stresses = load_data_vae()
