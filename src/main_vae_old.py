import numpy as np

from src.CANN.util_functions import disp_mat
from src.bcann_vae import disp_equation_weights_bcann, train_bcanns, train_bcann_vae, train_ensemble
from src.obsolete.models import train_cann, train_vae, load_vae, nuts_sample, test_vae
from src.plotting import plot_bcann, plot_bcann_raw_data, plot_vae
from src.utils import *
from src.obsolete.synth_data import generate_data


if __name__ == "__main__":
    params_train, stretches_train, stresses_train = generate_data(500)
    params, stretches_test, stresses_test = generate_data(10)
    # for i in range(17):
    #     print(i)
    # params = np.random.rand(34)
    # test_cann(stretches_train, params)
    # print(stresses_test.shape)
    # train_cann(stretches_test[0, :, :], stresses_test[0, :, :])
    tf.keras.backend.set_floatx('float64')
    model = train_vae(stretches_train, stresses_train)
    # model = load_vae(stretches_train, stresses_train)
    # #
    print(stretches_test.shape)
    # test_vae(model, stretches_test, stresses_test)
    from tensorflow.python.ops.numpy_ops import np_config

    np_config.enable_numpy_behavior()
    test_vae(model, stretches_test, stresses_test)
    stress_map = np.zeros_like(stresses_test[0, :, :])
    stress_map[0:200, :] = 1 # train using 0/90 only 01234
    print(stress_map.shape) # 1000 x 2?
    nuts_sample(model, stretches_test, stress_map, stresses_test[0, :, :])