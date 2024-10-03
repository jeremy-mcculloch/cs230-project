from synth_data import *
from models import *
import tensorflow as tf
if __name__ == "__main__":

    params_train, stretches_train, stresses_train = generate_data(500)
    params, stretches_test, stresses_test = generate_data(10)

    # for i in range(17):
    #     print(i)
    # params = np.random.rand(34)
    # test_cann(stretches_train, params)
    train_cann(stretches_test[0, :, :], stresses_test[0, :, :])

    tf.keras.backend.set_floatx('float64')


    # model = train_vae(stretches_train, stresses_train)
    model = load_vae(stretches_train, stresses_train)
    # #
    # test_vae(model, stretches_test, stresses_test)

    stress_map = np.zeros_like(stresses_test[0, :, :])
    stress_map[0:500, :] = 1

    nuts_sample(model, stretches_test, stress_map, stresses_test[0, :, :])
