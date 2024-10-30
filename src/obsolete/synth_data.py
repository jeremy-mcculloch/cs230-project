import numpy as np
from src.utils import *

from src.utils import get_output_grads


# def generate_data(num_samples):
#     log_means = np.array([9.598, 2.824, 2.387, 3.415, 2.752, 1.498, 1.553, 3.207, 1.866, 1.762])
#     stdevs = np.array([0.520, 0.733, 0.626, 0.968, 0.239, 0.734, 1.575, 0.562, 0.420, 0.978]) / 2
#     params = np.exp(log_means + stdevs * np.random.randn(num_samples, np.shape(log_means)[0]))
#     lambdas = 1 + np.linspace(0, 1, 100) * (np.array([[1, 1.05, 1.1, 1.1, 1.1], [1.1, 1.1, 1.1, 1.05, 1.0]]) - 1).reshape((2, 5, 1))
#     lambdas = lambdas.reshape((2, -1)).transpose()
#     strains090, dstraindlambda090 = get_strain_090(lambdas)
#     strains45, dstraindlambda45 = get_strain_45135(lambdas)
#     strains = np.concatenate((strains090, strains45), axis=0)
#     dstraindlambda = np.concatenate((dstraindlambda090, dstraindlambda45), axis=0)
#     stresses = get_stresses(params, strains, dstraindlambda)
#     return params, lambdas, strains, stresses
#
# def generate_data(num_samples):
#     n_thetas = 3
#     thetas = np.random.rand(num_samples, n_thetas) * np.pi / 2
#     params = np.exp(np.random.randn(num_samples, n_thetas, 5)) # ns x 3 x 5
#     stretches = 1 + np.linspace(0, 1, 100) * (np.array([[1, 1.05, 1.1, 1.1, 1.1], [1.1, 1.1, 1.1, 1.05, 1.0]]) - 1).reshape((2, 5, 1))
#     stretches = stretches.reshape((2, -1)).transpose() # 500 x 2
#     stretches = np.stack((stretches, stretches), axis=0)
#     invs = get_invs(stretches) # 1000 x 5
#     output_grads = get_output_grads(stretches)  # 1000 x 5 x 2
#
#     stresses = get_stresses(thetas, params, invs, output_grads)
#     return np.concatenate((np.reshape(params, (num_samples, -1)), thetas), axis=1), stretches, stresses
#

def generate_data(num_samples):
    n_thetas = 3
    thetas = np.ones((num_samples, 1)) * np.array([[0, np.pi / 3, np.pi/2]])
    params = np.exp(np.random.randn(num_samples, n_thetas, 5)) # ns x 3 x 5
    stretches = 1 + np.linspace(0, 1, 100) * (np.array([[1, 1.05, 1.1, 1.1, 1.1], [1.1, 1.1, 1.1, 1.05, 1.0]]) - 1).reshape((2, 5, 1))
    stretches = stretches.reshape((2, -1)).transpose() # 500 x 2
    stretches = np.stack((stretches, stretches), axis=0)
    invs = get_invs(stretches) # 1000 x 5
    output_grads = get_output_grads(stretches)  # 1000 x 5 x 2

    stresses = get_stresses(thetas, params, invs, output_grads).reshape((-1, 10, 100, 2))
    snr = 10.0
    # signal_variance = np.std(stresses, axis=2)
    noise = np.random.standard_normal(stresses.shape)
    stresses += noise * stresses / snr
    stresses = stresses.reshape((-1, 1000, 2))
    return params, stretches, stresses




def get_strain_090(lambdas):
    strains = np.zeros((np.shape(lambdas)[0], 2, 2))
    strains[:, 0, 0] = (lambdas[:, 0] ** 2 - 1) / 2
    strains[:, 1, 1] = (lambdas[:, 1] ** 2 - 1) / 2
    dstraindlambda = np.zeros((np.shape(lambdas)[0], 2, 2, 2))
    dstraindlambda[:, 0, 0, 0] = lambdas[:, 0]
    dstraindlambda[:, 1, 1, 1] = lambdas[:, 1]
    return strains, dstraindlambda

def get_strain_45135(lambdas):
    strains = np.zeros((np.shape(lambdas)[0], 2, 2))
    strains[:, 0, 0] = (lambdas[:, 0] ** 2 + lambdas[:, 1] ** 2 - 2) / 4
    strains[:, 1, 1] = strains[:, 0, 0]
    strains[:, 0, 1] = (lambdas[:, 0] ** 2 - lambdas[:, 1] ** 2) / 4
    strains[:, 1, 0] = strains[:, 0, 1]
    dstraindlambda = np.zeros((np.shape(lambdas)[0], 2, 2, 2))
    dstraindlambda[:, 0, 0, 0] = lambdas[:, 0] / 2
    dstraindlambda[:, 0, 0, 1] = lambdas[:, 1] / 2
    dstraindlambda[:, 0, 1, 0] = lambdas[:, 0] / 2
    dstraindlambda[:, 0, 1, 1] = -lambdas[:, 1] / 2
    dstraindlambda[:, 1, 0, 0] = lambdas[:, 0] / 2
    dstraindlambda[:, 1, 0, 1] = -lambdas[:, 1] / 2
    dstraindlambda[:, 1, 1, 0] = lambdas[:, 0] / 2
    dstraindlambda[:, 1, 1, 1] = lambdas[:, 1] / 2
    return strains, dstraindlambda

def get_stresses(thetas, params, invs, output_grads):
    I4thetas    = invs[:, 2, np.newaxis, np.newaxis] * (np.cos(thetas)) ** 2 \
                  + invs[:, 3, np.newaxis, np.newaxis] * (np.sin(thetas)) ** 2 \
                  + invs[:, 4, np.newaxis, np.newaxis] * np.sin(2 * thetas) - 1
    I4negthetas = invs[:, 2, np.newaxis, np.newaxis] * (np.cos(thetas)) ** 2 \
                  + invs[:, 3, np.newaxis, np.newaxis] * (np.sin(thetas)) ** 2 \
                  - invs[:, 4, np.newaxis, np.newaxis] * np.sin(2 * thetas) - 1


    # output_grads 1000 x 5 x 2
    # thetas ns x 3
    exp_scale = 1
    exp_scale2 = 1
    dpsi_dtheta = params[:, :, 0] * (np.exp(params[:, :, 1] * I4thetas * exp_scale) - 1) + params[:, :, 2] * I4thetas \
                  + params[:, :, 3] * I4thetas * np.exp(params[:, :, 4] * I4thetas ** 2 * exp_scale2)
    dpsi_dnegtheta = params[:, :, 0] * (np.exp(params[:, :, 1] * I4negthetas * exp_scale) - 1) + params[:, :, 2] * I4negthetas \
                  + params[:, :, 3] * I4negthetas * np.exp(params[:, :, 4] * I4negthetas ** 2 * exp_scale2) # 1000 x ns x 3

    grad_I4thetas = output_grads[:, 2, :, np.newaxis, np.newaxis] * (np.cos(thetas)) ** 2 \
                   + output_grads[:, 3, :, np.newaxis, np.newaxis] * (np.sin(thetas)) ** 2 \
                   + output_grads[:, 4, :, np.newaxis, np.newaxis] * np.sin(2 * thetas) # 1000 x 2 x ns x 3
    grad_I4negthetas = output_grads[:, 2, :, np.newaxis, np.newaxis] * (np.cos(thetas)) ** 2 \
                   + output_grads[:, 3, :, np.newaxis, np.newaxis] * (np.sin(thetas)) ** 2 \
                   - output_grads[:, 4, :, np.newaxis, np.newaxis] * np.sin(2 * thetas) # 1000 x 2 x ns x 3

    stresses = np.sum(dpsi_dtheta[:, np.newaxis, :, :] * grad_I4thetas + dpsi_dnegtheta[:, np.newaxis, :, :] * grad_I4negthetas, axis=3) # 1000 x 2 x ns
    return np.transpose(stresses, (2, 0, 1)) * 1e4 # ns x 1000 x 2

# def get_stresses(params, strains, dstraindlambda):
#     terms = np.stack([strains[:, 0, 0] ** 2, strains[:, 1, 1] ** 2, strains[:, 0, 1] ** 2 + strains[:, 1, 0] ** 2,
#              strains[:, 0, 0] * strains[:, 1, 1], strains[:, 0, 0] ** 3, strains[:, 1, 1] ** 3,
#              strains[:, 0, 1] ** 3 + strains[:, 1, 0] ** 3, strains[:, 0, 0] ** 3 * strains[:, 1, 1], strains[:, 0, 0] * strains[:, 1, 1] ** 3], axis=1)
#
#     w = params[:, np.newaxis, 0]/2 * np.exp(np.sum(params[:, np.newaxis, 1:] * terms[np.newaxis, :, :], axis=2))
#     params = params[:, :, np.newaxis]
#     dlogwdE = np.zeros((params.shape[0], terms.shape[0], 2, 2))
#     dlogwdE[:, :, 0, 0] = params[:, 1, :] * strains[:, 0, 0] * 2 + params[:, 4, :] * strains[:, 1, 1] \
#                           + 3 * params[:, 5, :] * strains[:, 0, 0] ** 2 + 3 * params[:, 8, :] * strains[:, 0, 0] ** 2 * strains[:, 1, 1] \
#                             + params[:, 9, :] * strains[:, 1, 1]
#     dlogwdE[:, :, 0, 1] = params[:, 3, :] * strains[:, 0, 1] * 2 + 3 * params[:, 7, :] * strains[:, 0, 1] ** 2
#     dlogwdE[:, :, 1, 0] = params[:, 3, :] * strains[:, 0, 1] * 2 + 3 * params[:, 7, :] * strains[:, 0, 1] ** 2
#     dlogwdE[:, :, 1, 1] = params[:, 2, :] * strains[:, 1, 1] * 2 + params[:, 4, :] * strains[:, 0, 0] \
#                           + 3 * params[:, 6, :] * strains[:, 1, 1] ** 2 + params[:, 8, :] * strains[:, 0, 0] ** 3 \
#                           + params[:, 9, :] * 3 * strains[:, 0, 0] * strains[:, 1, 1] ** 2
#     stress = np.sum(w[:, :, np.newaxis, np.newaxis, np.newaxis] * dlogwdE[:, :, :, :, np.newaxis] * dstraindlambda, axis=(2, 3))
#     return stress
