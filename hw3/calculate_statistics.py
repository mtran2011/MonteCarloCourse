from __future__ import division
import numpy as np
from generate_data import calculate_Y

def eval_log_likelihood(a_params, lamda_params, sigma, t_vec, f_samples):
    '''
    From a given set of parameters and sampled data values, calculate the log(likelihood) function
    Args:
        a_params (ndarray): array of size m of parameters A(i)
        lamda_params (ndarray): array of size m of parameters lamda(i)
        sigma (float): a scalar for the length scale of the Gaussian noise
        t_vec (ndarray): array of size N (number of samples to generate)
        f_samples (ndarray): array of size N; the samples observed at t_vec
    Return:
        log_L (float): one scalar for the value of the log(likelihood) function
    '''
    # set up the inputs and reshape arrays as needed
    assert a_params.size == lamda_params.size
    assert t_vec.size == f_samples.size
    m = a_params.size
    N = t_vec.size
    lamda_params = lamda_params.reshape(1, m)
    a_params = a_params.reshape(1, m)
    f_samples = f_samples.reshape(1, N)
    sigma = abs(sigma)
    # get the Y(j) which are the mean of f(j) given these parameters
    y_samples = calculate_Y(a_params, lamda_params, t_vec)
    y_samples = y_samples.reshape(1, N)
    # diff is a row of f(j) - Y(j)
    diff = f_samples - y_samples
    log_L = diff.dot(diff.T) / (-2 * sigma**2) - N * np.log(sigma * np.sqrt(2.0 * np.pi))
    return np.asscalar(log_L)

def eval_likelihood(a_params, lamda_params, sigma, t_vec, f_samples):
    '''
    From a given set of parameters and sampled data values, calculate the likelihood function
    Args:
        a_params (ndarray): array of size m of parameters A(i)
        lamda_params (ndarray): array of size m of parameters lamda(i)
        sigma (float): a scalar for the length scale of the Gaussian noise
        t_vec (ndarray): array of size N (number of samples to generate)
        f_samples (ndarray): array of size N; the samples observed at t_vec
    Return:
        L (float): one scalar for the value of the likelihood function
    '''
    # set up the inputs and reshape arrays as needed
    assert a_params.size == lamda_params.size
    assert t_vec.size == f_samples.size
    m = a_params.size
    N = t_vec.size
    lamda_params = lamda_params.reshape(1, m)
    a_params = a_params.reshape(1, m)
    f_samples = f_samples.reshape(1, N)
    sigma = abs(sigma)
    # get the Y(j) which are the mean of f(j) given these parameters
    y_samples = calculate_Y(a_params, lamda_params, t_vec)
    y_samples = y_samples.reshape(1, N)
    # diff is a row of f(j) - Y(j)
    diff = f_samples - y_samples
    L = diff.dot(diff.T)
    L = L / (-2.0 * sigma**2)
    L = np.exp(L) * ((1.0 / (sigma * np.sqrt(2.0 * np.pi)))**N)
    return np.asscalar(L)