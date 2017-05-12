from __future__ import division
import numpy as np


def calculate_Y(a_params, lamda_params, t_vec):
    '''
    From the parameters and the given time, calculate the mean Y(tj)
    Args: 
        a_params (ndarray): array of size m of parameters A(i)
        lamda_params (ndarray): array of size m of parameters lamda(i)
        t_vec (ndarray): array of size N (number of samples to generate)
    Return:        
        ndarray: array of (1,N) containing the true mean of the data (no noise yet)
    '''
    # set up the inputs and reshape arrays as needed
    m = a_params.size
    N = t_vec.size
    assert lamda_params.size == m
    lamda_params = lamda_params.reshape(1, m)
    a_params = a_params.reshape(1, m)
    t_vec = t_vec.reshape(N, 1)
    # multiply t(j) with -1 * lamda_params using t_vec as a column and
    # lamda_params as a row
    exp_matrix = np.dot(t_vec, (-1.0) * lamda_params)
    assert exp_matrix.shape[0] == N and exp_matrix.shape[1] == m
    exp_matrix = np.exp(exp_matrix)
    # dot product of exp_matrix with a_params to give Y(j)
    y_samples = exp_matrix.dot(a_params.T)
    assert y_samples.shape[0] == N and y_samples.shape[1] == 1
    return y_samples.reshape(1, N)

def create_fake_data(a_params, lamda_params, sigma, t_vec):
    '''
    From the parameters and the given time, first calculate Y(tj) and then add some Gaussian noise
    Args: 
        a_params (ndarray): array of size m of parameters A(i)
        lamda_params (ndarray): array of size m of parameters lamda(i)
        sigma (float): a scalar for the length scale of the Gaussian noise
        t_vec (ndarray): array of size N (number of samples to generate)
    Return:
        f_samples (ndarray): array of size N containing the fake data with noise
        y_samples (ndarray): array of size N containing the true mean of the data (no noise)
    '''
    N = t_vec.size
    # first get Y(j) which is the mean of f(j) with no noise
    y_samples = calculate_Y(a_params, lamda_params, t_vec)
    y_samples = y_samples.reshape(1, N)
    # create Gaussian noise and add noise to y_samples
    f_samples = y_samples + np.random.randn(1, N) * sigma
    assert f_samples.shape[0] == 1 and f_samples.shape[1] == N
    return f_samples, y_samples
