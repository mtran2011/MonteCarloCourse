from __future__ import division
import numpy as np
from generate_data import calculate_Y
from scipy.optimize import minimize


def maximum_likelihood_optimize(a_params, lamda_params, sigma, t_vec, f_samples):
    '''
    Find the parameters that will maximize the log(likelihood) function 
    Args:
        a_params, lamda_params, sigma: initial guess of the parameters
        t_vec, f_samples: the given observed data
    Return: 
        A, lamda, sigma: values after optimization to max the log(likelihood)
    '''
    # preliminary processing of the data
    assert a_params.size == lamda_params.size
    assert t_vec.size == f_samples.size
    sigma = np.asarray(sigma, dtype=float)
    m = a_params.size
    N = f_samples.size
    a_params = a_params.reshape(1, m)
    lamda_params = lamda_params.reshape(1, m)
    sigma = sigma.reshape(1, 1)
    t_vec = t_vec.reshape(1, N)
    f_samples = f_samples.reshape(1, N)

    # define the objective function that will be minimized
    # minimize the -1 * log(likelihood) in order to maximize likelihood
    def minus_log_likelihood(x):
        '''
        Args:
            x: array of 2m+1 elements; the first m are for a_params, next m are lamda_params, last is sigma
        Return:
            negative_log_likelihood: must be scalar; -1 * log(likelihood) given the observed data t_vec and f_samples
        '''
        assert x.size == (2*m + 1)
        A = x[:m]
        lamda = x[m:(2*m)]
        sig = abs(x[2*m])
        # get the Y(j) which are the mean of f(j) given these parameters
        Y = calculate_Y(A, lamda, t_vec)
        Y = Y.reshape(1, N)
        diff = f_samples - Y
        # sum of the squared of the diff between f(j) and Y(j)
        diff_sum_squared = diff.dot(diff.T)
        negative_log_likelihood = diff_sum_squared / (2.0 * sig**2) + N * np.log(sig * np.sqrt(2.0 * np.pi))
        return np.asscalar(negative_log_likelihood)
    
    # now apply scipy optimization using the above objective function
    # from the initial guess, minimize the objective function with no constraint 
    initial_params_guess = np.concatenate((a_params, lamda_params, sigma), axis=1)
    result_object = minimize(minus_log_likelihood, initial_params_guess)
    result_values = result_object.x
    return result_values[:m], result_values[m:(2*m)], result_values[2*m]


def gradient_loglikelihood(a_params, lamda_params, sigma, t_vec, f_samples):
    '''
    Given data f(j) and tj, and current values of the parameters, 
    find the gradient of log(likelihood) evaluated at these parameters
    Args:
        a_params, lamda_params, sigma: current values of the parameters (total 2m + 1 parameters)
        t_vec, f_samples: given observed data of N samples
    Return: 
        grad_A, grad_lambda: array size (1, m) for the gradient wrt a_params and lamda_params
        grad_sigma: scalar for the gradient wrt sigma 
    '''
    assert a_params.size == lamda_params.size
    assert t_vec.size == f_samples.size
    m = a_params.size
    N = f_samples.size 
    lamda_params = lamda_params.reshape(1, m)
    a_params = a_params.reshape(1, m)
    f_samples = f_samples.reshape(1, N)
    t_vec = t_vec.reshape(1, N)
    sigma = abs(sigma)
    # get the Y(j) which are the mean of f(j) given these parameters      
    Y = calculate_Y(a_params, lamda_params, t_vec)
    Y = Y.reshape(1, N)    
    diff = f_samples - Y
    # sum of the squared of the diff between f(j) and Y(j)
    diff_sum_squared = diff.dot(diff.T)
    # allocate the output: grad_A of size m, grad_lambda of size m
    grad_A = np.empty((1, m))
    grad_lambda = np.empty((1, m))
    # loop to find the partial derivative of log(L) w.r.t A(i) and lamda_params(i)
    for i in range(m):        
        # get exp(-1 * lamda_params[i] * tj)
        exponential = np.exp(-1.0 * lamda_params[0,i] * t_vec)
        exponential = exponential.reshape(1, N)
        grad_A[0,i] = np.asscalar(diff.dot(exponential.T) / (sigma**2))
        # get exp(-1 * lamda_params[i] * tj) * A(i) * (tj)
        temp = a_params[0,i] * exponential * t_vec
        grad_lambda[0,i] = np.asscalar(diff.dot(temp.T) * (-1.0) / (sigma**2))
    # partial derivative of log(L) w.r.t sigma
    grad_sigma = np.asscalar(diff_sum_squared / (sigma**3) - N / sigma)
    return grad_A, grad_lambda, grad_sigma