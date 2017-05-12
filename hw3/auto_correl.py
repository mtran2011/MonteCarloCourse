from __future__ import division
import pandas as pd
import pandas.util.testing as pdt
import numpy as np

def autocovar_lag_t(samples, t):
    '''
    Given n samples of m features, calculate covar C(t) for each of the features
    Args:
        samples(ndarray): size n x m including n samples each for m features aka m random variables
        t(int): the distance in autocovariance; C(t) = covar of X(k) and X(k+t)
    Return:
        Ct(ndarray): size 1 x m for the estimated covar of X(k) and X(k+t) for each of the m features
    '''
    n, m = samples.shape
    
    # calculate X(k) minus its average
    x_k = samples[:(n-t), :]
    x_k_average = np.mean(x_k, axis=0).reshape(1, m)
    x_k_diff = x_k - x_k_average
    assert x_k_diff.shape[0] == (n-t) and x_k_diff.shape[1] == m
    
    # now calculate X(k+t) minus its average
    x_kt = samples[t:, :]
    x_kt_average = np.mean(x_kt, axis=0).reshape(1, m)
    x_kt_diff = x_kt - x_kt_average
    assert x_kt_diff.shape[0] == (n-t) and x_kt_diff.shape[1] == m
    
    # now fill in C(t) = covar(X(k), X(k+t))
    Ct = np.empty((1, m))
    for i in range(m):
        # for each of the m features, fill in C(t)
        Ct[0,i] = np.asscalar(np.dot(np.transpose(x_k_diff[:, i]), x_kt_diff[:, i])) / (n - t - 1)
    return Ct

def autocorrel_time(samples):
    '''
    Calculate autocorrel_time tau = 1 + 2 * sum(rho(t))
    Args:
        samples(ndarray): size n x m including n samples each for m features aka m random variables
    Return:
        tau(ndarray): size 1 x m for the estimated autocorrel_time for each of the m features
        var_of_estimator(ndarray): size 1 x m for the estimated variance of Ahat, where Ahat is average of the n samples
        rho_matrix(ndarray): size (n/2) x m for the auto correlation of lag 1 to n/2 for each m features 
    '''
    n, m = samples.shape[0], samples.shape[1]
    max_t = int(n/2)
    
    # allocate for C(t) to run from C(0) to C(max_t)
    Ct_matrix = np.empty((1 + max_t, m))
    # now loop through values of t and fill in each row of Ct_matrix
    for t in range(1 + max_t):
        Ct_matrix[t, :] = autocovar_lag_t(samples, t)
    
    # using numpy broadcasting to create rho(t) for t from 1 to max_t
    rho_matrix = Ct_matrix[1:, :] / Ct_matrix[0, :]
    assert rho_matrix.shape[0] == max_t and rho_matrix.shape[1] == m
    
    # now calculate tau and var_of_estimator for m features
    tau = 1.0 + 2.0 * np.sum(rho_matrix, axis=0, dtype=float).reshape(1, m)
    var_of_estimator = Ct_matrix[0, :] * tau / n
    assert var_of_estimator.shape[0] == 1 and var_of_estimator.shape[1] == m
    return tau, var_of_estimator, rho_matrix

def test_autocovar_lag_t(samples, t, Ct):
    '''
    Given n samples of m features, for each of the features, compare C(t) with result from pandas
    Args:
        samples(ndarray): size n x m including n samples each for m features aka m random variables
        t(int): the distance in autocovariance; C(t) = covar of X(k) and X(k+t)
        Ct(ndarray): size 1 x m for the estimated covar of X(k) and X(k+t) for each of the m features
    Return:
        no AssertionError is raised if the values in C(t) agrees with pandas Series functions 
    '''
    n, m = samples.shape[0], samples.shape[1]
    df = pd.DataFrame(samples)
    for i in range(m):
        # for X being the variable in column i, C(t) should be covar of X(k) and X(k+t)
        # compare C(t) with the covar given by pd.Series and np.cov
        # first get the covar given by pd.Series 
        s_k = pd.Series(samples[:(n-t), i])
        s_kt = pd.Series(samples[t:, i])
        cov_series = s_k.cov(s_kt)

        # now get the cov given by np.cov 
        cov_mat = np.cov(samples[:(n-t), i].T, samples[t:, i].T)
        assert cov_mat.shape[0] == 2 and cov_mat.shape[1] == 2
        cov_np = cov_mat[1,0]
        # now check that they are all equal 
        assert abs(cov_series - cov_np) < 1e-8
        assert abs(cov_series - Ct[0,i]) < 1e-8