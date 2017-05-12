from __future__ import division
import numpy as np
from math import exp
from calculate_statistics import eval_log_likelihood
from maximum_likelihood import maximum_likelihood_optimize


def metropolis_onestep(a_params, lamda_params, sigma, t_vec, f_samples, r):
    '''
    Take one step in the metropolis hastings method using Gaussian proposal distribution
    The hastings ratio also assumes the prior density is uniform
    Args:
        a_params, lamda_params, sigma: the (old) current values of parameters
        t_vec, f_samples: observed data f_samples at time t_vec, to calculate likelihood function value        
        r: a scalar for length scale of the multivariate Gaussian proposal 
    Return:
        accepted: boolean true if the proposal was accepted
        new_a_params, new_lamda_params, new_sigma: new values of the parameters after one metropolis step
    '''
    # set up the inputs and reshape arrays as needed
    assert a_params.size == lamda_params.size
    assert t_vec.size == f_samples.size
    m = a_params.size
    N = t_vec.size
    lamda_params = lamda_params.reshape(1, m)
    a_params = a_params.reshape(1, m)    
    assert r > 0

    # step 1: simulate new parameters based on old parameters
    # the Gaussian proposal distribution is used
    new_a_params = a_params + np.random.normal(0, r, (1, m))
    new_lamda_params = lamda_params + np.random.normal(0, r, (1, m))
    new_sigma = sigma + np.random.randn() * r

    # step 2: calculate the hastings ratio, we assume A, lamda, sigma are independent so their joint density is the product
    # the proposal distribution for a_params and lamda_params is Gaussian and symmetric, so cancelled out 
    # the prior is uniform and thus also cancelled out
    # h1 is the denominator in the hastings ratio, after cancelled common terms
    log_h1 = eval_log_likelihood(a_params, lamda_params, sigma, t_vec, f_samples)
    # h2 is the numerator in the hastings ratio, after cancelled common terms
    log_h2 = eval_log_likelihood(new_a_params, new_lamda_params, new_sigma, t_vec, f_samples)
    # to avoid divide by 0, note that h2/h1 = exp(log(h2/h1)) = exp(log_h2 - log_h1)
    prob_acceptance = np.minimum(1.0, exp(log_h2 - log_h1))

    # step 3: reject or accept with prob_acceptance
    accepted = False
    uSamp = np.random.rand()
    if uSamp < prob_acceptance:
        # the proposed values are accepted 
        accepted = True
        return accepted, new_a_params, new_lamda_params, new_sigma
    else:
        # accepted remains False and the next step has the same old values
        return accepted, a_params, lamda_params, sigma


def metropolis_posterior_sampling(t_vec, f_samples, prior_A_max, prior_lamda_max, prior_sigma_max, m, r, K):
    '''
    Given the data in t_vec, f_samples, and given the prior distribution on [0,max], sample the posterior of the parameters
    Args:
        t_vec, f_samples: given observed data 
        prior_A_max, prior_lamda_max, prior_sigma_max: the prior density is uniform on [0,max] for each of the parameters
        m: the size of one sample of a_params and lamda_params
        r: length scale of the multivariate Gaussian proposal
        K: must be an int; how many samples of the posterior are required
    Return: 
        acceptance_count: ndarray counting how many acceptances of size Kx1
        posterior_A_samples, posterior_lamda_samples: K samples of the posterior; ndarray size K x m 
        posterior_sigma_samples: K samples of the posterior; ndarray size K x 1
    '''
    # step 1: construct a first guess for the parameters
    # draw some values for the parameters from the prior which is uniform 
    # these values will be fed as first guess into the maximum_likelihood_optimize
    a_params = np.random.uniform(0.0, prior_A_max * 1.0, (1, m))
    lamda_params = np.random.uniform(0.0, prior_lamda_max * 1.0, (1, m))
    sigma = np.random.uniform(0.0, prior_sigma_max * 1.0, 1)

    # use maximum_likelihood_optimize as the starting initial feed for metropolis hastings
    a_params, lamda_params, sigma = maximum_likelihood_optimize(
        a_params, lamda_params, sigma, t_vec, f_samples)
    
    # preallocate ndarray for output
    acceptance_count = np.empty((K, 1), dtype=bool)
    posterior_A_samples = np.empty((K, m))
    posterior_lamda_samples = np.empty((K, m))
    posterior_sigma_samples = np.empty((K, 1))

    # get the first element of the posterior samples by doing one metropolis step
    acceptance_count[0,:], posterior_A_samples[0,:], posterior_lamda_samples[0,:], posterior_sigma_samples[0,:] = metropolis_onestep(
        a_params, lamda_params, sigma, t_vec, f_samples, r)
    # fill in the rest of the posterior samples
    for i in range(1, K):
        acceptance_count[i,:], posterior_A_samples[i,:], posterior_lamda_samples[i,:], posterior_sigma_samples[i,:] = metropolis_onestep(
            posterior_A_samples[i-1,:], posterior_lamda_samples[i-1,:], posterior_sigma_samples[i-1,:], t_vec, f_samples, r)
        if i%10000 == 0:
            print('made 10,000 metropolis samples')
    return acceptance_count, posterior_A_samples, posterior_lamda_samples, posterior_sigma_samples