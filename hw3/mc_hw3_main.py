import numpy as np
import matplotlib.pyplot as plt

from generate_data import calculate_Y, create_fake_data
from calculate_statistics import eval_likelihood, eval_log_likelihood
from maximum_likelihood import maximum_likelihood_optimize, gradient_loglikelihood
from metropolis_hastings import metropolis_onestep, metropolis_posterior_sampling
from auto_correl import autocovar_lag_t, autocorrel_time, test_autocovar_lag_t
from plotting_module import plot_histogram, plot_acf

def run_mcmc(true_As, true_lamdas, true_sigma, t_vec, r=0.5, K=1e6):
    '''
    Args:
        true_As, true_lamdas (ndarray): array of size m for true param
        true_sigma (float): true sigma param
        t_vec (ndarray): size N for num observations in data
        r (float): scale of the Gaussian proposal
        K (long): number of posterior samples
    '''
    # step 1 create fake data
    # f_samples is the observed data with noise, true_Y is true mean, no noise
    f_samples, true_Y = create_fake_data(true_As, true_lamdas, true_sigma, t_vec)
    
    K = long(K)
    m = true_As.size
    assert m == true_lamdas.size
    
    # step 2 run metropolis hastings to sample the posterior
    # the starting point of metropolis is the max likelihood param values
    # to run optimize we need a first guess 
    # the first guess is drawn from a uniform [0, param_max]
    prior_a_max = 5.0 * np.amax(true_As)
    prior_lamda_max = 5.0 * np.amax(true_lamdas)
    prior_sigma_max = 5.0 * true_sigma
    
    percent_accepted, posterior_a_samples, posterior_lamda_samples, posterior_sigma_samples = metropolis_posterior_sampling(
        t_vec, f_samples, prior_a_max, prior_lamda_max, prior_sigma_max, m, r, K)    
    print('--- completed metropolis; percent_accepted is: %.2f%% ---' % (100 * percent_accepted))

def main():
    true_As = np.array([1.0, 1.5, 2.0])
    true_lamdas = np.array([0.4, 0.6, 0.8])
    true_sigma = 0.8
    N, time_step = 10, 0.1
    t_vec = np.linspace(time_step, time_step * N, num=N)
    run_mcmc(true_As, true_lamdas, true_sigma, t_vec)    

if __name__ == '__main__':
    main()