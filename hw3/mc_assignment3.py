from __future__ import division
import time
import numpy as np

from generate_data import calculate_Y, create_fake_data
from calculate_statistics import eval_likelihood, eval_log_likelihood
from maximum_likelihood import maximum_likelihood_optimize, gradient_loglikelihood
from metropolis_hastings import metropolis_onestep, metropolis_posterior_sampling
from auto_correl import autocovar_lag_t, autocorrel_time, test_autocovar_lag_t
from plotting_module import plot_histogram, plot_acf


def testing():    
    m, N = 2, 5
    a_params = np.linspace(0.5, m * 0.5, m)
    lamda_params = np.linspace(0.2, m * 0.2, m)
    t_vec = np.linspace(0.1, N * 0.1, N)
    sigma = -1.5
    
    # test calculate_Y and create_fake_data
    # create_fake_data returns f_samples (data with noise) and y_samples(no noise)
    f_samples, y_samples = create_fake_data(a_params, lamda_params, sigma, t_vec)    
    for j in range(N):
        y = np.zeros(1)
        for i in range(m):
            y += a_params[i] * np.exp(-lamda_params[i] * t_vec[j])
        np.testing.assert_allclose(np.asscalar(y_samples[0,j]), y)
    
    # test eval_likelihood and eval_log_likelihood
    diff = f_samples - y_samples
    diff = diff.reshape(1, N)
    L_true = diff.dot(diff.T) / (-2 * abs(sigma)**2)
    L_true = np.exp(L_true) / (abs(sigma) * np.sqrt(2 * np.pi))**N
    np.testing.assert_allclose(L_true, eval_likelihood(a_params, lamda_params, sigma, t_vec, f_samples), rtol=1e-4)
    np.testing.assert_allclose(L_true, np.exp(eval_log_likelihood(a_params, lamda_params, sigma, t_vec, f_samples)), rtol=1e-4)
    
    # test maximum_likelihood_optimizer by checking if squared norm of the gradient is close to zero 
    optimal_a, optimal_lamda, optimal_sigma = maximum_likelihood_optimize(
        np.random.rand(m), np.random.rand(m), np.random.rand(), t_vec, f_samples)
    grad_a, grad_lamda, grad_sigma = gradient_loglikelihood(optimal_a, optimal_lamda, optimal_sigma, t_vec, f_samples)
    np.testing.assert_allclose(grad_a.dot(grad_a.T), np.zeros((1,1)), rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(grad_lamda.dot(grad_lamda.T), np.zeros((1,1)), rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(grad_sigma**2, np.zeros((1,1)), rtol=1e-2, atol=1e-2)
    
    # test metropolis_onestep and metropolis_posterior_sampling
    r = 1
    prior_a_max = 0.5 * m
    prior_lamda_max = 0.2 * m
    prior_sigma_max = 5.0
    K = int(1e3)
    acceptance_count, posterior_a_samples, posterior_lamda_samples, posterior_sigma_samples = metropolis_posterior_sampling(
        t_vec, f_samples, prior_a_max, prior_lamda_max, prior_sigma_max, m, r, K)
    percent_accepted = np.sum(acceptance_count) / (acceptance_count.size * 1.0)
    print('portion of proposal accepted: %.2f' % percent_accepted)
    
    # test autocovar_lag_t and autocorrel_time
    for t in range(1 + int(K/2)):
        Ct = autocovar_lag_t(posterior_a_samples, t)
        # no AssertionError is raised in the test method below if C(t) agrees with np.cov
        test_autocovar_lag_t(posterior_a_samples, t, Ct)
    tau_a, var_a_hat, rho_a_mat = autocorrel_time(posterior_a_samples)
    tau_lamda, var_lamda_hat, rho_lamda_mat = autocorrel_time(posterior_lamda_samples)
    tau_sigma, var_sigma_hat, rho_sigma_mat = autocorrel_time(posterior_sigma_samples)
    print('tau_a is: \n')
    print(tau_a)
    print('var_a_hat is: \n')
    print(var_a_hat)
    print('all tests passed')
    

def main():
    # a) create some fake data
    # m is the size of a_params and lamda_params
    # N the size of t_vec and f_samples, aka the number of observed data
    m, N, time_step = 2, 10, 0.1
    true_A = 2 * np.random.rand(m)
    true_lamda = 2 * np.random.rand(m)
    true_sigma = -1.5
    t_vec = np.linspace(time_step, time_step * N, num=N)
    # f_samples is the observed data with noise, true_Y is true mean, no noise     
    f_samples, true_Y = create_fake_data(true_A, true_lamda, true_sigma, t_vec)
    print('--- completed creating fake data ---')

    # b) using metropolis_hastings to sample from the posterior distribution of the parameters
    # imagine we are given the data in t_vec and f_samples and do not know the true parameters
    # asssume that the prior density of parameters are uniform [0,max]
    prior_a_max = 2.0
    prior_lamda_max = 2.0
    prior_sigma_max = 2.0
    # set input for metropolis with r = scale of the Gaussian proposal
    # and K = number of posterior samples
    r = 0.55
    K = int(1e3)
    # generate samples of the posterior using metropolis
    # the first guess fed into metropolis will be a maximum likelihood estimate
    acceptance_count, posterior_a_samples, posterior_lamda_samples, posterior_sigma_samples = metropolis_posterior_sampling(
        t_vec, f_samples, prior_a_max, prior_lamda_max, prior_sigma_max, m, r, K)
    percent_accepted = np.sum(acceptance_count) / acceptance_count.size
    print('--- completed metropolis; percent_accepted is: %.2f%% ---' % (100 * percent_accepted))

    # c) from the posterior samples get the tau and variance of the estimator
    # tau_A is a 1 x m array of the auto correlation time for each m features 
    # var_a_hat is a 1 x m array of the estimated variance of the estimator of each m features 
    # rho_a_mat is a (K/2) x m array of the autocorrelation at lags from 1 to (K/2)
    tau_A, var_a_hat, rho_a_mat = autocorrel_time(posterior_a_samples)
    tau_lamda, var_lamda_hat, rho_lamda_mat = autocorrel_time(posterior_lamda_samples)
    tau_sigma, var_sigma_hat, rho_sigma_mat = autocorrel_time(posterior_sigma_samples)
    print('--- completed calculating autocorrelation ---')
    
    # d) plot the results of posterior sampling
    # plot the histogram density of posterior_a_samples with true_A and var_a_hat, tau_A, percent_accepted 
    # plot_histogram(posterior_a_samples, var_a_hat, true_A, tau_A, percent_accepted, 'A')
    # plot_histogram(posterior_sigma_samples, var_sigma_hat, true_sigma, tau_sigma, percent_accepted, 'sigma')
    # print('--- completed plotting histograms ---')

    # plot the autocorrelation at various lags using rho_a_mat
    plot_acf(rho_a_mat, tau_A, percent_accepted, 'A')
    plot_acf(rho_sigma_mat, tau_sigma, percent_accepted, 'sigma')
    print('--- completed plotting ACF ---')

if __name__ == '__main__':
    # testing()
    start_time = time.time()
    main()
    print('--- %s seconds ---' % (time.time() - start_time))