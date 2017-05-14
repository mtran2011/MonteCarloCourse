import numpy as np
import matplotlib.pyplot as plt

from generate_data import create_fake_data
from metropolis_hastings import metropolis_posterior_sampling
from auto_correl import autocorrel_time
from plotting_module import plot_multiple_cdf, plot_multiple_pdf, plot_acf

def run_mcmc(true_As, true_lamdas, true_sigma, t_vec, r=0.2, K=1e6, nruns=1):
    '''
    Run MCMC nruns times, each time is K samples of the posterior
    Args:
        true_As, true_lamdas (ndarray): array of size m for true param
        true_sigma (float): true sigma param
        t_vec (ndarray): size N for num observations in data
        r (float): scale of the Gaussian proposal
        K (int): number of posterior samples
        nruns (int): how many runs, each run is K samples of posterior
    '''
    # step 1 create fake data
    # f_samples is the observed data with noise, true_Y is true mean, no noise
    f_samples, true_Y = create_fake_data(true_As, true_lamdas, 
                                         true_sigma, t_vec)
    K = int(K)
    m = true_As.size
    assert m == true_lamdas.size
    
    # step 2 run metropolis hastings to sample the posterior
    # the starting point of metropolis is the max likelihood param values
    # to run optimize we need a first guess 
    # the first guess is drawn from a uniform [0, param_max]
    prior_a_max = 2.0 * np.amax(true_As)
    prior_lamda_max = 2.0 * np.amax(true_lamdas)
    prior_sigma_max = 2.0 * true_sigma
    
    pct_list = np.empty(nruns)
    a_list = np.empty((K, m, nruns), dtype=float)
    lamda_list = np.empty((K, m, nruns), dtype=float)
    sigma_list = np.empty((K, 1, nruns), dtype=float)
    for run in range(nruns):
        pct_accepted, posterior_a, posterior_lamda, posterior_sigma = metropolis_posterior_sampling(t_vec, f_samples, prior_a_max, 
                prior_lamda_max, prior_sigma_max, m, r, K)    
        
        pct_list[run] = pct_accepted
        a_list[:, :, run] = posterior_a
        lamda_list[:, :, run] = posterior_lamda
        sigma_list[:, :, run] = posterior_sigma
        print('----completed one run of metropolis sampling; percent_accepted is: %.2f%%----' % (100 * pct_accepted))
    
    # step 3 plot cdf for each A(i) and lamda(i) and sigma for i in [0,m]
    for col in range(m):
        filename = 'cdf_posterior_A' + str(col+1)
        plot_multiple_cdf(a_list[:, col, :], filename)
        filename = 'pdf_posterior_A' + str(col+1)
        plot_multiple_pdf(a_list[:, col, :], filename)
                
        filename = 'cdf_posterior_lamda' + str(col+1)
        plot_multiple_cdf(lamda_list[:, col, :], filename)
        filename = 'pdf_posterior_lamda' + str(col+1)
        plot_multiple_pdf(lamda_list[:, col, :], filename)
    
    plot_multiple_cdf(sigma_list[:, 0, :], 'cdf_posterior_sigma')
    plot_multiple_pdf(sigma_list[:, 0, :], 'pdf_posterior_sigma')
    
    # plot the autocorrel time tau and rho(t) of each param from the first run
    # tau_A is a 1 x m array of the auto correlation time for each m features 
    # var_a_hat is a 1 x m array of the estimated variance of the estimator of each m features 
    # rho_a_mat is a (K/2) x m array of the autocorrelation at lags from 1 to (K/2)
    if nruns == 1:
        tau_A, var_a_hat, rho_a_mat = autocorrel_time(a_list[:, :, 0])
        tau_lamda, var_lamda_hat, rho_lamda_mat = autocorrel_time(
            lamda_list[:, :, 0])
        tau_sigma, var_sigma_hat, rho_sigma_mat = autocorrel_time(
            sigma_list[:, :, 0])
        print('--- completed calculating autocorrelation ---')
        plot_acf(rho_a_mat, tau_A, pct_list[0], 'A')
        plot_acf(rho_lamda_mat, tau_lamda, pct_list[0], 'lamda')
        plot_acf(rho_sigma_mat, tau_sigma, pct_list[0], 'sigma')
        
    print('plot done')
    
def main():
    true_As = np.array([0.2, 1.0])
    true_lamdas = np.array([0.1, 0.8])
    true_sigma = 0.5
    N, time_step = 5, 0.1
    t_vec = np.linspace(time_step, time_step * N, num=N)
    run_mcmc(true_As, true_lamdas, true_sigma, t_vec, r=0.4, K=2e7, nruns=2)

if __name__ == '__main__':
    main()