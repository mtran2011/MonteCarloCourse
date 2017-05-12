import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_histogram(samples, var_of_estimator, true_values, tau_vec, percent_accepted, str_name):
    '''
    Plot histograms (normalized into density) of each of m parameters
    Args:
        samples: array of size n x m which are n samples each for m parameters 
        var_of_estimator: array of size 1 x m for variance of the estimator of m parameters 
        true_values: array of size 1 x m for true values of the m parameters 
        tau_vec: array of size 1 x m for autocorrelation time of the m parameters
        percent_accepted: a scalar for which portion of metropolis proposal was accepted 
        str_name: string name of the parameters (A or lamda or sigma)
    Return:
        save to a pdf file the histograms (normalized into density) of each of m parameters 
    '''
    n, m = samples.shape
    var_of_estimator = var_of_estimator.reshape(1, m)
    true_values = true_values.reshape(1, m)
    tau_vec = tau_vec.reshape(1, m)

    # first create a pdf file to save the plots
    filename = 'posterior histogram of ' + str_name + '.pdf'
    pp = PdfPages(filename)
    # for each parameter from 1 to m, plot a normalized histogram
    for i in range(m):
        plt.figure()
        normed_counts, bin_edges, _ = plt.hist(
            samples[:, i], bins='auto', normed=True, histtype='stepfilled')
        title_name = 'Normalized histogram (density) of the posterior samples of parameter ' + \
            str_name + '(' + str(i + 1) + ') \n Dashed line shows true param value'
        plt.suptitle(title_name, fontsize=10)
        x_label = 'Number of samples: %d. Variance of the estimator: %.2f \n' % (n, var_of_estimator[0, i])
        x_label = x_label + 'Autocorrelation time: %.2f. Percentage of metropolis proposal accepted: %.2f%%' % (tau_vec[0, i], percent_accepted * 100)
        plt.xlabel(x_label, fontsize=8)
        plt.ylabel('histogram counts normalized into pdf density', fontsize=8)
        # also plot a vlines at the true_values
        plt.vlines(true_values[0, i], np.amin(normed_counts), np.amax(normed_counts), linestyles='dashed')
        pp.savefig()
    pp.close()

def plot_acf(rho_mat, tau_vec, percent_accepted, str_name):
    '''
    Plot the autocorrelation function rho(t) at different lag t 
    Args: 
        rho_mat: array of size max_lag x m where the lags are from 1 to max_lag, for each m parameters
        tau_vec: array of size 1 x m for autocorrelation time of the m parameters
        percent_accepted: a scalar for the portion of metropolis proposal that was accepted
        str_name: string name of the parameters (A or lamda or sigma)
    Return: 
        save to a pdf file the autocorrelation function rho(t) at different lag t
    '''
    max_lag, m = rho_mat.shape[0], rho_mat.shape[1]
    # first create a pdf file to save the plots
    filename = 'ACF plot of ' + str_name + '.pdf'
    pp = PdfPages(filename)
    # for each parameter from 1 to m, plot the acf
    for col in range(m):
        plt.figure()
        plt.plot(np.arange(1, max_lag + 1, 1, dtype=int), rho_mat[:, col], '-o')
        title_name = 'Autocorrelation function of parameter ' + str_name + '(' + str(col + 1) + ')'
        plt.title(title_name)
        x_label = 'Lag of t in autocorrelation. '
        x_label = x_label + 'Autocorrelation time tau: %.2f.\n' % (tau_vec[0, col])
        x_label = x_label + 'Percentage of metropolis proposal accepted: %.2f%%' % (percent_accepted * 100)
        plt.xlabel(x_label, fontsize=8)
        plt.ylabel('Autocorrelation rho(t)')
        pp.savefig()
    pp.close()