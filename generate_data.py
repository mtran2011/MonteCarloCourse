import numpy as np
# this module is used to create fake data 
def create_fake_data(A_params_vec, lambda_params_vec, sigma, t_vec):
    '''
    Args: 
        A_params_vec: np.ndarray of size = m of parameters A(i)
        lambda_params_vec: np.ndarray of size = m of parameters lambda(i)
        sigma: one number for standard deviation
        t_vec: np.ndarray of size = num_samples (number of samples to generate)
    Return:
        f_samples_vec: np.ndarray of size = num_samples containing the fake data with noise
        Y_samples_vec: np.ndarray of size = num_samples containing the fake data with no noise
    '''
    # assert size of A_params_vec and lambda_params_vec is equal, each being a row 
    assert A_params_vec.shape[0] == 1 and lambda_params_vec.shape[0] == 1
    assert A_params_vec.shape[1] == lambda_params_vec.shape[1]
    assert t_vec.shape[0] == 1
    # multiply t(j) with -1 * lambda_params_vec using: -1 * t_vec as a column * lambda_params_vec as a row
    t_vec = (-1) * t_vec 
    exp_matrix = np.dot(t_vec.T, lambda_params_vec)
    assert exp_matrix.shape[0] == t_vec.shape[1]
    assert exp_matrix.shape[1] == lambda_params_vec.shape[1]
    # for each t(j) eval exp(-lambda x t) using numpy.exp(x)
    exp_matrix = np.exp(exp_matrix)
    # dot product of exp() with A_params_vec to give Y(j)
    Y_samples_vec = np.dot(exp_matrix, A_params_vec.T)
    # create white noise
    epsilon_vec = np.random.normal(0, sigma, (1,Y_samples_vec.size))
    f_samples_vec = Y_samples_vec + epsilon_vec
    return f_samples_vec, Y_samples_vec
