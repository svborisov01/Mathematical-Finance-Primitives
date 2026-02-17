import stochastic_engine as se
import numpy as np
import matplotlib.pyplot as plt

def geometric_brownian_motion(initial_vector = 100, drift_vector = 0.1, diffusion_vector = 0.2, correlation_matrix = np.eye(1), T = 1, granularity = 252, n_paths = 100, visualize = False):
    """
    Generates a 1D geometric brownian motion

    Inputs:
    - initial: initial value of the brownian motion. Default is 100.
    - drift: drift coefficient (mu) in annual terms. Default is 0.1.
    - diffusion: diffusion coefficient (sigma) in annual terms. Default is 0.2.
    - T: time in years. Default is 1. To get the number of steps, multiply T by 252.
    - sim_num: number of simulations to run. Default is 100.
    - visualize: whether to plot the paths. Default is False.

    Ouputs:
    - paths: 2D array of paths of the simulation with dimensions (round(T * 252) + 1, sim_num).
    - (optional) plot of the paths.

    Notes:
    - the simulation is (round(T * 252) + 1) in length to account for the initial value.
    - to select a single path, use geometric_brownian_motion(...)[:, i]
    - to select a single point in time across paths, use geometric_brownian_motion(...)[i, :]
    """

    if (initial_vector.shape != drift_vector.shape) | (diffusion_vector.shape != drift_vector.shape):
        raise ValueError("The inputs for drift, diffusion and initial positions have different dimensions. Check!")
    
    n_dim = initial_vector.shape[0]

    increments = se.generate_uncorrelated_white_noise(int(granularity * T), n_paths, n_dim)
    correlated_increments = se.correlate_noise(increments, correlation_matrix)
    correlated_increments[0, :, :] = 0.
    correlated_increments = correlated_increments * diffusion_vector / np.sqrt(granularity) + drift_vector / granularity
    paths = initial_vector * np.exp(correlated_increments.cumsum(axis = 0))

    if visualize == True:
        plt.figure(figsize = (10, 6), dpi = 150)
        plt.plot(paths)
        plt.title('Geometric Brownian Motion')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.show()

    return paths