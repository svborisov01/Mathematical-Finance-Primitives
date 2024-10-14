import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

sns.set_theme()

def random_walk(initial = 0, prob_up = 0.5, size_up = 1, size_down = 1, T = 1, sim_num = 100, visualize = False):
    """
    Generates a 1D random walk

    Inputs:
    - initial: initial value of the random walk. Default is 0.
    - prob_up: probability of going up. Default is 0.5. Probability of going down is defined as 1 - prob_up.
    - size_up: size of the upward step. Default is 1.
    - size_down: size of the downward step. Default is 1.
    - T: time in years. Default is 1. To get the number of steps, multiply T by 252.
    - sim_num: number of simulations to run. Default is 100.
    - visualize: whether to plot the paths. Default is False.

    Ouputs:
    - paths: 2D array of paths of the simulation with dimensions (round(T * 252) + 1, sim_num).
    - (optional) plot of the paths.

    Notes:
    - the simulation is (round(T * 252) + 1) in length to account for the initial value.
    - to select a single path, use random_walk(...)[:, i]
    - to select a single point in time across paths, use randomm_walk(...)[i, :]
    """
    increments = st.bernoulli.rvs(prob_up, size = (round(T * 252) + 1, sim_num)) * (size_up + size_down) - size_down
    increments[0,:] = initial
    paths = increments.cumsum(axis = 0)

    if visualize == True:
        plt.figure(figsize = (10, 6), dpi = 150)
        plt.plot(paths)
        plt.title('Random walk')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.show()

    return paths

def algebraic_brownian_motion(initial = 0, drift = 0, diffusion = 10, T = 1, sim_num = 100, visualize = False):
    """
    Generates a 1D algebraic brownian motion

    Inputs:
    - initial: initial value of the brownian motion. Default is 0.
    - drift: drift coefficient (mu) in annual terms. Default is 0.
    - diffusion: diffusion coefficient (sigma) in annual terms. Default is 10.
    - T: time in years. Default is 1. To get the number of steps, multiply T by 252.
    - sim_num: number of simulations to run. Default is 100.
    - visualize: whether to plot the paths. Default is False.

    Ouputs:
    - paths: 2D array of paths of the simulation with dimensions (round(T * 252) + 1, sim_num).
    - (optional) plot of the paths.

    Notes:
    - the simulation is (round(T * 252) + 1) in length to account for the initial value.
    - to select a single path, use algebraic_brownian_motion(...)[:, i]
    - to select a single point in time across paths, use algebraic_brownian_motion(...)[i, :]
    """

    increments = st.norm.rvs(loc = drift / 252, scale = diffusion / np.sqrt(252), size = (round(T * 252) + 1, sim_num))
    increments[0, :] = initial
    paths = increments.cumsum(axis = 0)

    if visualize == True:
        plt.figure(figsize = (10, 6), dpi = 150)
        plt.plot(paths)
        plt.title('Algebraic Brownian Motion')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.show()

    return paths

def geometric_brownian_motion(initial = 100, drift = 0.1, diffusion = 0.2, T = 1, sim_num = 100, visualize = False):
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

    increments = st.norm.rvs(loc = drift / 252, scale = diffusion / np.sqrt(252), size = (round(T * 252) + 1, sim_num))
    increments[0, :] = np.zeros(len(increments[0, :]))
    paths = initial * np.exp(increments.cumsum(axis = 0))

    if visualize == True:
        plt.figure(figsize = (10, 6), dpi = 150)
        plt.plot(paths)
        plt.title('Geometric Brownian Motion')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.show()

    return paths

def multivariate_geometric_brownian_motion(initial = np.array([100, 100]), drift = np.array([0.1, 0.1]), diffusion = np.array([[0.2, 0.05], [0.05, 0.2]]), T = 1, sim_num = 100, visualize_single_trajectory = False, visualize_single_asset = False):
    """
    Generates N-dimensional Geometric Brownian Motion

    Inputs:
    - initial: initial values of the process (spot prices)
    - drift: array of drift coefficients (mu) in annual terms
    - diffusion: covariance matrix (sigma) in annual terms
    - T: time in years. The function simulates round(T * 252) + 1 steps
    - sim_num: number of paths to generate
    - visualize_single_trajectory: generate one path for all assets in the basket - for visual debugging only
    - visualize_single_asset: generate all paths, but for one asset from the basket - for visual debugging only

    Ouputs:
    - paths: 3D array of paths of the simulation with dimensions (round(T * 252) + 1, sim_num, n_assets)
    - (optional) plot of the paths

    Notes:
    - the simulation is round(T * 252) + 1 in length to account for initial value
    - always check that initial values array, drift array and covariance matrix have the same dimensions!
    - use paths[i, :, :] to choose one time
    - use paths[:, i, :] to choose one path
    - use paths[:, :, i] to choose one asset
    """

    n_assets = len(drift)

    increments = st.multivariate_normal.rvs(mean = drift / 252, cov = diffusion / np.sqrt(252), size = (round(T * 252) + 1, sim_num))
    increments[0, :, :] = np.zeros([sim_num, n_assets])
    paths = initial * np.exp(increments.cumsum(axis = 0))

    if visualize_single_trajectory == True:
        plt.figure(figsize = (10, 6), dpi = 150)
        plt.plot(paths[:, 0, :])
        plt.title('All assets in the first generated path')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.show()

    if visualize_single_trajectory == True:
        plt.figure(figsize = (10, 6), dpi = 150)
        plt.plot(paths[:, :, 0])
        plt.title('All paths for the first asset')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.show()

    return paths