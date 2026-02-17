import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import stochastic_engine as se
import jax.numpy as jnp

sns.set_theme()

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

def geometric_brownian_motion(
    initial_vector=100,
    drift_vector=0.1,
    diffusion_vector=0.2,
    correlation_matrix=np.eye(1),
    T=1,
    granularity=252,
    n_paths=100,
    visualize=False,
    method='antithetic_variates'
):
    """
    Simulates correlated multidimensional Geometric Brownian Motion (GBM) paths.

    This function generates Monte Carlo paths for a vector of assets following 
    correlated GBM dynamics under the physical (real-world) measure:
        dS_i(t) = mu_i * S_i(t) dt + sigma_i * S_i(t) dW_i(t),
    where the Brownian motions W_i are correlated according to `correlation_matrix`.

    Parameters
    ----------
    initial_vector : array-like, shape (n_dim,) or scalar
        Initial asset prices S(0). If scalar, broadcast to all dimensions.
    drift_vector : array-like, shape (n_dim,) or scalar
        Annualized drift coefficients (mu). If scalar, applied to all assets.
    diffusion_vector : array-like, shape (n_dim,) or scalar
        Annualized volatility coefficients (sigma). If scalar, applied to all assets.
    correlation_matrix : array-like, shape (n_dim, n_dim)
        Positive semi-definite correlation matrix for the underlying Brownian motions.
        Must be symmetric with ones on the diagonal.
    T : float
        Time horizon in years (e.g., T=1 for 1 year).
    granularity : int
        Number of time steps per year (e.g., 252 for daily steps in trading calendar).
        Total number of simulation steps: N = int(granularity * T).
    n_paths : int
        Number of independent Monte Carlo paths to simulate.
    visualize : bool
        If True, plots all simulated paths for each asset dimension.
    method : str
        Variance reduction method for noise generation (e.g., 'antithetic_variates').

    Returns
    -------
    paths : ndarray, shape (N + 1, n_paths, n_dim)
        Simulated GBM paths. 
        - Axis 0: time steps from t=0 to t=T (inclusive), total N+1 points.
        - Axis 1: independent Monte Carlo paths.
        - Axis 2: asset dimensions.

    Notes
    -----
    - The time step size is dt = T / N, where N = int(granularity * T).
    - Drift and diffusion parameters are interpreted in annualized terms.
    - The first time slice (paths[0, :, :]) equals `initial_vector` for all paths.
    - To extract the i-th path for asset j: paths[:, i, j]
    - To extract all paths at time step k: paths[k, :, :]
    - The current implementation forces the first increment to zero; this may cause
      a slight bias in the effective time horizon (T_eff = (N-1)/N * T).

    Examples
    --------
    >>> paths = geometric_brownian_motion(
    ...     initial_vector=[100, 95],
    ...     drift_vector=[0.05, 0.03],
    ...     diffusion_vector=[0.2, 0.25],
    ...     correlation_matrix=[[1, 0.5], [0.5, 1]],
    ...     T=1.0,
    ...     granularity=252,
    ...     n_paths=1000
    ... )
    >>> print(paths.shape)  # (253, 1000, 2)
    """

    if (initial_vector.shape != drift_vector.shape) | (diffusion_vector.shape != drift_vector.shape):
        raise ValueError("The inputs for drift, diffusion and initial positions have different dimensions. Check!")
    
    n_dim = initial_vector.shape[0]
    N = int(granularity * T)
    dt = T / N

    increments = se.generate_uncorrelated_white_noise(N, n_paths, n_dim, method=method)
    correlated_increments = se.correlate_noise(increments, correlation_matrix)

    # Log-return increments: shape (N, n_paths, n_dim)
    log_increments = (correlated_increments * diffusion_vector * jnp.sqrt(dt) + (drift_vector - 0.5 * diffusion_vector**2) * dt)

    # Cumulative log returns from time 0 to T (N steps)
    cum_log_returns = jnp.concatenate([jnp.zeros((1, n_paths, n_dim)), jnp.cumsum(log_increments, axis=0)], axis=0)

    # Final paths: S(t) = S0 * exp(cum_log_return)
    paths = initial_vector * jnp.exp(cum_log_returns)

    return paths

def ornstein_uhlenbeck_process(initial = 0, long_term = 0, diffusion = 10, mean_reversion_coef = 1, T = 1, sim_num = 100, visualize = False):
    """
    Generates a 1D Orstein-Uhlenbeck process
    Can also be used to generate Vasicek and Heston (sqrt(v))

    Inputs:
    - initial: initial value of the brownian motion. Default is 0.
    - long-term: the value to which the process reverts. Default is 0.
    - diffusion: diffusion coefficient (sigma) in annual terms. Default is 10.
    - mean_reversion_coef: how quickly the process reverts to the long-term mean. Default is 1.
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

    increments = st.norm.rvs(loc = (mean_reversion_coef * long_term) / 252, scale = diffusion / np.sqrt(252), size = (round(T * 252) + 1, sim_num))
    paths = np.zeros((round(T * 252) + 1, sim_num))
    paths[0, :] = initial
    for i in range(1, round(T * 252) + 1):
        paths[i, :] = paths[i-1, :] + increments[i, :] - mean_reversion_coef * paths[i-1, :] / 252

    if visualize == True:
        plt.figure(figsize = (10, 6), dpi = 150)
        plt.plot(paths)
        plt.title('Algebraic Brownian Motion')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.show()

    return paths
