import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.stats import norm
from jax.scipy.linalg import cholesky

import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import qmc
import time

def generate_uncorrelated_white_noise(steps, sim_dim, sim_num, seed = None, method = 'antithetic_variates'):

    """
    Generate uncorrelated white noise increments for stochastic process simulation.
    
    This function produces normally distributed random increments suitable for use in
    stochastic differential equation simulations such as Geometric Brownian Motion.
    
    Parameters:
    -----------
    steps : int
        Number of discrete time steps for the simulation.
        Must be a positive integer.
    
    sim_dim : int
        Number of independent dimensions/processes to simulate.
        Must be a positive integer.
    
    sim_num : int
        Number of Monte Carlo simulation paths to generate.
        Must be a positive integer. For 'antithetic_variates' method, this will be
        rounded up to the nearest even number if odd.
    
    seed : int, optional
        Random seed for reproducibility. If None, a seed is generated based on
        current time. Default is None.
    
    method : str, optional
        Sampling method to use for random number generation. Available options:
        - 'antithetic_variates': Uses antithetic variates for variance reduction.
                                 Generates pairs of paths (X, -X) for better
                                 convergence properties.
        - 'quasi_mc': Uses Quasi-Monte Carlo with Sobol sequences for improved
                     convergence. Suitable for high-dimensional problems.
        - 'generic_mc': Uses standard Monte Carlo sampling with pseudo-random numbers.
                       Fast but slower convergence compared to other methods.
        Default is 'antithetic_variates'.
    
    Returns:
    --------
    jnp.ndarray
        Array of shape (steps, sim_dim, sim_num) containing uncorrelated standard
        normal random variates (mean=0, variance=1). The array structure is:
        - Axis 0: Time steps (0 to steps-1)
        - Axis 1: Dimensions (0 to sim_dim-1) 
        - Axis 2: Simulation paths (0 to sim_num-1)
    
    Raises:
    -------
    ValueError
        If an invalid method is specified or if input parameters are not positive integers.
    
    Notes:
    ------
    - For 'antithetic_variates' method, the actual number of paths may be increased
      to the next even number to ensure proper pairing.
    - For 'quasi_mc' method, the number of samples per sequence is increased to the
      next power of 2 to satisfy Sobol sequence requirements.

    """
    
    if method == 'antithetic_variates':

        if sim_num % 2 != 0:
            sim_num += 1

        half_sim_num = sim_num // 2

        if seed is None:
            seed = int(time.time() * 1000) % (2 ** 32 - 1)

        key = jrandom.PRNGKey(seed)
        key, subkey = jrandom.split(key)
        increments = jrandom.normal(subkey, (steps, sim_dim, half_sim_num))
        increments = jnp.concatenate([increments, -increments], axis = 2)

    elif method == 'quasi_mc':
        if seed is None:
            seed = int(time.time() * 1000) % (2 ** 32 - 1)
        
        total_sequences = steps * sim_dim
        points_per_sequence = 2 ** jnp.ceil(jnp.log2(sim_num)).astype(int)

        # Generate all Sobol sequences
        all_sobol_points = []
        for i in range(total_sequences):
            seq_seed = seed + i
            sobol_engine = qmc.Sobol(d=1, seed=seq_seed, scramble=True)
            points = sobol_engine.random(n=points_per_sequence)[:sim_num]
            all_sobol_points.append(points.flatten())
        
        sobol_array = jnp.array(all_sobol_points).reshape(steps, sim_dim, sim_num)
        increments = jnp.array(norm.ppf(sobol_array))
    
        # Transpose to get (steps, sim_num, sim_dim)
        increments = jnp.transpose(increments, (0, 1, 2))

    elif method == 'generic_mc':
        increments = st.norm.rvs(size = (steps, sim_dim, sim_num))

    else:
        raise ValueError("Incorrect method selected. Please specify explicitly from the available options")
    return increments


def validate_generation(paths):

    """
    Service function used for validating the independence and first two moments of the generated sample

    Inputs:
    - paths array of the shape (steps, sim_dim, sim_num)

    Outputs:
    - graphs, mean and std values across one dimension (total 3)
    """
    
    plt.plot(paths[:, 0, 0])
    plt.title('Autocorrelation check')
    plt.show()
    print(paths[:, 0, 0].mean(), paths[:, 0, 0].std())

    plt.plot(paths[0, 0, :])
    plt.title('Cross-paths check')
    plt.show()
    print(paths[0, 0, :].mean(), paths[0, 0, :].std())

    plt.plot(paths[0, :, 0])
    plt.title('Cross-dimensions check')
    plt.show()
    print(paths[0, :, 0].mean(), paths[0, :, 0].std())

def correlate_noise(uncorrelated_array, correlation_matrix):
    """
    Apply correlation using batched matrix multiplication for better performance.
    """
    # Validate and compute Cholesky decomposition
    steps, sim_dim, sim_num = uncorrelated_array.shape
    
    if correlation_matrix.shape != (sim_dim, sim_dim):
        raise ValueError("correlation_matrix dimensions must match sim_dim")
    
    L = cholesky(correlation_matrix, lower=True)
    
    # Use einsum for efficient batched matrix multiplication
    correlated = jnp.einsum('ilk,lj->ijk', uncorrelated_array, L.T)
    
    return correlated

def validate_correlating():
    paths = generate_uncorrelated_white_noise(1000, 3, 10)
    correlation_matrix = jnp.array([[1, 0.9, 0.7], [0.9, 1, 0.3], [0.1, 0.5, 1.]])
    return jnp.corrcoef(correlate_noise(paths, correlation_matrix=correlation_matrix)[:, :, 0].T)