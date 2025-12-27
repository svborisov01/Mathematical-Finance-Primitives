import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.stats import norm
from jax.scipy.linalg import cholesky

import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import qmc
import time

def generate_uncorrelated_white_noise(n_steps, n_paths, n_dim, seed=None, method='antithetic_variates'):
    """
    Generate uncorrelated white noise increments for stochastic process simulation.

    This function produces normally distributed random increments suitable for use in
    stochastic differential equation simulations such as Geometric Brownian Motion.

    Parameters:
    -----------
    n_steps : int
        Number of discrete time steps for the simulation.
        Must be a positive integer.

    n_paths : int
        Number of Monte Carlo simulation paths to generate.
        Must be a positive integer. For 'antithetic_variates' method, this will be
        rounded up to the nearest even number if odd.

    n_dim : int
        Number of independent dimensions/processes to simulate.
        Must be a positive integer.

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
        Array of shape (n_steps, n_paths, n_dim) containing uncorrelated standard
        normal random variates (mean=0, variance=1). The array structure is:
        - Axis 0: Time steps (0 to n_steps - 1)
        - Axis 1: Simulation paths (0 to n_paths - 1)
        - Axis 2: Dimensions (0 to n_dim - 1)

    Raises:
    -------
    ValueError
        If an invalid method is specified or if input parameters are not positive integers.
        If n_steps, n_paths, or n_dim are not positive integers.

    Notes:
    ------
    - For 'antithetic_variates' method, the actual number of paths may be increased
      to the next even number to ensure proper pairing.
    - For 'quasi_mc' method, the number of samples per sequence is increased to the
      next power of 2 to satisfy Sobol sequence requirements.
    """

    # Input validation
    if not (isinstance(n_steps, int) and n_steps > 0):
        raise ValueError("n_steps must be a positive integer.")
    if not (isinstance(n_paths, int) and n_paths > 0):
        raise ValueError("n_paths must be a positive integer.")
    if not (isinstance(n_dim, int) and n_dim > 0):
        raise ValueError("n_dim must be a positive integer.")

    if method == 'antithetic_variates':
        orig_n_paths = n_paths
        if n_paths % 2 != 0:
            n_paths = n_paths + 1

        half_n_paths = n_paths // 2

        if seed is None:
            seed = int(time.time() * 1000) % (2 ** 32 - 1)

        key = jrandom.PRNGKey(seed)
        key, subkey = jrandom.split(key)
        base_samples = jrandom.normal(subkey, (n_steps, half_n_paths, n_dim))
        increments = jnp.concatenate([base_samples, -base_samples], axis=1)

        # Trim if we added an extra path
        increments = increments[:, :orig_n_paths, :]

    elif method == 'quasi_mc':
        if seed is None:
            seed = int(time.time() * 1000) % (2 ** 32 - 1)

        total_sequences = n_steps * n_dim
        points_per_sequence = 2 ** int(jnp.ceil(jnp.log2(n_paths)))

        all_sobol_points = []
        for i in range(total_sequences):
            seq_seed = seed + i
            sobol_engine = qmc.Sobol(d=1, seed=seq_seed, scramble=True)
            points = sobol_engine.random(n=points_per_sequence)[:n_paths]
            all_sobol_points.append(points.flatten())

        sobol_array = jnp.array(all_sobol_points).reshape(n_steps, n_dim, n_paths)
        # Now transpose to (n_steps, n_paths, n_dim)
        sobol_array = jnp.transpose(sobol_array, (0, 2, 1))
        increments = norm.ppf(sobol_array)

    elif method == 'generic_mc':
        # Using scipy for compatibility; consider using jax.random for full JAX compatibility
        increments = st.norm.rvs(size=(n_steps, n_paths, n_dim))

    else:
        raise ValueError("Incorrect method selected. Please specify explicitly from the available options: "
                         "'antithetic_variates', 'quasi_mc', or 'generic_mc'.")

    return increments


def validate_generation(paths):
    """
    Validate the independence and first two moments of the generated white noise sample.

    Parameters:
    -----------
    paths : jnp.ndarray or np.ndarray
        Array of shape (n_steps, n_paths, n_dim)

    Outputs:
    --------
    Displays three plots and prints mean/std for:
    1. Time series of a single path and dimension (autocorrelation check)
    2. Cross-section across paths at fixed time and dimension (path independence)
    3. Cross-section across dimensions at fixed time and path (dimension independence)
    """

    n_steps, n_paths, n_dim = paths.shape

    # 1. Autocorrelation check: one path, one dimension over time
    plt.figure()
    plt.plot(paths[:, 0, 0])
    plt.title('Autocorrelation check (single path, single dim over time)')
    plt.xlabel('Time step')
    plt.ylabel('Noise value')
    plt.show()
    print(f"Time series (path 0, dim 0): mean = {paths[:, 0, 0].mean():.4f}, std = {paths[:, 0, 0].std():.4f}")

    # 2. Cross-paths check: fixed time and dimension, vary path
    plt.figure()
    plt.plot(paths[0, :, 0])
    plt.title('Cross-paths check (fixed time=0, dim=0)')
    plt.xlabel('Path index')
    plt.ylabel('Noise value')
    plt.show()
    print(f"Cross-paths (time 0, dim 0): mean = {paths[0, :, 0].mean():.4f}, std = {paths[0, :, 0].std():.4f}")

    # 3. Cross-dimensions check: fixed time and path, vary dimension
    plt.figure()
    plt.plot(paths[0, 0, :])
    plt.title('Cross-dimensions check (fixed time=0, path=0)')
    plt.xlabel('Dimension index')
    plt.ylabel('Noise value')
    plt.show()
    print(f"Cross-dimensions (time 0, path 0): mean = {paths[0, 0, :].mean():.4f}, std = {paths[0, 0, :].std():.4f}")

def correlate_noise(uncorrelated_array, correlation_matrix):
    """
    Apply correlation to uncorrelated white noise using batched matrix multiplication.

    Parameters:
    -----------
    uncorrelated_array : jnp.ndarray
        Array of shape (n_steps, n_paths, n_dim) containing uncorrelated standard normal samples.
    
    correlation_matrix : jnp.ndarray
        Square positive semi-definite matrix of shape (n_dim, n_dim) specifying the target
        correlation structure across dimensions.

    Returns:
    --------
    jnp.ndarray
        Array of shape (n_steps, n_paths, n_dim) with the specified cross-dimensional correlation
        applied, while preserving independence across time steps and paths.

    Notes:
    ------
    Correlation is applied independently at each (step, path) by treating the dimension axis
    as a vector to be linearly transformed via the Cholesky factor of the correlation matrix.
    """
    n_steps, n_paths, n_dim = uncorrelated_array.shape

    if correlation_matrix.shape != (n_dim, n_dim):
        raise ValueError(f"correlation_matrix must be of shape ({n_dim}, {n_dim}) to match n_dim={n_dim}")

    # Compute Cholesky decomposition (lower triangular)
    L = cholesky(correlation_matrix, lower=True)  # shape: (n_dim, n_dim)

    # Apply correlation: for each (step, path), multiply the n_dim-vector by L
    # uncorrelated_array: (n_steps, n_paths, n_dim)
    # We want: correlated = uncorrelated_array @ L.T
    # Using einsum: 'ijk,kl -> ijl' → but simpler to use matmul
    correlated = jnp.matmul(uncorrelated_array, L.T)

    return correlated


def validate_correlating():
    """
    Validate that the correlation structure is correctly imposed.
    """
    # Generate uncorrelated noise: (n_steps=1000, n_paths=10, n_dim=3)
    paths = generate_uncorrelated_white_noise(n_steps=1000, n_paths=10, n_dim=3, method='generic_mc', seed=42)

    # Define a valid correlation matrix (symmetric, positive semi-definite)
    correlation_matrix = jnp.array([
        [1.0, 0.9, 0.7],
        [0.9, 1.0, 0.5],
        [0.7, 0.5, 1.0]
    ])

    print("Target correlation matrix:")
    print(correlation_matrix)

    # Apply correlation
    correlated_paths = correlate_noise(paths, correlation_matrix=correlation_matrix)

    # To estimate empirical correlation, we need many samples across the same dimension pair.
    # Since paths are independent and identically distributed, we can flatten over time and paths.
    # Shape: (n_steps * n_paths, n_dim)
    samples = correlated_paths.reshape(-1, 3)  # stack all (step, path) samples

    # Compute empirical correlation matrix
    empirical_corr = jnp.corrcoef(samples.T)  # corrcoef expects variables in rows → use .T

    print("\nEmpirical correlation matrix (from 1000*10 = 10,000 samples):")
    print(empirical_corr)

    return empirical_corr
    