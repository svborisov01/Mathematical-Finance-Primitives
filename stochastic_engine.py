import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.stats import norm

import scipy.stats as st
from scipy.stats import qmc
import time


def generate_uncorrelated_white_noise(steps, sim_dim, sim_num, seed = None, method = 'antithetic_variates'):
    
    """
    Generate uncorrelated white noise using various sampling methods.
    
    Parameters:
    -----------
    steps : int
        Number of time steps
    sim_dim : int
        Number of dimensions
    sim_num : int
        Number of simulations
    seed : int, optional
        Random seed for reproducibility. None value can be entered
    method : str, optional
        Sampling method: 
        - 'antithetic_variates': Antithetic variates variance reduction
        - 'quasi_mc': Quasi-Monte Carlo with Sobol sequences
        - 'generic_mc': Standard Monte Carlo sampling
    
    Returns:
    --------
    jnp.ndarray
        Array of shape (steps, sim_num, sim_dim) containing normal random variates
    
    Raises:
    -------
    ValueError
        If an invalid method is specified
    """

    if method == 'antithetic_variates':

        if sim_num % 2 != 0:
            sim_num += 1

        half_sim_num = sim_num // 2

        if seed is None:
            seed = int(time.time() * 1000) % (2 ** 32 - 1)

        key = jrandom.PRNGKey(seed)
        key, subkey = jrandom.split(key)
        increments = jrandom.normal(subkey, (steps, half_sim_num, sim_dim))
        increments = jnp.concatenate([increments, -increments], axis = 1)

    elif method == 'quasi_mc':
        if seed is None:
            seed = int(time.time() * 1000) % (2 ** 32 - 1)
        
        total_samples_needed = steps * sim_num
        total_sobol_samples = 2 ** jnp.ceil(jnp.log2(total_samples_needed)).astype(int)

        # Initialize Sobol sequence generator
        sobol_engine = qmc.Sobol(d=sim_dim, seed=seed, scramble=True)
        
        # Generate single long Sobol sequence of power-of-2 length
        sobol_points = sobol_engine.random(n=total_sobol_samples)
        
        # Take only the needed number of samples
        sobol_points = sobol_points[:total_samples_needed]
        
        # Reshape to (steps, sim_num, sim_dim)
        sobol_points_reshaped = sobol_points.reshape(steps, sim_num, sim_dim)
        
        # Transform to normal distribution
        increments = norm.ppf(sobol_points_reshaped)
        
        # Convert to JAX array
        increments = jnp.array(increments)

    elif method == 'generic_mc':
        increments = st.norm.rvs(size = (steps, sim_num, sim_dim))

    else:
        raise ValueError("Incorrect method selected. Please specify explicitly from the available options")
    return increments