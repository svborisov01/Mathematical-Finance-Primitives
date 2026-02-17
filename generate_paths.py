import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import stochastic_engine as se
import jax.numpy as jnp

sns.set_theme()

def algebraic_brownian_motion(
    initial_vector=0.0,
    drift_vector=0.0,
    diffusion_vector=10.0,
    correlation_matrix=None,
    T=1.0,
    granularity=252,
    n_paths=100,
    visualize=False,
    method='antithetic_variates'
):
    """
    Simulates correlated multidimensional Arithmetic (Algebraic) Brownian Motion (ABM) paths.

    This function generates Monte Carlo paths for a vector of processes following 
    correlated ABM dynamics:
        dX_i(t) = mu_i dt + sigma_i dW_i(t),
    where the Brownian motions W_i are correlated according to `correlation_matrix`.

    Parameters
    ----------
    initial_vector : array-like, shape (n_dim,) or scalar
        Initial values X(0). If scalar, broadcast to all dimensions.
    drift_vector : array-like, shape (n_dim,) or scalar
        Annualized drift coefficients (mu). If scalar, applied to all assets.
    diffusion_vector : array-like, shape (n_dim,) or scalar
        Annualized volatility coefficients (sigma). If scalar, applied to all assets.
    correlation_matrix : array-like, shape (n_dim, n_dim), optional
        Positive semi-definite correlation matrix for the underlying Brownian motions.
        If None and n_dim > 1, defaults to identity. For n_dim=1, ignored.
    T : float
        Time horizon in years (e.g., T=1 for 1 year).
    granularity : int
        Number of time steps per year (e.g., 252 for daily steps).
        Total number of simulation steps: N = int(granularity * T).
    n_paths : int
        Number of independent Monte Carlo paths to simulate.
    visualize : bool
        If True, plots all simulated paths for each dimension.
    method : str
        Variance reduction method for noise generation (e.g., 'antithetic_variates').

    Returns
    -------
    paths : jnp.ndarray, shape (N + 1, n_paths, n_dim)
        Simulated ABM paths.
        - Axis 0: time steps from t=0 to t=T (inclusive), total N+1 points.
        - Axis 1: independent Monte Carlo paths.
        - Axis 2: process dimensions.

    Notes
    -----
    - The time step size is dt = T / N, where N = int(granularity * T).
    - Drift and diffusion parameters are interpreted in annualized terms.
    - The first time slice (paths[0, :, :]) equals `initial_vector` for all paths.
    - To extract the i-th path for dimension j: paths[:, i, j]
    - To extract all paths at time step k: paths[k, :, :]
    - Fully compatible with JAX transformations (grad, vmap, etc.) for AAD.
    - Correlation sensitivities can be computed via jax.grad w.r.t. correlation_matrix.
    - Uses the provided `correlate_noise` function for consistent correlation handling.

    Examples
    --------
    >>> paths = algebraic_brownian_motion(
    ...     initial_vector=[0, 0],
    ...     drift_vector=[0.5, -0.2],
    ...     diffusion_vector=[5, 8],
    ...     correlation_matrix=[[1, 0.4], [0.4, 1]],
    ...     T=1.0,
    ...     granularity=252,
    ...     n_paths=1000
    ... )
    >>> print(paths.shape)  # (253, 1000, 2)
    """
    # Convert inputs to JAX arrays and ensure proper shapes
    initial_vector = jnp.atleast_1d(initial_vector)
    drift_vector = jnp.atleast_1d(drift_vector)
    diffusion_vector = jnp.atleast_1d(diffusion_vector)

    if not (initial_vector.shape == drift_vector.shape == diffusion_vector.shape):
        raise ValueError("initial_vector, drift_vector, and diffusion_vector must have the same shape.")

    n_dim = initial_vector.shape[0]

    # Handle correlation matrix
    if correlation_matrix is None:
        correlation_matrix = jnp.eye(n_dim)
    else:
        correlation_matrix = jnp.asarray(correlation_matrix)
        if correlation_matrix.shape != (n_dim, n_dim):
            raise ValueError(f"correlation_matrix must be ({n_dim}, {n_dim})")

    # Time discretization
    N = int(granularity * T)
    if N <= 0:
        raise ValueError("granularity * T must be positive")
    dt = T / N
    sqrt_dt = jnp.sqrt(dt)

    # Generate uncorrelated noise: shape (N, n_paths, n_dim)
    increments = se.generate_uncorrelated_white_noise(N, n_paths, n_dim, method=method)

    # Apply correlation using the provided utility function
    correlated_increments = se.correlate_noise(increments, correlation_matrix)

    # Scale by diffusion and add drift: dX = mu * dt + sigma * sqrt(dt) * dW_corr
    abm_increments = (
        drift_vector * dt +
        diffusion_vector * sqrt_dt * correlated_increments
    )  # shape: (N, n_paths, n_dim)

    # Build full paths: start at X0, then cumulative sum of increments
    # Prepend a zero-increment row for t=0
    zero_row = jnp.zeros((1, n_paths, n_dim))
    all_increments = jnp.concatenate([zero_row, abm_increments], axis=0)  # (N+1, n_paths, n_dim)
    paths = initial_vector + jnp.cumsum(all_increments, axis=0)

    # Visualization
    if visualize:
        for dim in range(n_dim):
            plt.figure(figsize=(10, 6), dpi=150)
            plt.plot(paths[:, :, dim])
            plt.title(f'Arithmetic Brownian Motion (Dimension {dim})')
            plt.xlabel('Time Step')
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

def ornstein_uhlenbeck_process(
    initial_vector=0.0,
    long_term_vector=0.0,
    diffusion_vector=10.0,
    mean_reversion_coef=1.0,
    correlation_matrix=None,
    T=1.0,
    granularity=252,
    n_paths=100,
    visualize=False,
    method='antithetic_variates'
):
    """
    Simulates correlated multidimensional Ornstein-Uhlenbeck (OU) processes.

    The dynamics follow:
        dX_i(t) = κ_i (θ_i - X_i(t)) dt + σ_i dW_i(t),
    where W_i are correlated Brownian motions with correlation matrix `correlation_matrix`.

    This includes the Vasicek model (for interest rates) and can be used as a building
    block for stochastic volatility models (e.g., Heston variance process).

    Parameters
    ----------
    initial_vector : array-like, shape (n_dim,) or scalar
        Initial values X(0). If scalar, broadcast to all dimensions.
    long_term_vector : array-like, shape (n_dim,) or scalar
        Long-term mean levels θ. If scalar, applied to all dimensions.
    diffusion_vector : array-like, shape (n_dim,) or scalar
        Annualized volatility coefficients σ. If scalar, applied to all dimensions.
    mean_reversion_coef : array-like, shape (n_dim,) or scalar
        Mean reversion speeds κ > 0. If scalar, applied to all dimensions.
    correlation_matrix : array-like, shape (n_dim, n_dim), optional
        Positive semi-definite correlation matrix for the underlying Brownian motions.
        If None and n_dim > 1, defaults to identity. For n_dim=1, ignored.
    T : float
        Time horizon in years (e.g., T=1 for 1 year).
    granularity : int
        Number of time steps per year (e.g., 252 for daily steps).
        Total number of simulation steps: N = int(granularity * T).
    n_paths : int
        Number of independent Monte Carlo paths to simulate.
    visualize : bool
        If True, plots all simulated paths for each dimension.
    method : str
        Variance reduction method for noise generation (e.g., 'antithetic_variates').

    Returns
    -------
    paths : jnp.ndarray, shape (N + 1, n_paths, n_dim)
        Simulated OU paths.
        - Axis 0: time steps from t=0 to t=T (inclusive), total N+1 points.
        - Axis 1: independent Monte Carlo paths.
        - Axis 2: process dimensions.

    Notes
    -----
    - Uses Euler-Maruyama discretization:
          X_{t+dt} = X_t + κ(θ - X_t)dt + σ√dt · dW_t
    - Time step: dt = T / N, where N = int(granularity * T)
    - Fully compatible with JAX transformations (grad, vmap, etc.) for AAD.
    - Correlation sensitivities can be computed via jax.grad w.r.t. correlation_matrix.
    - Reuses the provided `se.correlate_noise` function for consistent correlation handling.

    Examples
    --------
    >>> paths = ornstein_uhlenbeck_process(
    ...     initial_vector=[0.05, 0.03],
    ...     long_term_vector=[0.04, 0.04],
    ...     diffusion_vector=[0.01, 0.015],
    ...     mean_reversion_coef=[0.8, 1.2],
    ...     correlation_matrix=[[1, 0.3], [0.3, 1]],
    ...     T=1.0,
    ...     granularity=252,
    ...     n_paths=1000
    ... )
    >>> print(paths.shape)  # (253, 1000, 2)
    """
    # Convert inputs to JAX arrays and ensure proper shapes
    initial_vector = jnp.atleast_1d(initial_vector)
    long_term_vector = jnp.atleast_1d(long_term_vector)
    diffusion_vector = jnp.atleast_1d(diffusion_vector)
    mean_reversion_coef = jnp.atleast_1d(mean_reversion_coef)

    shapes = [initial_vector.shape, long_term_vector.shape, diffusion_vector.shape, mean_reversion_coef.shape]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError("All input vectors must have the same shape.")

    n_dim = initial_vector.shape[0]

    # Handle correlation matrix
    if correlation_matrix is None:
        correlation_matrix = jnp.eye(n_dim)
    else:
        correlation_matrix = jnp.asarray(correlation_matrix)
        if correlation_matrix.shape != (n_dim, n_dim):
            raise ValueError(f"correlation_matrix must be ({n_dim}, {n_dim})")

    # Time discretization
    N = int(granularity * T)
    if N <= 0:
        raise ValueError("granularity * T must be positive")
    dt = T / N
    sqrt_dt = jnp.sqrt(dt)

    # Generate uncorrelated noise: shape (N, n_paths, n_dim)
    dW = se.generate_uncorrelated_white_noise(N, n_paths, n_dim, method=method)

    # Apply correlation using the provided utility
    correlated_dW = se.correlate_noise(dW, correlation_matrix)  # (N, n_paths, n_dim)

    # Pre-allocate path array
    paths = jnp.zeros((N + 1, n_paths, n_dim))
    paths = paths.at[0].set(initial_vector)  # Set initial values at t=0

    # Euler-Maruyama update: vectorized over time using scan or loop
    # Since JAX doesn't allow in-place updates, we use a functional loop
    def _step(carry, dw_step):
        x_prev = carry  # (n_paths, n_dim)
        # Drift: κ(θ - x_prev) * dt
        drift = mean_reversion_coef * (long_term_vector - x_prev) * dt
        # Diffusion: σ * sqrt(dt) * dW
        diffusion = diffusion_vector * sqrt_dt * dw_step  # (n_paths, n_dim)
        x_next = x_prev + drift + diffusion
        return x_next, x_next

    # Run the recurrence
    _, xs = jax.lax.scan(_step, initial_vector, correlated_dW)  # xs: (N, n_paths, n_dim)

    # Concatenate initial state with evolved states
    paths = jnp.concatenate([paths[:1], xs], axis=0)  # (N+1, n_paths, n_dim)

    # Visualization
    if visualize:
        for dim in range(n_dim):
            plt.figure(figsize=(10, 6), dpi=150)
            plt.plot(paths[:, :, dim])
            plt.title(f'Ornstein-Uhlenbeck Process (Dimension {dim})')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.show()

    return paths
