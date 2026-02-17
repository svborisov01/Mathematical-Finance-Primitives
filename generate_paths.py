import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import stochastic_engine as se
import jax.numpy as jnp
import jax
import warnings

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

    This function generates Monte Carlo paths for a vector of mean-reverting processes
    following the dynamics:
        dX_i(t) = κ_i (θ_i - X_i(t)) dt + σ_i dW_i(t),
    where W_i are correlated Brownian motions with correlation matrix `correlation_matrix`.
    This includes the Vasicek interest rate model as a special case.

    Parameters
    ----------
    initial_vector : array-like, shape (n_dim,) or scalar
        Initial values X(0). If scalar, broadcast to all dimensions.
    long_term_vector : array-like, shape (n_dim,) or scalar
        Long-term mean reversion levels θ_i. If scalar, applied to all dimensions.
    diffusion_vector : array-like, shape (n_dim,) or scalar
        Annualized volatility coefficients σ_i. If scalar, applied to all dimensions.
    mean_reversion_coef : array-like, shape (n_dim,) or scalar
        Mean reversion speeds κ_i > 0. If scalar, applied to all dimensions.
    correlation_matrix : array-like, shape (n_dim, n_dim), optional
        Positive semi-definite correlation matrix for the underlying Brownian motions.
        Must be symmetric with ones on the diagonal. Defaults to identity if None.
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
          X_{t+dt} = X_t + κ(θ - X_t)dt + σ√dt · ΔW_t
    - Time step size: dt = T / N, where N = int(granularity * T).
    - All parameters are interpreted in annualized terms.
    - The first time slice (paths[0, :, :]) equals `initial_vector` for all paths.
    - Fully compatible with JAX transformations (`jax.grad`, `jax.vmap`, etc.) for AAD.
    - Correlation sensitivities can be computed via `jax.grad` w.r.t. `correlation_matrix`.
    - Reuses the provided `se.correlate_noise` function for consistent correlation handling.

    Examples
    --------
    >>> paths = ornstein_uhlenbeck_process(
    ...     initial_vector=[0.03, 0.05],
    ...     long_term_vector=[0.04, 0.04],
    ...     diffusion_vector=[0.01, 0.015],
    ...     mean_reversion_coef=[0.8, 1.2],
    ...     correlation_matrix=[[1.0, 0.3], [0.3, 1.0]],
    ...     T=1.0,
    ...     granularity=252,
    ...     n_paths=1000
    ... )
    >>> print(paths.shape)  # (253, 1000, 2)
    >>> # paths[:, :, 0] → short rate 1 (Vasicek), paths[:, :, 1] → short rate 2
    """

    # Normalize inputs to arrays
    initial_vector = jnp.atleast_1d(initial_vector)
    long_term_vector = jnp.atleast_1d(long_term_vector)
    diffusion_vector = jnp.atleast_1d(diffusion_vector)
    mean_reversion_coef = jnp.atleast_1d(mean_reversion_coef)

    shapes = [initial_vector.shape, long_term_vector.shape,
              diffusion_vector.shape, mean_reversion_coef.shape]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError("All parameter vectors must have the same shape.")
    n_dim = initial_vector.shape[0]

    # Handle correlation matrix
    if correlation_matrix is None:
        correlation_matrix = jnp.eye(n_dim)
    else:
        correlation_matrix = jnp.asarray(correlation_matrix)
        if correlation_matrix.shape != (n_dim, n_dim):
            raise ValueError(f"correlation_matrix must be ({n_dim}, {n_dim})")

    # Time grid
    N = int(granularity * T)
    if N <= 0:
        raise ValueError("granularity * T must be positive")
    dt = T / N
    sqrt_dt = jnp.sqrt(dt)

    # Generate noise
    dW = se.generate_uncorrelated_white_noise(N, n_paths, n_dim, method=method)
    correlated_dW = se.correlate_noise(dW, correlation_matrix)  # (N, n_paths, n_dim)

    # ✅ CRITICAL: Initialize carry with shape (n_paths, n_dim)
    initial_state = jnp.broadcast_to(initial_vector, (n_paths, n_dim))  # (n_paths, n_dim)

    def _step(x_prev, dw_step):
        # x_prev: (n_paths, n_dim)
        # dw_step: (n_paths, n_dim)
        drift = mean_reversion_coef * (long_term_vector - x_prev) * dt  # (n_paths, n_dim)
        diffusion = diffusion_vector * sqrt_dt * dw_step                 # (n_paths, n_dim)
        x_next = x_prev + drift + diffusion
        return x_next, x_next

    # Run scan
    _, xs = jax.lax.scan(_step, initial_state, correlated_dW)  # xs: (N, n_paths, n_dim)

    # Prepend initial state
    initial_expanded = initial_state[None, :, :]  # (1, n_paths, n_dim)
    paths = jnp.concatenate([initial_expanded, xs], axis=0)  # (N+1, n_paths, n_dim)

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

def hybrid_stochastic_process(
    process_specs,
    initial_vector,
    correlation_matrix=None,
    T=1.0,
    granularity=252,
    n_paths=100,
    visualize=False,
    method='antithetic_variates'
):
    """
    Simulates a hybrid multidimensional stochastic process where each dimension
    independently follows GBM, OU, ABM, or Feller (CIR) dynamics.

    Dynamics per dimension i:
      - 'gbm':   dS = μ S dt + σ S dW
      - 'ou':    dX = κ(θ − X) dt + σ dW
      - 'abm':   dY = μ dt + σ dW
      - 'feller': dv = κ(θ − v) dt + σ√v dW

    Parameters
    ----------
    process_specs : list[dict]
        Each dict must contain 'type' ∈ {'gbm', 'ou', 'abm', 'feller'} and:
        - 'gbm': {'drift', 'diffusion'}
        - 'ou':  {'long_term', 'diffusion', 'mean_reversion'}
        - 'abm': {'drift', 'diffusion'}
        - 'feller': {'long_term', 'diffusion', 'mean_reversion'}
    initial_vector : array-like, shape (n_dim,)
        Initial values (prices for GBM, levels for others, variance for Feller).
    ... (other params same as before) ...

    Returns
    -------
    paths : jnp.ndarray, shape (N+1, n_paths, n_dim)

    Notes
    -----
    - Feller dimensions are simulated in level space (not log).
    - Full truncation (√max(v,0)) ensures non-negativity.
    - A warning is issued if Feller condition (2κθ ≥ σ²) is violated.
    - All processes share correlated Brownian motions via `correlation_matrix`.
    """
    n_dim = len(process_specs)
    initial_vector = jnp.asarray(initial_vector, dtype=jnp.float32)
    if initial_vector.shape != (n_dim,):
        raise ValueError(f"`initial_vector` must have shape ({n_dim},)")

    # Initialize full parameter arrays
    drift_full = []
    diffusion_full = []
    long_term_full = []
    mean_rev_full = []
    process_types = []

    for spec in process_specs:
        typ = spec['type']
        process_types.append(typ)

        if typ == 'gbm':
            drift_full.append(spec['drift'])
            diffusion_full.append(spec['diffusion'])
            long_term_full.append(0.0)
            mean_rev_full.append(0.0)
        elif typ == 'ou':
            drift_full.append(0.0)
            diffusion_full.append(spec['diffusion'])
            long_term_full.append(spec['long_term'])
            mean_rev_full.append(spec['mean_reversion'])
        elif typ == 'abm':
            drift_full.append(spec['drift'])
            diffusion_full.append(spec['diffusion'])
            long_term_full.append(0.0)
            mean_rev_full.append(0.0)
        elif typ == 'feller':
            # Feller has no independent drift; use mean_rev & long_term for drift
            drift_full.append(0.0)  # unused
            diffusion_full.append(spec['diffusion'])
            long_term_full.append(spec['long_term'])
            mean_rev_full.append(spec['mean_reversion'])
        else:
            raise ValueError(f"Unknown process type: {typ}. Must be 'gbm', 'ou', 'abm', or 'feller'.")

    # Convert to arrays
    drift_full = jnp.array(drift_full, dtype=jnp.float32)
    diffusion_full = jnp.array(diffusion_full, dtype=jnp.float32)
    long_term_full = jnp.array(long_term_full, dtype=jnp.float32)
    mean_rev_full = jnp.array(mean_rev_full, dtype=jnp.float32)

    is_gbm = jnp.array([t == 'gbm' for t in process_types])
    is_feller = jnp.array([t == 'feller' for t in process_types])
    is_ou = jnp.array([t == 'ou' for t in process_types])
    is_abm = jnp.array([t == 'abm' for t in process_types])

    # 🔔 Feller condition warning (same logic as standalone function)
    feller_mask = is_feller
    if jnp.any(feller_mask):
        kappa = jnp.where(feller_mask, mean_rev_full, 1.0)
        theta = jnp.where(feller_mask, long_term_full, 1.0)
        sigma = jnp.where(feller_mask, diffusion_full, 1.0)
        feller_violated = (2.0 * kappa * theta < sigma ** 2) & feller_mask
        if jnp.any(feller_violated):
            violated_dims = jnp.where(feller_violated)[0].tolist()
            warnings.warn(
                f"Feller condition violated in dimensions {violated_dims}. "
                "Paths may hit zero.",
                UserWarning
            )

    # Correlation, time grid, noise — same as before
    if correlation_matrix is None:
        correlation_matrix = jnp.eye(n_dim)
    else:
        correlation_matrix = jnp.asarray(correlation_matrix, dtype=jnp.float32)
        if correlation_matrix.shape != (n_dim, n_dim):
            raise ValueError(f"correlation_matrix must be ({n_dim}, {n_dim})")

    N = int(granularity * T)
    if N <= 0:
        raise ValueError("granularity * T must be positive")
    dt = T / N
    sqrt_dt = jnp.sqrt(dt)

    dW = se.generate_uncorrelated_white_noise(N, n_paths, n_dim, method=method)
    correlated_dW = se.correlate_noise(dW, correlation_matrix)

    # Unified state: log(S) for GBM, level otherwise
    unified_initial = jnp.where(
        is_gbm,
        jnp.log(jnp.maximum(initial_vector, 1e-12)),  # avoid log(0)
        initial_vector
    )
    carry_init = jnp.broadcast_to(unified_initial, (n_paths, n_dim))

    def _step(unified_prev, dw_step):
        # Drift term
        drift_term = jnp.where(
            is_gbm,
            (drift_full - 0.5 * diffusion_full**2) * dt,
            jnp.where(
                is_ou | is_feller,
                mean_rev_full * (long_term_full - unified_prev) * dt,
                drift_full * dt  # ABM
            )
        )

        # Diffusion term
        diffusion_base = jnp.where(
            is_feller,
            jnp.sqrt(jnp.maximum(unified_prev, 0.0)),  # √v for Feller
            1.0
        )
        diffusion_term = diffusion_full * diffusion_base * sqrt_dt * dw_step

        unified_next = unified_prev + drift_term + diffusion_term
        return unified_next, unified_next

    _, unified_tail = jax.lax.scan(_step, carry_init, correlated_dW)
    unified_paths = jnp.concatenate([carry_init[None, :, :], unified_tail], axis=0)

    # Convert GBM back to price space; others remain in level space
    final_paths = jnp.where(is_gbm, jnp.exp(unified_paths), unified_paths)

    if visualize:
        for i in range(n_dim):
            plt.figure(figsize=(10, 6), dpi=150)
            plt.plot(final_paths[:, :, i])
            plt.title(f'Hybrid Process: {process_specs[i]["type"].upper()} (Dim {i})')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.show()

    return final_paths

def feller_square_root_process(
    initial_vector=0.05,
    long_term_vector=0.04,
    diffusion_vector=0.1,
    mean_reversion_coef=1.0,
    correlation_matrix=None,
    T=1.0,
    granularity=252,
    n_paths=100,
    visualize=False,
    method='antithetic_variates'
):
    """
    Simulates correlated multidimensional Feller square-root (CIR) processes.

    The dynamics follow the Cox-Ingersoll-Ross (CIR) model:
        dv_i(t) = κ_i (θ_i - v_i(t)) dt + σ_i √(v_i(t)) dW_i(t),
    where W_i are correlated Brownian motions with correlation matrix `correlation_matrix`.

    This process is widely used to model interest rate volatility, variance in Heston model,
    or any strictly positive mean-reverting quantity.

    Parameters
    ----------
    initial_vector : array-like, shape (n_dim,) or scalar
        Initial values v(0) ≥ 0. If scalar, broadcast to all dimensions.
    long_term_vector : array-like, shape (n_dim,) or scalar
        Long-term mean reversion levels θ_i > 0.
    diffusion_vector : array-like, shape (n_dim,) or scalar
        Volatility coefficients σ_i > 0.
    mean_reversion_coef : array-like, shape (n_dim,) or scalar
        Mean reversion speeds κ_i > 0.
    correlation_matrix : array-like, shape (n_dim, n_dim), optional
        Positive semi-definite correlation matrix for underlying Brownian motions.
        Defaults to identity if None.
    T : float
        Time horizon in years.
    granularity : int
        Time steps per year (e.g., 252). Total steps: N = int(granularity * T).
    n_paths : int
        Number of Monte Carlo paths.
    visualize : bool
        If True, plots all paths for each dimension.
    method : str
        Variance reduction method for noise generation.

    Returns
    -------
    paths : jnp.ndarray, shape (N + 1, n_paths, n_dim)
        Simulated CIR paths. All values are non-negative due to full truncation.

    Notes
    -----
    - Uses **Euler-Maruyama with full truncation**: √(max(v_t, 0)) to ensure stability.
    - **Feller condition**: 2κθ ≥ σ² guarantees strict positivity (no hits to zero).
      If violated, a warning is issued and paths may touch zero (but remain ≥ 0).
    - Time step: dt = T / N.
    - Fully compatible with JAX transformations (`jax.grad`, `jax.vmap`, etc.).
    - Reuses `se.correlate_noise` for consistent cross-asset correlation.

    Examples
    --------
    >>> paths = feller_square_root_process(
    ...     initial_vector=[0.04, 0.06],
    ...     long_term_vector=[0.05, 0.05],
    ...     diffusion_vector=[0.1, 0.15],
    ...     mean_reversion_coef=[1.2, 0.8],
    ...     correlation_matrix=[[1.0, 0.3], [0.3, 1.0]],
    ...     T=1.0,
    ...     granularity=252,
    ...     n_paths=1000
    ... )
    >>> print(paths.shape)  # (253, 1000, 2)
    """
    # Normalize inputs
    initial_vector = jnp.atleast_1d(initial_vector)
    long_term_vector = jnp.atleast_1d(long_term_vector)
    diffusion_vector = jnp.atleast_1d(diffusion_vector)
    mean_reversion_coef = jnp.atleast_1d(mean_reversion_coef)

    shapes = [initial_vector.shape, long_term_vector.shape,
              diffusion_vector.shape, mean_reversion_coef.shape]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError("All parameter vectors must have the same shape.")
    n_dim = initial_vector.shape[0]

    if jnp.any(initial_vector < 0):
        raise ValueError("Initial values must be non-negative.")

    # 🔔 Feller condition check (2κθ ≥ σ²)
    feller_lhs = 2.0 * mean_reversion_coef * long_term_vector
    feller_rhs = diffusion_vector ** 2
    feller_violated = feller_lhs < feller_rhs

    if jnp.any(feller_violated):
        violated_dims = jnp.where(feller_violated)[0].tolist()
        warnings.warn(
            f"Feller condition violated in dimensions {violated_dims}: "
            f"2κθ < σ². Paths may hit zero. "
            f"Check parameters: κ={mean_reversion_coef[violated_dims]}, "
            f"θ={long_term_vector[violated_dims]}, "
            f"σ={diffusion_vector[violated_dims]}",
            UserWarning
        )

    # Correlation matrix
    if correlation_matrix is None:
        correlation_matrix = jnp.eye(n_dim)
    else:
        correlation_matrix = jnp.asarray(correlation_matrix, dtype=jnp.float32)
        if correlation_matrix.shape != (n_dim, n_dim):
            raise ValueError(f"correlation_matrix must be ({n_dim}, {n_dim})")

    # Time grid
    N = int(granularity * T)
    if N <= 0:
        raise ValueError("granularity * T must be positive")
    dt = T / N
    sqrt_dt = jnp.sqrt(dt)

    # Generate and correlate noise
    dW = se.generate_uncorrelated_white_noise(N, n_paths, n_dim, method=method)
    correlated_dW = se.correlate_noise(dW, correlation_matrix)

    # Initialize state
    carry_init = jnp.broadcast_to(initial_vector, (n_paths, n_dim))

    def _step(v_prev, dw_step):
        sqrt_v = jnp.sqrt(jnp.maximum(v_prev, 0.0))
        drift = mean_reversion_coef * (long_term_vector - v_prev) * dt
        diffusion = diffusion_vector * sqrt_v * sqrt_dt * dw_step
        v_next = v_prev + drift + diffusion
        return jnp.maximum(v_next, 0.0), v_next  # enforce non-negativity

    _, vs = jax.lax.scan(_step, carry_init, correlated_dW)
    initial_expanded = carry_init[None, :, :]
    paths = jnp.concatenate([initial_expanded, vs], axis=0)

    if visualize:
        for dim in range(n_dim):
            plt.figure(figsize=(10, 6), dpi=150)
            plt.plot(paths[:, :, dim])
            plt.title(f'Feller Square-Root Process (Dim {dim})')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.show()

    return paths