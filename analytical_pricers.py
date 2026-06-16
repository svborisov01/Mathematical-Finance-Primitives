import numpy as np


def heston_cf(u, T, r, q, kappa, theta, sigma, rho, v0, s0):
    """
    Risk-neutral characteristic function of log(S_T) under Heston.
    u can be scalar or numpy array.
    """
    u = np.asarray(u, dtype=np.complex128)
    x0 = np.log(s0)
    i = 1j

    d = np.sqrt((rho * sigma * i * u - kappa)**2 + sigma**2 * (i * u + u**2))
    g = (kappa - rho * sigma * i * u - d) / (kappa - rho * sigma * i * u + d)

    exp_neg_dT = np.exp(-d * T)

    C = (
        (r - q) * i * u * T
        + (kappa * theta / sigma**2)
        * ((kappa - rho * sigma * i * u - d) * T
           - 2.0 * np.log((1.0 - g * exp_neg_dT) / (1.0 - g)))
    )

    D = ((kappa - rho * sigma * i * u - d) / sigma**2) * (
        (1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT)
    )

    return np.exp(C + D * v0 + i * u * x0)


def _chi_psi(a, b, c, d, k):
    """
    COS auxiliary coefficients on interval [c, d] within truncation [a, b].
    k is an array of indices.
    """
    k = np.asarray(k, dtype=np.float64)
    omega = k * np.pi / (b - a)

    exp_c = np.exp(c)
    exp_d = np.exp(d)

    cos_term_d = np.cos(omega * (d - a))
    cos_term_c = np.cos(omega * (c - a))
    sin_term_d = np.sin(omega * (d - a))
    sin_term_c = np.sin(omega * (c - a))

    denom = 1.0 + omega**2

    chi = (
        (cos_term_d * exp_d - cos_term_c * exp_c)
        + omega * (sin_term_d * exp_d - sin_term_c * exp_c)
    ) / denom

    psi = np.zeros_like(k)
    psi[0] = d - c
    psi[1:] = (sin_term_d[1:] - sin_term_c[1:]) * (b - a) / (k[1:] * np.pi)

    return chi, psi


def cos_put_coefficients(a, b, K, N):
    """
    COS payoff coefficients for put payoff (K - exp(x))^+.
    Assumes x = log(S_T), exercise region x < log(K).
    """
    k = np.arange(N, dtype=np.float64)
    c = a
    d = np.log(K)

    chi, psi = _chi_psi(a, b, c, d, k)
    Vk = 2.0 / (b - a) * (K * psi - chi)
    return Vk


def heston_cos_price_put(
    s0, K, T, r, q,
    kappa, theta, sigma, rho, v0,
    N=256, L=12.0
):
    """
    European put via COS under Heston.
    """
    # Simple truncation interval around log-forward
    c1 = np.log(s0) + (r - q) * T
    a = c1 - L * np.sqrt(T + np.sqrt(v0 * T + theta * T))
    b = c1 + L * np.sqrt(T + np.sqrt(v0 * T + theta * T))

    k = np.arange(N, dtype=np.float64)
    u = k * np.pi / (b - a)

    cf_vals = heston_cf(u, T, r, q, kappa, theta, sigma, rho, v0, s0)
    Vk = cos_put_coefficients(a, b, K, N)

    weights = np.ones(N)
    weights[0] = 0.5

    price = np.exp(-r * T) * np.sum(
        weights * np.real(cf_vals * np.exp(-1j * u * a)) * Vk
    )
    return np.real(price)


def heston_cos_price_call(
    s0, K, T, r, q,
    kappa, theta, sigma, rho, v0,
    N=256, L=12.0
):
    """
    European call via put-call parity for stability.
    """
    put = heston_cos_price_put(
        s0, K, T, r, q, kappa, theta, sigma, rho, v0, N=N, L=L
    )
    forward_discounted = s0 * np.exp(-q * T) - K * np.exp(-r * T)
    return put + forward_discounted

def heston_price(
    s0, K, T, r, q,
    kappa, theta, sigma, rho, v0, call_put = "Call",
    N=256, L=12.0
):
    if call_put == "Call":
        heston_cos_price_call(s0, K, T, r, q, kappa, theta, sigma, rho, v0, N, L)
    else:
        heston_cos_price_put(s0, K, T, r, q, kappa, theta, sigma, rho, v0, N, L)

def heston_cos_price_surface(
    s0, strikes, maturities, r, q,
    kappa, theta, sigma, rho, v0,
    cp="call", N=256, L=12.0
):
    """
    Price a matrix of strikes x maturities.
    Returns array shape (len(maturities), len(strikes)).
    """
    strikes = np.asarray(strikes, dtype=np.float64)
    maturities = np.asarray(maturities, dtype=np.float64)

    out = np.zeros((len(maturities), len(strikes)))

    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            if cp == "Call":
                out[i, j] = heston_cos_price_call(
                    s0, K, T, r, q, kappa, theta, sigma, rho, v0, N=N, L=L
                )
            else:
                out[i, j] = heston_cos_price_put(
                    s0, K, T, r, q, kappa, theta, sigma, rho, v0, N=N, L=L
                )
    return out
