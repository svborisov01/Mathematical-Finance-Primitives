import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

def heston_char_func(u, tau, S0, r, q, v0, kappa, theta, sigma, rho, j=2):
    """
    Heston characteristic function - Little Heston Trap formulation (Albrecher et al. 2007)
    
    Parameters:
    -----------
    j : int (1 or 2)
        j=1 for P1, j=2 for P2
    """
    i = 1j
    
    # Parameters that differ for P1 and P2
    u1 = 0.5
    u2 = -0.5
    uj = u1 if j == 1 else u2
    
    # Coefficients
    a = kappa * theta
    b = kappa  # Zero volatility risk premium
    
    # Calculate d and g carefully
    delta = b - rho * sigma * u * i
    discriminant = delta**2 - sigma**2 * (2 * uj * u * i - u**2)
    
    # Use principal square root
    d = np.sqrt(discriminant + 0j)  # Ensure complex
    
    # Avoid division by zero
    if np.abs(d) < 1e-10:
        d = 1e-10 + 0j
    
    # Calculate g
    g = (delta - d) / (delta + d)
    
    # Calculate D term
    exp_term = np.exp(-d * tau)
    D = (delta - d) / sigma**2 * (1 - exp_term) / (1 - g * exp_term)
    
    # Calculate C term - use log1p for better numerical stability when g*exp_term is close to 1
    log_arg = (1 - g * exp_term) / (1 - g)
    
    # Handle numerical issues with log
    if np.abs(log_arg) < 1e-10:
        log_val = np.log(1e-10)
    else:
        log_val = np.log(log_arg + 0j)
    
    C = (r - q) * u * i * tau + (a / sigma**2) * (
        (delta - d) * tau - 2 * log_val
    )
    
    # Final characteristic function
    cf = np.exp(C + D * v0 + i * u * np.log(S0))
    
    # Check for numerical issues
    if np.isnan(cf) or np.isinf(cf):
        return 0.0 + 0j
    
    return cf


def heston_call_price(S0, K, T, r, q, v0, kappa, theta, sigma, rho, 
                      integration_limit=100.0):
    """
    Heston call option price using Gil-Pelaez inversion
    """
    tau = T
    
    def integrand_P2(u):
        """Integrand for P2"""
        if u < 1e-10:
            return 0.0
        cf = heston_char_func(u, tau, S0, r, q, v0, kappa, theta, sigma, rho, j=2)
        result = np.real(np.exp(-1j * u * np.log(K)) * cf / (1j * u))
        return result if not (np.isnan(result) or np.isinf(result)) else 0.0
    
    def integrand_P1(u):
        """Integrand for P1"""
        if u < 1e-10:
            return 0.0
        cf = heston_char_func(u, tau, S0, r, q, v0, kappa, theta, sigma, rho, j=1)
        result = np.real(np.exp(-1j * u * np.log(K)) * cf / (1j * u))
        return result if not (np.isnan(result) or np.isinf(result)) else 0.0
    
    # Integrate with error handling
    try:
        P2_integral, _ = quad(integrand_P2, 1e-8, integration_limit, 
                              limit=200, epsabs=1e-6, epsrel=1e-6)
        P1_integral, _ = quad(integrand_P1, 1e-8, integration_limit, 
                              limit=200, epsabs=1e-6, epsrel=1e-6)
    except:
        print("Integration failed - returning NaN")
        return np.nan
    
    P2 = 0.5 + P2_integral / np.pi
    P1 = 0.5 + P1_integral / np.pi
    
    # Ensure probabilities are in valid range
    P1 = np.clip(P1, 0, 1)
    P2 = np.clip(P2, 0, 1)
    
    # Call price formula
    call_price = np.exp(-q * tau) * S0 * P1 - np.exp(-r * tau) * K * P2
    
    return call_price


# Test the implementation
if __name__ == "__main__":
    # Parameters - start with simple case
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.0
    v0 = 0.04
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    
    print("Testing Heston pricer...")
    print(f"Parameters: S0={S0}, K={K}, T={T}, r={r}, q={q}")
    print(f"v0={v0}, kappa={kappa}, theta={theta}, sigma={sigma}, rho={rho}")
    print()
    
    # Test with sigma -> 0 (should converge to BS)
    print("Test 1: sigma=0.001 (should be close to BS with vol=0.2)")
    heston_price = heston_call_price(S0, K, T, r, q, v0, kappa, theta, 
                                     sigma=0.001, rho=0.0)
    
    # Black-Scholes price
    d1 = (np.log(S0/K) + (r - q + 0.5*0.2**2)*T) / (0.2*np.sqrt(T))
    d2 = d1 - 0.2*np.sqrt(T)
    bs_price = np.exp(-q*T)*S0*norm.cdf(d1) - np.exp(-r*T)*K*norm.cdf(d2)
    
    print(f"Heston Price: {heston_price:.6f}")
    print(f"BS Price: {bs_price:.6f}")
    print(f"Difference: {abs(heston_price - bs_price):.6f}")
    print()
    
    # Test with full parameters
    print("Test 2: Full Heston parameters (sigma=0.3)")
    heston_price_full = heston_call_price(S0, K, T, r, q, v0, kappa, theta, sigma, rho)
    print(f"Heston Price: {heston_price_full:.6f}")
