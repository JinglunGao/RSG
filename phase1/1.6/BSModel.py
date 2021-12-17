import numpy as np
import scipy.stats as si

def black_scholes_call_option(S, K, T, q, r, sigma):
    '''
    S: Stock price
    K: Strike price
    T: Maturity
    q: Dividend rate
    r: Risk free rate
    sigma: Volatility
    '''
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call = (S * np.exp(-q * T) * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2))
    
    return call

def black_scholes_put_option(S, K, T, q, r, sigma):
    '''
    S: Stock price
    K: Strike price
    T: Maturity
    q: Dividend rate
    r: Risk free rate
    sigma: Volatility
    '''
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * np.exp(-q * T) * si.norm.cdf(-d1)
    
    return put

