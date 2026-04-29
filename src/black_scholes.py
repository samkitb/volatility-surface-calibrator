"""
So this pretty much uses data on options price to calculate the implied volatility.
IV is a prediction of the future volatility of the underlying asset.
People use IV in options pricing to predict the future price of the underlying asset.
Higher the IV, the higher an option is priced which means people pay a higher premium for the option.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def implied_volatility(market_price, S, K, T, r=0.045, option_type="call"):
    intrinsic = max(0, S - K) if option_type == "call" else max(0, K - S)
    if market_price <= intrinsic:
        return None
    try:
        iv = brentq(
            lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type) - market_price,
            a=0.01,
            b=5.0,
            xtol=1e-6,
            maxiter=200
        )
        return iv
    except (ValueError, RuntimeError):
        return None

def add_implied_vols(df, r=0.045):
    ivs = []
    for _, row in df.iterrows():
        iv = implied_volatility(
            market_price=row["mid_price"],
            S=row["spot"],
            K=row["strike"],
            T=row["time_to_expiry"],
            r=r,
            option_type=row["option_type"]
        )
        ivs.append(iv)
    df = df.copy()
    df["implied_vol"] = ivs
    df = df.dropna(subset=["implied_vol"])
    df = df[(df["implied_vol"] > 0.01) & (df["implied_vol"] < 3.0)]
    print(f"Successfully computed IV for {len(df)} contracts")
    return df
