"""
LAYER 1
This layer is responsible for fetching the options chain from Yahoo Finance.
It grabs every options contract available for a ticker and then gives it to black_sholes.py to get converted into IV
An options chain is the hundreds of contracts available for a given ticker. Each row in that table
has a strike price, expiry date, and price.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("fetch.py is running...")

from black_scholes import add_implied_vols
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
def fetch_options_chain(ticker: str) -> pd.DataFrame:
    """
    Pull every option contract available for a ticker from Yahoo Finance.
    Returns a DataFrame with strikes, expiries, prices, and moneyness.
    """
    stock = yf.Ticker(ticker)
    
    # Get current stock price, this is needed to calculate the moneyness of the options. (how far it is from strike price   )
    spot_price = stock.history(period="1d")["Close"].iloc[-1]
    expiry_dates = stock.options  # all available expiry dates

    print(f"{ticker} spot price: ${spot_price:.2f}")
    print(f"Available expiries: {expiry_dates}")

    all_options = []
    # Loop through each expiry date and get the options chain for that expiry date
    for expiry in expiry_dates:
        chain = stock.option_chain(expiry)
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d") # find the number of days to expiry
        days_to_expiry = (expiry_dt - datetime.now()).days

        # Skip very short dated (noisy) and very long dated (illiquid)
        if days_to_expiry < 5 or days_to_expiry > 365:
            continue

        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            df = df.copy()
            df["option_type"]     = opt_type
            df["expiry"]          = expiry
            df["days_to_expiry"]  = days_to_expiry
            df["time_to_expiry"]  = days_to_expiry / 365.0
            df["spot"]            = spot_price
            df["mid_price"]       = (df["bid"] + df["ask"]) / 2 # average of bid and ask price
            df["moneyness"]       = df["strike"] / spot_price # how far the spot price is from strike price

            # Only keep liquid contracts
            df = df[
                (df["bid"] > 0) &
                (df["volume"] > 0) &
                (df["moneyness"] > 0.7) & # keep options that are only atleast 70% of the spot price
                (df["moneyness"] < 1.3) # keep options that are only atmost 130% of the spot price
            ]

            all_options.append(df[[
                "strike", "moneyness", "expiry", "days_to_expiry",
                "time_to_expiry", "option_type", "mid_price", "spot",
                "volume", "openInterest"
            ]])

    if not all_options:
        raise ValueError(f"No liquid options found for {ticker}")

    result = pd.concat(all_options, ignore_index=True)
    print(f"Fetched {len(result)} liquid option contracts")
    return result, spot_price


def get_vol_surface_data(ticker: str):
    """
    fetch_options_chain is called to get raw prices and then passes DF to add_implied_vols to get IV
    """
    os.makedirs("data", exist_ok=True)

    df, spot = fetch_options_chain(ticker)
    df = add_implied_vols(df)

    save_path = f"data/{ticker.lower()}_vol_surface.csv"
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} data points to {save_path}")

    return df, spot


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    df, spot = get_vol_surface_data(ticker)
    print(f"\nSample data:")
    print(df[["moneyness", "time_to_expiry", "implied_vol"]].head(10))