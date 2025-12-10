# ticker_utils.py
import requests
import yfinance as yf
import pandas as pd
from typing import Tuple

def name_to_ticker_safe(query: str) -> str:
    if not query or str(query).strip() == "":
        return ""
    q = str(query).strip()

    # 1) Try direct short download (user supplied ticker)
    try:
        test = yf.download(q, period="5d", progress=False)
        if test is not None and not test.empty:
            return q.upper()
    except Exception:
        pass

    # 2) Manual mapping for common names & indices
    manual_map = {
        "nifty": "^NSEI",
        "nifty 50": "^NSEI",
        "sensex": "^BSESN",
        "bank nifty": "^NSEBANK",
        "nasdaq": "^IXIC",
        "dow": "^DJI",
        "s&p": "^GSPC",
        "s&p 500": "^GSPC",
        "tesla": "TSLA",
        "apple": "AAPL",
        "microsoft": "MSFT",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "meta": "META",
        "facebook": "META",
        "amazon": "AMZN",
        "reliance": "RELIANCE.NS",
        "tcs": "TCS.NS",
        "infosys": "INFY.NS"
    }
    key = q.lower()
    if key in manual_map:
        return manual_map[key]

    # 3) Yahoo search API
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/search"
        r = requests.get(url, params={"q": q, "quotesCount": 1}, timeout=6)
        j = r.json()
        if "quotes" in j and len(j["quotes"]) > 0:
            symbol = j["quotes"][0].get("symbol")
            if symbol:
                return symbol
    except Exception:
        pass

    # 4) fallback to uppercase user input
    return q.upper()

def normalize_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    pref = f"Close_{ticker}" in df.columns
    if pref:
        return df
    rename_map = {}
    for k in ["Open","High","Low","Close","Volume"]:
        if k in df.columns:
            rename_map[k] = f"{k}_{ticker}"
    if rename_map:
        return df.rename(columns=rename_map)
    return df

def get_col(df: pd.DataFrame, base_name: str, ticker: str) -> str:
    candidates = [
        base_name,
        f"{base_name}_{ticker}",
        f"{base_name}_{ticker.upper()}",
        f"{base_name}_adj",
        f"{base_name}_Adj"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for col in df.columns:
        if base_name.lower() in col.lower():
            return col
    raise KeyError(f"Column `{base_name}` not found. Available: {list(df.columns)}")
