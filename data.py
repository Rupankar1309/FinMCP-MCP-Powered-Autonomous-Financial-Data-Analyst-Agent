"""
Market data access layer. Wraps yfinance and caches results to disk so that
(a) repeated eval runs don't hammer Yahoo Finance and (b) the agent stays
fast during a live demo.

NOTE: this module needs outbound internet access to query.finance.yahoo.com.
It cannot be exercised inside a network-sandboxed environment — that's
expected. Test it on the machine you actually run the agent on.
"""

from __future__ import annotations
import time
import json
from pathlib import Path
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_SECONDS = 60 * 60 * 4  # 4 hours — fine for daily-bar analysis


def _cache_path(key: str) -> Path:
    safe = key.replace("/", "_")
    return CACHE_DIR / f"{safe}.parquet"


def get_price_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV history for a ticker, with disk caching.

    period: yfinance period string, e.g. '1mo', '6mo', '1y', '2y', '5y'
    interval: '1d', '1wk', '1mo' (intraday intervals need a much shorter period)
    """
    ticker = ticker.upper().strip()
    key = f"{ticker}_{period}_{interval}"
    path = _cache_path(key)

    if path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df

    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty:
        raise ValueError(
            f"No price data returned for '{ticker}'. Check the ticker symbol is valid "
            f"(e.g. 'AAPL', 'MSFT', 'RELIANCE.NS' for NSE-listed stocks)."
        )
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.to_parquet(path)
    return df


def get_company_info(ticker: str) -> dict:
    """Basic company metadata — sector, market cap, etc. Not cached (cheap call, changes rarely matters)."""
    ticker = ticker.upper().strip()
    info = yf.Ticker(ticker).get_info()
    keys = [
        "shortName", "sector", "industry", "marketCap", "currency",
        "trailingPE", "forwardPE", "dividendYield", "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow", "currentPrice", "country",
    ]
    return {k: info.get(k) for k in keys if k in info}


def get_multiple_price_histories(tickers: list[str], period: str = "6mo", interval: str = "1d") -> dict[str, pd.DataFrame]:
    """Fetch several tickers' histories, skipping (and reporting) any that fail."""
    out, errors = {}, {}
    for t in tickers:
        try:
            out[t.upper().strip()] = get_price_history(t, period, interval)
        except Exception as e:
            errors[t] = str(e)
    if errors:
        out["_errors"] = errors  # caller can strip this before passing to analytics
    return out
