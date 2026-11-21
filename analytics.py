"""
Pure computation functions for the finance analyst agent.

Design note: every function here takes a pandas DataFrame (or a dict of them)
and returns plain Python data structures (dict/list of dicts). None of them
touch the network. This matters for two reasons:

1. Testability — we can unit test all the math with synthetic data, with no
   dependency on Yahoo Finance being reachable.
2. Ground truth — the eval harness (eval/run_eval.py) calls these SAME
   functions to compute the "correct" answer for a benchmark question, then
   checks whether the LLM agent's natural-language answer matches. Using one
   source of truth for both the tool and the grader avoids silently grading
   against stale, hand-typed numbers.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def compute_returns(df: pd.DataFrame) -> dict:
    """Daily and cumulative simple returns from a price DataFrame with a 'Close' column."""
    close = df["Close"].dropna()
    daily = close.pct_change().dropna()
    cumulative = (1 + daily).prod() - 1
    return {
        "start_date": str(close.index[0].date()),
        "end_date": str(close.index[-1].date()),
        "start_price": round(float(close.iloc[0]), 2),
        "end_price": round(float(close.iloc[-1]), 2),
        "cumulative_return_pct": round(float(cumulative) * 100, 2),
        "avg_daily_return_pct": round(float(daily.mean()) * 100, 4),
        "best_day": {
            "date": str(daily.idxmax().date()),
            "return_pct": round(float(daily.max()) * 100, 2),
        },
        "worst_day": {
            "date": str(daily.idxmin().date()),
            "return_pct": round(float(daily.min()) * 100, 2),
        },
    }


def compute_moving_averages(df: pd.DataFrame, windows: list[int] | None = None) -> dict:
    """Simple moving averages for the given window sizes (in trading days)."""
    windows = windows or [20, 50]
    close = df["Close"].dropna()
    result = {"as_of": str(close.index[-1].date()), "last_close": round(float(close.iloc[-1]), 2)}
    for w in windows:
        if len(close) >= w:
            sma = close.rolling(window=w).mean().iloc[-1]
            result[f"sma_{w}"] = round(float(sma), 2)
        else:
            result[f"sma_{w}"] = None  # not enough history
    return result


def compute_volatility(df: pd.DataFrame, window: int = 21) -> dict:
    """Rolling volatility of daily returns, annualized (assumes 252 trading days/year)."""
    close = df["Close"].dropna()
    daily = close.pct_change().dropna()
    rolling_std = daily.rolling(window=window).std().iloc[-1]
    annualized = rolling_std * np.sqrt(252)
    return {
        "window_days": window,
        "rolling_daily_volatility_pct": round(float(rolling_std) * 100, 3),
        "annualized_volatility_pct": round(float(annualized) * 100, 2),
        "full_period_annualized_volatility_pct": round(float(daily.std() * np.sqrt(252)) * 100, 2),
    }


def detect_price_anomalies(df: pd.DataFrame, z_threshold: float = 2.5) -> dict:
    """Flag days whose return is an outlier relative to the period's own distribution (z-score)."""
    close = df["Close"].dropna()
    daily = close.pct_change().dropna()
    mean, std = daily.mean(), daily.std()
    z = (daily - mean) / std
    flagged = z[abs(z) >= z_threshold]
    anomalies = [
        {
            "date": str(idx.date()),
            "return_pct": round(float(daily.loc[idx]) * 100, 2),
            "z_score": round(float(z.loc[idx]), 2),
        }
        for idx in flagged.index
    ]
    return {
        "z_threshold": z_threshold,
        "num_anomalies": len(anomalies),
        "anomalies": sorted(anomalies, key=lambda a: abs(a["z_score"]), reverse=True),
    }


def compare_stocks(price_dfs: dict[str, pd.DataFrame]) -> dict:
    """Compare cumulative return and volatility across multiple tickers over the same period."""
    rows = []
    for ticker, df in price_dfs.items():
        r = compute_returns(df)
        v = compute_volatility(df)
        rows.append({
            "ticker": ticker,
            "cumulative_return_pct": r["cumulative_return_pct"],
            "annualized_volatility_pct": v["full_period_annualized_volatility_pct"],
        })
    rows.sort(key=lambda x: x["cumulative_return_pct"], reverse=True)
    return {"comparison": rows, "best_performer": rows[0]["ticker"] if rows else None}


def correlation_matrix(price_dfs: dict[str, pd.DataFrame]) -> dict:
    """Pearson correlation of daily returns across tickers."""
    returns = {}
    for ticker, df in price_dfs.items():
        returns[ticker] = df["Close"].pct_change().dropna()
    combined = pd.DataFrame(returns).dropna()
    corr = combined.corr().round(3)
    return {"tickers": list(corr.columns), "matrix": corr.to_dict()}
