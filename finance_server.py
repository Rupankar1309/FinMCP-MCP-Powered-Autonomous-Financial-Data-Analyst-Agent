"""
FinMCP — an MCP server exposing finance data-analysis tools over stock data.

Run standalone (stdio transport, the default MCP client/agent talks to this):
    python -m server.finance_server

Every tool here does real computation over real (or cached) market data —
there is no path by which the agent can "answer" a numeric question without
actually calling one of these functions and getting a real number back.
That's the deliberate design choice: it removes numeric hallucination as a
failure mode, at the cost of the agent needing to plan tool calls correctly.
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt

from mcp.server.mcpserver import MCPServer

from server import data as market_data
from server import analytics

server = MCPServer(
    "finmcp",
    instructions=(
        "Tools for analyzing publicly traded stocks: price history, returns, "
        "moving averages, volatility, anomaly detection, cross-ticker comparison, "
        "correlation, and charting. Always call a tool to get real numbers before "
        "answering a quantitative question — never estimate or guess a figure."
    ),
)

CHARTS_DIR = Path(__file__).parent.parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)


@server.tool()
def get_stock_summary(ticker: str, period: str = "6mo") -> dict:
    """
    Get price history summary and return statistics for a single stock ticker.
    period: one of '1mo','3mo','6mo','1y','2y','5y'. Use short periods for
    recent-performance questions, longer ones for trend questions.
    """
    df = market_data.get_price_history(ticker, period=period)
    return {"ticker": ticker.upper(), "period": period, **analytics.compute_returns(df)}


@server.tool()
def get_company_info(ticker: str) -> dict:
    """Get basic company info for a ticker: sector, market cap, P/E, 52-week range, etc."""
    return {"ticker": ticker.upper(), **market_data.get_company_info(ticker)}


@server.tool()
def get_moving_averages(ticker: str, period: str = "6mo", windows: list[int] | None = None) -> dict:
    """
    Compute simple moving averages (SMA) of closing price for a ticker.
    windows: list of day counts, e.g. [20, 50, 200]. Defaults to [20, 50].
    """
    df = market_data.get_price_history(ticker, period=period)
    return {"ticker": ticker.upper(), **analytics.compute_moving_averages(df, windows)}


@server.tool()
def get_volatility(ticker: str, period: str = "6mo", window: int = 21) -> dict:
    """
    Compute rolling and full-period annualized volatility of daily returns for a ticker.
    window: rolling window size in trading days (21 ≈ 1 month).
    """
    df = market_data.get_price_history(ticker, period=period)
    return {"ticker": ticker.upper(), **analytics.compute_volatility(df, window)}


@server.tool()
def get_anomalies(ticker: str, period: str = "6mo", z_threshold: float = 2.5) -> dict:
    """
    Detect statistically anomalous daily price moves (z-score of daily return
    vs. the period's own mean/std). Use this for "any unusual activity /
    spikes / crashes" questions.
    """
    df = market_data.get_price_history(ticker, period=period)
    return {"ticker": ticker.upper(), **analytics.detect_price_anomalies(df, z_threshold)}


@server.tool()
def compare_tickers(tickers: list[str], period: str = "6mo") -> dict:
    """
    Compare cumulative return and annualized volatility across 2+ tickers
    over the same period. Use for "which performed better, X or Y" questions.
    """
    dfs = market_data.get_multiple_price_histories(tickers, period=period)
    errors = dfs.pop("_errors", {})
    result = analytics.compare_stocks(dfs) if dfs else {"comparison": []}
    if errors:
        result["skipped_tickers"] = errors
    return result


@server.tool()
def get_correlation(tickers: list[str], period: str = "6mo") -> dict:
    """
    Compute the correlation matrix of daily returns across 2+ tickers.
    Use for "how correlated are X and Y" / diversification questions.
    """
    dfs = market_data.get_multiple_price_histories(tickers, period=period)
    dfs.pop("_errors", None)
    if len(dfs) < 2:
        return {"error": "Need at least 2 valid tickers to compute correlation."}
    return analytics.correlation_matrix(dfs)


@server.tool()
def plot_price_chart(ticker: str, period: str = "6mo", show_sma: bool = True) -> dict:
    """
    Generate and save a price chart (with optional 20/50-day SMA overlay) for a ticker.
    Returns the file path where the chart PNG was saved.
    """
    df = market_data.get_price_history(ticker, period=period)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df.index, df["Close"], label="Close", linewidth=1.5)
    if show_sma:
        for w in (20, 50):
            if len(df) >= w:
                ax.plot(df.index, df["Close"].rolling(w).mean(), label=f"SMA{w}", linewidth=1)
    ax.set_title(f"{ticker.upper()} — {period}")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    out_path = CHARTS_DIR / f"{ticker.upper()}_{period}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return {"ticker": ticker.upper(), "period": period, "chart_path": str(out_path)}


if __name__ == "__main__":
    server.run()  # stdio transport by default — what the agent bridge expects
