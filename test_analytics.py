"""
Unit tests for server/analytics.py using synthetic price data.
Run with: pytest tests/ -v

These deliberately don't touch the network or yfinance — that's the whole
point of keeping analytics.py pure. server/data.py (the yfinance wrapper)
is integration-tested by hand on a machine with real internet access
(see README) rather than mocked here, since mocking yfinance's response
shape thoroughly would test the mock more than the code.
"""
import numpy as np
import pandas as pd
import pytest

from server.analytics import (
    compute_returns,
    compute_moving_averages,
    compute_volatility,
    detect_price_anomalies,
    compare_stocks,
    correlation_matrix,
)


def make_price_df(start=100.0, drift=0.0005, vol=0.015, n=120, seed=1, spike_at=None, spike_ret=0.15):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n)
    rets = rng.normal(drift, vol, n)
    if spike_at is not None:
        rets[spike_at] = spike_ret
    prices = start * (1 + pd.Series(rets)).cumprod()
    return pd.DataFrame({"Close": prices.values}, index=dates)


class TestComputeReturns:
    def test_known_two_point_series(self):
        # Deterministic case: 100 -> 110 is exactly +10% cumulative
        dates = pd.bdate_range("2025-01-01", periods=2)
        df = pd.DataFrame({"Close": [100.0, 110.0]}, index=dates)
        r = compute_returns(df)
        assert r["cumulative_return_pct"] == pytest.approx(10.0, abs=0.01)
        assert r["start_price"] == 100.0
        assert r["end_price"] == 110.0

    def test_best_and_worst_day_are_actually_extremes(self):
        df = make_price_df(spike_at=40, spike_ret=0.20)
        r = compute_returns(df)
        # the injected +20% day should be identified as the best day
        assert r["best_day"]["return_pct"] > 15.0


class TestMovingAverages:
    def test_flat_series_sma_equals_price(self):
        dates = pd.bdate_range("2025-01-01", periods=60)
        df = pd.DataFrame({"Close": [50.0] * 60}, index=dates)
        r = compute_moving_averages(df, windows=[20])
        assert r["sma_20"] == pytest.approx(50.0)

    def test_insufficient_history_returns_none(self):
        df = make_price_df(n=10)
        r = compute_moving_averages(df, windows=[200])
        assert r["sma_200"] is None


class TestVolatility:
    def test_higher_input_vol_gives_higher_output_vol(self):
        calm = make_price_df(vol=0.005, seed=2)
        wild = make_price_df(vol=0.05, seed=2)
        v_calm = compute_volatility(calm)
        v_wild = compute_volatility(wild)
        assert v_wild["full_period_annualized_volatility_pct"] > v_calm["full_period_annualized_volatility_pct"]

    def test_zero_vol_series_gives_zero_volatility(self):
        dates = pd.bdate_range("2025-01-01", periods=50)
        df = pd.DataFrame({"Close": [100.0] * 50}, index=dates)
        v = compute_volatility(df)
        assert v["full_period_annualized_volatility_pct"] == pytest.approx(0.0, abs=1e-6)


class TestAnomalies:
    def test_injected_spike_is_detected(self):
        df = make_price_df(vol=0.01, spike_at=60, spike_ret=0.25, seed=3)
        a = detect_price_anomalies(df, z_threshold=2.5)
        assert a["num_anomalies"] >= 1
        assert any(x["return_pct"] > 15.0 for x in a["anomalies"])

    def test_calm_series_flags_nothing_at_high_threshold(self):
        dates = pd.bdate_range("2025-01-01", periods=60)
        rng = np.random.default_rng(4)
        prices = 100 * (1 + pd.Series(rng.normal(0, 0.001, 60))).cumprod()
        df = pd.DataFrame({"Close": prices.values}, index=dates)
        a = detect_price_anomalies(df, z_threshold=5.0)
        assert a["num_anomalies"] == 0


class TestCompareAndCorrelate:
    def test_compare_ranks_best_performer_first(self):
        low = make_price_df(drift=0.0001, seed=5)
        high = make_price_df(drift=0.003, seed=6)
        result = compare_stocks({"LOW": low, "HIGH": high})
        assert result["comparison"][0]["ticker"] == result["best_performer"]
        assert result["comparison"][0]["cumulative_return_pct"] >= result["comparison"][1]["cumulative_return_pct"]

    def test_correlation_self_is_one(self):
        a = make_price_df(seed=7)
        b = make_price_df(seed=8)
        corr = correlation_matrix({"A": a, "B": b})
        assert corr["matrix"]["A"]["A"] == pytest.approx(1.0)
        assert corr["matrix"]["B"]["B"] == pytest.approx(1.0)

    def test_identical_series_are_perfectly_correlated(self):
        a = make_price_df(seed=9)
        corr = correlation_matrix({"A": a, "A_COPY": a})
        assert corr["matrix"]["A"]["A_COPY"] == pytest.approx(1.0, abs=0.001)
