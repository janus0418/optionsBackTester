import pandas as pd

from backtester.metrics import PerformanceMetrics, PnLAttributionEngine


def test_performance_metrics_basic():
    returns = pd.Series([0.01, -0.005, 0.02])
    metrics = PerformanceMetrics(returns)
    assert metrics.cumulative_pnl() > 0
    assert metrics.annualized_vol() > 0


def test_pnl_attribution_conservation():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    values = pd.Series([100, 101, 102], index=dates)
    spot = pd.Series([100, 101, 102], index=dates)
    rates = pd.Series([0.01, 0.01, 0.01], index=dates)
    greeks = {
        "delta": pd.Series([1.0, 1.0, 1.0], index=dates),
        "gamma": pd.Series(0.0, index=dates),
        "vega": pd.Series(0.0, index=dates),
        "theta": pd.Series(0.0, index=dates),
        "rho": pd.Series(0.0, index=dates),
    }
    engine = PnLAttributionEngine(spot, rates)
    result = engine.attribute(values, greeks)
    assert abs(result.total_pnl.sum() - values.diff().fillna(0).sum()) < 1e-8
