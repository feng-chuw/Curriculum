import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import numpy as np

import Components.strategy as strategy_module
from Components.strategy import BaseStrategy, compute_stats


# =========================
# 1. mock run_backtest（关键）
# =========================
class DummyBT:
    def __init__(self):
        self.initial_cash = 10000
        self.multiplier = 1.0


def mock_run_backtest(*args, **kwargs):
    idx = pd.date_range("2024-01-01", periods=5)

    result = pd.DataFrame({
        "equity": [10000, 10100, 10200, 10150, 10300],
        "qty": [0, 1, 1, 0, -1],
        "close": [100, 101, 102, 101, 103],
        "ret": [0, 0.01, 0.0099, -0.0049, 0.0147],
        "dd": [0, 0, 0, -0.005, 0],
    }, index=idx)

    return DummyBT(), result


# =========================
# 2. Dummy Strategy
# =========================
class DummyStrategy(BaseStrategy):
    def build_signals(self, df):
        df["position"] = [0, 1, 1, 0, -1]
        return df


# =========================
# 3. compute_stats
# =========================
def test_compute_stats_basic():
    df = pd.DataFrame({
        "net_value": [1.0, 1.1, 1.05],
        "ret": [0.0, 0.1, -0.045],
        "dd": [0.0, 0.0, -0.045],
        "trade_count": [0, 1, 0]
    })

    stats = compute_stats(df, 10000, "1d")

    assert "total_return" in stats
    assert stats["max_drawdown"] <= 0


def test_compute_stats_zero_vol():
    df = pd.DataFrame({
        "net_value": [1, 1, 1],
        "ret": [0, 0, 0],
        "dd": [0, 0, 0],
        "trade_count": [0, 0, 0]
    })

    stats = compute_stats(df, 10000, "1d")

    assert np.isnan(stats["sharpe"])


# =========================
# 4. run 主流程（重点）
# =========================
def test_strategy_run(monkeypatch):
    monkeypatch.setattr(strategy_module, "run_backtest", mock_run_backtest)

    idx = pd.date_range("2024-01-01", periods=5)

    ohlc = pd.DataFrame({
        "open": [100]*5,
        "high": [100]*5,
        "low":  [100]*5,
        "close":[100]*5,
    }, index=idx)

    strat = DummyStrategy(data=ohlc.copy(), name="test")

    run = strat.run(ohlc)

    assert run.name == "test"
    assert "net_value" in run.daily_df
    assert "trade_count" in run.daily_df
    assert "turnover" in run.daily_df
    assert isinstance(run.stats, pd.Series)


# =========================
# 5. position ffill
# =========================
def test_position_ffill(monkeypatch):
    monkeypatch.setattr(strategy_module, "run_backtest", mock_run_backtest)

    class SparseStrategy(BaseStrategy):
        def build_signals(self, df):
            df = df.iloc[[0]]
            df["position"] = 1
            return df

    idx = pd.date_range("2024-01-01", periods=5)

    ohlc = pd.DataFrame({
        "open": [100]*5,
        "high": [100]*5,
        "low":  [100]*5,
        "close":[100]*5,
    }, index=idx)

    strat = SparseStrategy(data=ohlc.copy(), name="sparse")
    run = strat.run(ohlc)

    assert (run.daily_df["qty"] != 0).any()


# =========================
# 6. error: missing position
# =========================
def test_missing_position(monkeypatch):
    monkeypatch.setattr(strategy_module, "run_backtest", mock_run_backtest)

    class BadStrategy(BaseStrategy):
        def build_signals(self, df):
            return df

    idx = pd.date_range("2024-01-01", periods=3)

    ohlc = pd.DataFrame({
        "open": [100]*3,
        "high": [100]*3,
        "low":  [100]*3,
        "close":[100]*3,
    }, index=idx)

    strat = BadStrategy(data=ohlc.copy(), name="bad")

    with pytest.raises(ValueError):
        strat.run(ohlc)


# =========================
# 7. trade_count & turnover
# =========================
def test_trade_count_turnover(monkeypatch):
    monkeypatch.setattr(strategy_module, "run_backtest", mock_run_backtest)

    idx = pd.date_range("2024-01-01", periods=5)

    ohlc = pd.DataFrame({
        "open": [100]*5,
        "high": [100]*5,
        "low":  [100]*5,
        "close":[100]*5,
    }, index=idx)

    strat = DummyStrategy(data=ohlc.copy(), name="test")

    run = strat.run(ohlc)

    assert run.daily_df["trade_count"].sum() > 0
    assert run.daily_df["turnover"].sum() >= 0