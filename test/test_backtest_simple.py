import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



import pytest
import pandas as pd
import numpy as np

from Components.backtest_simple import Backtester


# =========================
# 1. slippage
# =========================
def test_apply_slippage():
    px = 100
    assert Backtester._apply_slippage(px, 1, 10) == pytest.approx(100 * 1.001)
    assert Backtester._apply_slippage(px, -1, 10) == pytest.approx(100 * 0.999)


# =========================
# 2. trade: open long
# =========================
def test_open_long_position():
    bt = Backtester()
    t = pd.Timestamp("2024-01-01")

    bt._trade_once(t, 100, 1)

    assert bt.qty == 1
    assert bt.avg_cost == 100
    assert len(bt.fills) == 1
    assert bt.fills[0].side == 1


# =========================
# 3. trade: close long
# =========================
def test_close_long_position():
    bt = Backtester()
    t = pd.Timestamp("2024-01-01")

    bt._trade_once(t, 100, 1)
    bt._trade_once(t, 110, -1)

    assert bt.qty == 0
    assert bt.avg_cost == 0
    assert len(bt.fills) == 2

    # 盈利应该为正
    assert bt.fills[-1].realized_pnl > 0


# =========================
# 4. reverse position (long -> short)
# =========================
def test_reverse_position():
    bt = Backtester()
    t = pd.Timestamp("2024-01-01")

    bt._trade_once(t, 100, 1)
    bt._trade_once(t, 90, -2)

    assert bt.qty == -1
    assert len(bt.fills) == 3  # close + open


# =========================
# 5. no trade
# =========================
def test_zero_delta_no_trade():
    bt = Backtester()
    t = pd.Timestamp("2024-01-01")

    bt._trade_once(t, 100, 0)

    assert bt.qty == 0
    assert len(bt.fills) == 0


# =========================
# 6. equity update
# =========================
def test_equity_update():
    bt = Backtester(initial_cash=1000)
    bt.qty = 1

    equity = bt.equity_update(100)

    assert equity == 1100


# =========================
# 7. run basic flow
# =========================
def test_run_basic():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")

    ohlc = pd.DataFrame({
        "open": [100, 101, 102, 103, 104],
        "high": [100, 101, 102, 103, 104],
        "low":  [100, 101, 102, 103, 104],
        "close":[100, 101, 102, 103, 104],
    }, index=idx)

    target = pd.Series([0, 1, 1, 0, 0], index=idx)

    bt = Backtester()
    result = bt.run(ohlc, target)

    assert "equity" in result.columns
    assert "ret" in result.columns
    assert len(result) == 5


# =========================
# 8. result_analysis
# =========================
def test_result_analysis():
    bt = Backtester()

    idx = pd.date_range("2024-01-01", periods=3)
    equities = [1000, 1100, 1050]
    qtys = [0, 1, 1]
    close = pd.Series([100, 110, 105], index=idx)

    result = bt.result_analysis(equities, qtys, close)

    assert "dd" in result.columns
    assert result["dd"].min() <= 0


# =========================
# 9. annualize factor
# =========================
def test_annualize_factor():
    assert Backtester._annualize_factor("1d") == 252
    assert Backtester._annualize_factor("1min") > 0
    assert np.isnan(Backtester._annualize_factor("unknown"))


# =========================
# 10. print summary (只覆盖)
# =========================
def test_print_summary(capsys):
    bt = Backtester()

    idx = pd.date_range("2024-01-01", periods=3)
    result = pd.DataFrame({
        "equity": [1000, 1100, 1050],
        "ret": [0, 0.1, -0.045],
        "dd": [0, 0, -0.045]
    }, index=idx)

    bt.print_summary(result)

    captured = capsys.readouterr()
    assert "Backtest Summary" in captured.out