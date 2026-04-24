from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Fill:
    """
    Trade fill record used for transaction-level analytics.

    This data structure represents an executed trade (fill) and is primarily
    used for performance attribution, PnL decomposition, and execution analysis.

    Attributes:
        time (pd.Timestamp):
            Timestamp of the execution.

        side (int):
            Trade direction indicator.
            +1 for buy (long), -1 for sell (short).

        qty (float):
            Executed quantity. Always positive.

        price (float):
            Execution price.

        notional (float):
            Notional value of the trade:
            price * qty * multiplier.

        fee (float):
            Transaction cost associated with this fill.

        realized_pnl (float):
            Realized profit and loss contributed by this fill.
            Only applies to the portion that closes an existing position.
    """
    time: pd.Timestamp
    side: int
    qty: float
    price: float
    notional: float
    fee: float
    realized_pnl: float


class Backtester:
    """
    Event-driven backtesting engine for position-based trading strategies.

    The engine simulates trading by adjusting positions toward a target quantity
    series. Trades are executed at the next bar's open price (T+1 execution),
    with optional slippage and transaction costs.

    Args:
        open_fee (float, optional):
            Fee rate for opening positions.
        close_fee (float, optional):
            Fee rate for closing positions.
        multiplier (float, optional):
            Contract multiplier.
        slippage_bps (float, optional):
            Slippage in basis points.
        initial_cash (float, optional):
            Initial capital.

    Attributes:
        cash (float):
            Current available cash.
        qty (float):
            Current position size.
        avg_cost (float):
            Average entry price.
        fills (List[Fill]):
            Executed trades.
        equity_series (pd.Series):
            Equity curve.
        position_series (pd.Series):
            Position time series.
    """

    def __init__(
        self,
        open_fee: float = 0.0005,
        close_fee: float = 0.0005,
        multiplier: float = 1.0,
        slippage_bps: float = 0.0,
        initial_cash: float = 10_000.0
    ):
        self.open_fee = open_fee
        self.close_fee = close_fee
        self.multiplier = multiplier
        self.slippage_bps = slippage_bps
        self.initial_cash = float(initial_cash)

        self.cash = float(initial_cash)
        self.qty = 0.0
        self.avg_cost = 0.0

        self.fills: List[Fill] = []
        self.equity_series = None
        self.position_series = None

    @staticmethod
    def _apply_slippage(px_open_next: float, side: int, slippage_bps: float) -> float:
        """
        Apply slippage to execution price.

        Args:
            px_open_next (float): Next bar open price.
            side (int): +1 buy, -1 sell.
            slippage_bps (float): Slippage in basis points.

        Returns:
            float: Adjusted execution price.
        """
        return px_open_next * (1.0 + side * slippage_bps / 10000.0)

    def _trade_once(self, t: pd.Timestamp, px: float, delta_qty: float) -> None:
        """
        Execute a single trade.

        Args:
            t (pd.Timestamp): Execution time.
            px (float): Execution price.
            delta_qty (float): Trade size (+buy, -sell).

        Notes:
            - Handles position closing first, then opening.
            - Updates cash, position, and average cost.
            - Records fills for later analysis.
        """
        if abs(delta_qty) < 1e-12:
            return

        side = 1 if delta_qty > 0 else -1
        remain = abs(delta_qty)

        if self.qty * delta_qty < 0 and self.qty != 0.0:
            close_qty = min(abs(self.qty), remain)
            close_side = -1 if self.qty > 0 else +1

            notional = px * close_qty * self.multiplier
            fee = notional * self.close_fee

            signed_close_qty = close_qty if self.qty > 0 else -close_qty
            realized_pnl = (px - self.avg_cost) * signed_close_qty * self.multiplier

            cash_delta = (-close_side * notional) - fee
            self.cash += cash_delta

            self.qty -= signed_close_qty
            if abs(self.qty) < 1e-12:
                self.qty = 0.0
                self.avg_cost = 0.0

            self.fills.append(Fill(t, close_side, close_qty, px, notional, fee, realized_pnl))
            remain -= close_qty

        if remain > 1e-12:
            open_qty = remain
            notional = px * open_qty * self.multiplier
            fee = notional * self.open_fee

            cash_delta = (-side * notional) - fee
            self.cash += cash_delta

            if self.qty == 0.0:
                self.avg_cost = px
                self.qty = side * open_qty
            else:
                new_abs = abs(self.qty) + open_qty
                self.avg_cost = (abs(self.qty) * self.avg_cost + open_qty * px) / new_abs
                self.qty += side * open_qty

            self.fills.append(Fill(t, side, open_qty, px, notional, fee, 0.0))

    def equity_update(self, close_px: float) -> float:
        """
        Compute portfolio equity.

        Args:
            close_px (float): Current close price.

        Returns:
            float: Portfolio equity.
        """
        return self.cash + self.qty * close_px * self.multiplier

    def run(self, ohlc: pd.DataFrame, target_qty: pd.Series) -> pd.DataFrame:
        """
        Run backtest simulation.

        Args:
            ohlc (pd.DataFrame):
                OHLC data.
            target_qty (pd.Series):
                Target position.

        Returns:
            pd.DataFrame: Result DataFrame.
        """
        ohlc = ohlc[['open', 'high', 'low', 'close']].copy()
        ohlc = ohlc.sort_index()
        target = target_qty.reindex(ohlc.index).ffill().fillna(0.0)

        idx = ohlc.index
        equities = []
        qtys = []

        for i in range(len(idx)):
            t = idx[i]
            close_px = float(ohlc.at[t, 'close'])

            equities.append(self.equity_update(close_px))
            qtys.append(self.qty)

            if i == len(idx) - 1:
                break

            t_next = idx[i + 1]
            open_px_next = float(ohlc.at[t_next, 'open'])

            desired = float(target.at[t])
            delta = desired - self.qty
            if abs(delta) < 1e-12:
                continue

            side = 1 if delta > 0 else -1
            px_exec = self._apply_slippage(open_px_next, side, self.slippage_bps)

            self._trade_once(t_next, px_exec, delta)

        result = self.result_analysis(equities, qtys, ohlc['close'])
        self.equity_series = result['equity']
        self.position_series = result['qty']
        return result

    def result_analysis(self, equities: list, qtys: list, close: pd.Series) -> pd.DataFrame:
        """
        Generate result DataFrame.

        Args:
            equities (list): Equity values.
            qtys (list): Positions.
            close (pd.Series): Prices.

        Returns:
            pd.DataFrame: Result with equity, returns, drawdown.
        """
        idx = close.index
        result = pd.DataFrame({
            'equity': pd.Series(equities, index=idx),
            'qty': pd.Series(qtys, index=idx),
            'close': close
        })
        result['ret'] = result['equity'].pct_change().fillna(0.0)
        cummax = result['equity'].cummax()
        result['dd'] = result['equity'] / cummax - 1.0
        return result

    @staticmethod
    def _annualize_factor(freq: str) -> float:
        """
        Convert frequency to annualization factor.

        Args:
            freq (str): Data frequency.

        Returns:
            float: Annualization factor.
        """
        f = freq.lower()
        if f in ('1min','1m','min','minute'):
            return 252*24*60
        if f in ('1d','d'):
            return 252
        return np.nan

    def print_summary(self, result: pd.DataFrame, freq: str = '1d') -> None:
        """
        Print backtest summary.

        Args:
            result (pd.DataFrame): Backtest results.
            freq (str): Frequency.
        """
        print("==== Backtest Summary ====")

    @staticmethod
    def plot_equity_and_drawdown(result: pd.DataFrame, title: str = "Backtest Results") -> None:
        """
        Plot equity and drawdown.

        Args:
            result (pd.DataFrame): Result DataFrame.
            title (str): Plot title.
        """
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()

        ax1.plot(result.index, result['equity'])
        ax2.fill_between(result.index, result['dd'])

        plt.title(title)
        plt.show()


def run_backtest(
    ohlc: pd.DataFrame,
    target_qty: pd.Series,
    initial_cash: float = 10_000.0,
    open_fee: float = 0.0005,
    close_fee: float = 0.0005,
    slippage_bps: float = 0.0,
    multiplier: float = 1.0,
    freq: str = '1d',
    plot: bool = True
) -> Tuple[Backtester, pd.DataFrame]:
    """
    Run full backtest pipeline.

    Args:
        ohlc (pd.DataFrame): Market data.
        target_qty (pd.Series): Target positions.
        initial_cash (float): Initial capital.
        open_fee (float): Open fee.
        close_fee (float): Close fee.
        slippage_bps (float): Slippage.
        multiplier (float): Multiplier.
        freq (str): Frequency.
        plot (bool): Whether to plot.

    Returns:
        Tuple[Backtester, pd.DataFrame]
    """
    bt = Backtester(open_fee, close_fee, multiplier, slippage_bps, initial_cash)
    result = bt.run(ohlc, target_qty)

    bt.print_summary(result, freq=freq)
    if plot:
        Backtester.plot_equity_and_drawdown(result)

    return bt, result