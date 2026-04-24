import pandas as pd
import numpy as np
from dataclasses import dataclass

from Components.backtest_simple import Backtester, run_backtest


MARKET_VALUE = 10000
FREQ = '1min'
COMMISSION_RATE = 0.0
SLIPPAGE_BPS = 0.0


def compute_stats(result: pd.DataFrame, initial_cash: float, freq: str) -> pd.Series:
    """
    Compute performance statistics for a backtest result.

    This function calculates key performance metrics such as total return,
    annualized return, volatility, Sharpe ratio, maximum drawdown, win rate,
    and number of trades.

    Args:
        result (pd.DataFrame):
            Backtest result DataFrame. Must contain the following columns:
                - 'net_value': normalized equity curve
                - 'ret': per-period return
                - 'dd': drawdown series
                - 'trade_count': indicator of trades (1 if trade occurred)
        initial_cash (float):
            Initial capital used in the backtest.
        freq (str):
            Data frequency (e.g., '1min', '5min', '1D') used to annualize metrics.

    Returns:
        pd.Series:
            A Series containing:
                - total_return (float): cumulative return
                - annual_return (float): annualized return
                - annual_vol (float): annualized volatility
                - sharpe (float): Sharpe ratio
                - max_drawdown (float): maximum drawdown
                - win_rate (float): fraction of positive returns
                - trades (int): number of trades executed

    Notes:
        - Annualization is performed using Backtester._annualize_factor().
        - If volatility is zero or undefined, Sharpe ratio is set to NaN.
        - Returns assume no risk-free rate (pure excess return not considered).
    """
    net_value = result['net_value']
    total_return = net_value.iloc[-1] - 1
    ann_fac = Backtester._annualize_factor(freq)
    ret = result['ret']

    if np.isnan(ann_fac) or ret.empty:
        ann_return = np.nan
        ann_vol = np.nan
        sharpe = np.nan
    else:
        mu = ret.mean()
        sigma = ret.std(ddof=1)
        if np.isnan(sigma) or sigma == 0:
            ann_return = np.nan
            ann_vol = np.nan
            sharpe = np.nan
        else:
            ann_return = (1 + mu) ** ann_fac - 1
            ann_vol = sigma * np.sqrt(ann_fac)
            sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    win_rate = (ret > 0).mean()
    trades = (result['trade_count'] > 0).sum()

    stats = pd.Series(
        {
            'total_return': total_return,
            'annual_return': ann_return,
            'annual_vol': ann_vol,
            'sharpe': sharpe,
            'max_drawdown': result['dd'].min(),
            'win_rate': win_rate,
            'trades': trades,
        }
    )
    return stats


@dataclass
class StrategyRun:
    """
    Container for strategy execution results.

    Attributes:
        name (str):
            Strategy name.
        signal_df (pd.DataFrame):
            DataFrame containing generated signals (e.g., position).
        daily_df (pd.DataFrame):
            Backtest result DataFrame with enriched metrics.
        stats (pd.Series):
            Performance statistics.
        backtester (Backtester):
            Backtester instance used to run the simulation.
    """
    name: str
    signal_df: pd.DataFrame
    daily_df: pd.DataFrame
    stats: pd.Series
    backtester: Backtester


class BaseStrategy:
    """
    Base class for trading strategies.

    Subclasses must implement the `build_signals` method, which generates
    trading signals (positions) from input data.
    """

    def __init__(self, data: pd.DataFrame, name: str):
        """
        Initialize the strategy.

        Args:
            data (pd.DataFrame):
                Input market data used to generate signals.
            name (str):
                Strategy name.
        """
        self.data = data
        self.name = name

    def build_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.

        This method must be implemented by subclasses.

        Args:
            df (pd.DataFrame):
                Input data.

        Returns:
            pd.DataFrame:
                Must contain a 'position' column representing target position.

        Raises:
            NotImplementedError:
                If not implemented in subclass.
        """
        raise NotImplementedError

    def run(
        self,
        ohlc: pd.DataFrame,
        initial_cash: float = MARKET_VALUE,
        freq: str = FREQ,
        plot: bool = False,
        **bt_kwargs,
    ) -> StrategyRun:
        """
        Execute the strategy and run backtest.

        Args:
            ohlc (pd.DataFrame):
                OHLC price data with columns ['open', 'high', 'low', 'close'].
            initial_cash (float, optional):
                Initial capital. Defaults to MARKET_VALUE.
            freq (str, optional):
                Data frequency for annualization. Defaults to FREQ.
            plot (bool, optional):
                Whether to plot equity curve and drawdown. Defaults to False.
            **bt_kwargs:
                Additional keyword arguments passed to the backtester.

        Returns:
            StrategyRun:
                Object containing signals, results, statistics, and backtester.

        Raises:
            ValueError:
                If 'position' column is missing in signal DataFrame.

        Notes:
            - Positions are forward-filled to align with OHLC index.
            - Transaction cost and slippage are configurable via kwargs.
            - Output includes normalized net value and turnover metrics.
        """
        signal_df = self.build_signals(self.data.copy())

        if 'position' not in signal_df:
            raise ValueError("Strategy must output a 'position' column.")

        signal_df = signal_df[['position']]
        position = signal_df['position'].astype(float).reindex(ohlc.index).ffill().fillna(0.0)

        bt_kwargs.setdefault('open_fee', COMMISSION_RATE)
        bt_kwargs.setdefault('close_fee', COMMISSION_RATE)
        bt_kwargs.setdefault('slippage_bps', SLIPPAGE_BPS)
        bt_kwargs.setdefault('multiplier', 1.0)

        bt, result = run_backtest(
            ohlc=ohlc[['open', 'high', 'low', 'close']],
            target_qty=position,
            initial_cash=initial_cash,
            freq=freq,
            plot=False,
            **bt_kwargs,
        )

        enriched = result.copy()
        enriched['net_value'] = enriched['equity'] / bt.initial_cash
        qty_change = enriched['qty'].diff().abs().fillna(0.0)
        enriched['trade_count'] = (qty_change > 0).astype(int)
        enriched['turnover'] = qty_change * enriched['close'] * bt.multiplier

        stats = compute_stats(enriched, bt.initial_cash, freq)

        if plot:
            Backtester.plot_equity_and_drawdown(enriched, title=f'{self.name} Backtest')

        return StrategyRun(self.name, signal_df, enriched, stats, bt)


def plot_run(run: StrategyRun):
    """
    Plot equity curve and drawdown for a strategy run.

    Args:
        run (StrategyRun):
            Strategy run result.
    """
    Backtester.plot_equity_and_drawdown(run.daily_df, title=f'{run.name} Backtest')


from IPython.display import display

def run_and_plot(strategy: BaseStrategy, ohlc_df: pd.DataFrame, freq: str = FREQ, **bt_kwargs) -> StrategyRun:
    """
    Run a strategy, display statistics, and plot results.

    Args:
        strategy (BaseStrategy):
            Strategy instance to execute.
        ohlc_df (pd.DataFrame):
            OHLC price data.
        freq (str, optional):
            Data frequency. Defaults to FREQ.
        **bt_kwargs:
            Additional backtest parameters.

    Returns:
        StrategyRun:
            Result object containing all outputs.

    Notes:
        - This is a convenience wrapper combining run + display + plot.
        - Suitable for notebook-based analysis workflows.
    """
    run = strategy.run(ohlc_df, freq=freq, plot=False, **bt_kwargs)
    display(run.stats.to_frame(name=strategy.name))
    plot_run(run)
    return run