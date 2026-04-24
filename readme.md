### Strategy Overview

This trading strategy is based on a **moving average crossover framework**, applied across multiple futures markets, including steel, coke, iron ore, base metals (copper, aluminum, nickel), asphalt, polypropylene, soybeans, and palm oil.

The core signal is derived from the interaction between:
- A short-term moving average (5-day)
- A long-term moving average (20-day)

**Trading Logic:**
- A **buy signal** is generated when the 5-day moving average crosses above the 20-day moving average.
- A **sell signal** is generated when the 5-day moving average crosses below the 20-day moving average.

---

### Backtest Performance Summary

- **Annualized Return:** 12.2651%
- **Annualized Volatility:** 53.6993%
- **Sharpe Ratio:** 0.19116
- **Maximum Drawdown:** -58.4735%
- **Drawdown Recovery Period:** 111 days
- **Win Rate:** 48.2569%
- **Number of Trades:** 47,709
- **Trading PnL:** -403,233.82
- **Holding PnL:** 727,940.19
- **Average Profit/Loss Ratio:** 1.1667
- **Return-to-Risk Ratio:** 0.2284
- **Calmar Ratio:** 0.20975

![Backtest Result](image.png)

---

### Analysis and Discussion

#### Profitability

The strategy achieves an annualized return of **12.27%**, indicating a certain level of profitability.

However, the **negative trading PnL (-403,233.82)** suggests that execution or frequent trading introduces significant costs or inefficiencies, offsetting part of the gains.

---

#### Risk Analysis

The annualized volatility is **53.70%**, indicating substantial exposure to market risk.

The **maximum drawdown of -58.47%** is particularly large, implying that the strategy can suffer significant capital loss under adverse market conditions.

---

#### Efficiency (Risk-Adjusted Performance)

- The **Sharpe ratio (0.19)** is relatively low, suggesting limited excess return per unit of risk.
- The **Calmar ratio (0.21)** is also low, indicating weak performance when adjusted for drawdown risk.

Overall, the strategy exhibits **poor risk-adjusted returns**.

---

#### Positioning and Trading Characteristics

- The **positive holding PnL** indicates that longer-term positions contribute positively to returns.
- The strategy executes a **large number of trades**, but the win rate is below 50%, suggesting that high trading frequency does not translate into consistent profitability.

---

### Conclusion

In summary, the strategy demonstrates **moderate profitability**, particularly from holding positions. However, it suffers from:

- High volatility
- Large drawdowns
- Weak risk-adjusted performance (low Sharpe and Calmar ratios)

These issues indicate that the strategy takes on **significant risk without sufficient compensation**.

Future improvements may include:
- Reducing trading frequency or transaction costs
- Enhancing signal quality
- Incorporating risk management mechanisms
- Improving execution efficiency

Overall, further optimization is required to achieve a more robust and scalable trading strategy.