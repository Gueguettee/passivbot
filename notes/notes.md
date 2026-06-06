These are the multi-objective optimization metrics:

adg (Average Daily Gain) - Mean daily percentage return across the backtest period

drawdown_worst - Maximum peak-to-trough decline in equity (lower is better, so optimizer minimizes this)

loss_profit_ratio - Ratio of total losses to total profits (lower means losses are smaller relative to gains)

mdg_w (Minimum Daily Gain, windowed) - The worst daily gain over rolling windows - rewards consistency and penalizes bad streaks

sharpe_ratio - Risk-adjusted return: (mean return - risk free rate) / std deviation of returns. Higher means better return per unit of risk
