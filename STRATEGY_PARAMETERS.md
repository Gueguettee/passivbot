# Strategy Parameters Description

This document describes the parameters used in the Passivbot configuration file (e.g., `hype_top_4pairs.json`).

## Logging

*   `level`: Logging verbosity level.
*   `memory_snapshot_interval_minutes`: Interval in minutes to take a memory usage snapshot.
*   `volume_refresh_info_threshold_seconds`: Threshold in seconds to log volume refresh information.

## Backtest

Configuration for the backtesting engine.

*   `balance_sample_divider`: Divides the timeframe to sample balance updates.
*   `base_dir`: Directory where backtest results are stored.
*   `combine_ohlcvs`: If true, combines OHLCV data from multiple exchanges.
*   `compress_cache`: If true, compresses cached data to save disk space.
*   `end_date`: End date for the backtest (YYYY-MM-DD).
*   `exchanges`: List of exchanges to use for backtesting.
*   `filter_by_min_effective_cost`: If true, filters out trades where cost is lower than minimum effective cost.
*   `gap_tolerance_ohlcvs_minutes`: Maximum allowed gap in OHLCV data in minutes.
*   `start_date`: Start date for the backtest (YYYY-MM-DD).
*   `starting_balance`: Initial balance for the backtest.
*   `btc_collateral_cap`: Cap on BTC collateral usage.
*   `btc_collateral_ltv_cap`: Cap on Loan-to-Value ratio for BTC collateral.
*   `max_warmup_minutes`: Maximum minutes to warm up indicators before backtest starts.
*   `coin_sources`: Dictionary mapping specific coins to exchanges (overrides default selection).
*   `coins`: Dictionary specifying long/short coins for the backtest.
*   `suite`: Configuration for running multiple scenarios (suite mode).

## Bot (Long / Short)

These parameters control the core trading logic. They are defined separately for `long` and `short` sides.

### EMA (Exponential Moving Average) Spans
*   `ema_span_0`, `ema_span_1`: Used to calculate EMA bands (low, high) for initial pricing and unstuck levels.
*   `entry_volatility_ema_span_hours`: Span for calculating volatility used in entry logic.
*   `filter_volatility_ema_span`: Span for volatility filter.
*   `filter_volume_ema_span`: Span for volume filter.

### Initial Entry
*   `entry_initial_ema_dist`: Distance from EMA for the initial entry price.
    *   Long: `EMA_low * (1 - entry_initial_ema_dist)`
    *   Short: `EMA_high * (1 + entry_initial_ema_dist)`
*   `entry_initial_qty_pct`: Percentage of balance/wallet exposure for the initial order quantity.

### Grid Re-entries (DCA)
*   `entry_grid_spacing_pct`: Base percentage spacing between grid orders.
*   `entry_grid_spacing_we_weight`: Adjusts grid spacing based on wallet exposure (WE).
*   `entry_grid_spacing_volatility_weight`: Adjusts grid spacing based on volatility.
*   `entry_grid_double_down_factor`: Multiplier for the quantity of subsequent grid orders (martingale factor).
*   `n_positions`: Maximum number of simultaneous positions (grid levels).

### Trailing Entries
Activates after a favorable excursion followed by a pullback.
*   `entry_trailing_threshold_pct`: Price distance required to activate trailing.
*   `entry_trailing_retracement_pct`: Price distance retracement required to execute the order.
*   `entry_trailing_grid_ratio`: Controls how grid spacing affects trailing behavior.
*   `entry_trailing_double_down_factor`: Adjusts quantity for trailing re-entries.
*   `entry_trailing_threshold_we_weight`, `entry_trailing_threshold_volatility_weight`: Adjusts threshold based on WE and volatility.
*   `entry_trailing_retracement_we_weight`, `entry_trailing_retracement_volatility_weight`: Adjusts retracement based on WE and volatility.

### Take-profit (Close) Grid
*   `close_grid_markup_start`: Markup for the first close order.
*   `close_grid_markup_end`: Markup for the last close order.
*   `close_grid_qty_pct`: Percentage of position to close at each grid level.

### Trailing Closes
Mirrors trailing entry logic but for exits.
*   `close_trailing_threshold_pct`: Activation threshold for trailing close.
*   `close_trailing_retracement_pct`: Retracement percentage to execute close.
*   `close_trailing_grid_ratio`: Adjusts trailing behavior based on grid.
*   `close_trailing_qty_pct`: Quantity percentage for trailing close orders.
*   `close_trailing_threshold_we_weight`: Adjusts threshold based on wallet exposure.
*   `close_trailing_retracement_we_weight`: Adjusts retracement using wallet exposure.

### Auto-Unstucking
Mechanisms to realize losses and free up capital when stuck.
*   `unstuck_close_pct`: Percentage of position to close when unstucking.
*   `unstuck_ema_dist`: Distance from EMA to place unstucking orders.
*   `unstuck_loss_allowance_pct`: Maximum allowed loss (percentage of exposure) before triggering unstucking.
*   `unstuck_threshold`: Wallet exposure ratio threshold to enable unstucking.

### Filters
*   `filter_volatility_drop_pct`: Minimum volatility drop required to trade.
*   `filter_volume_drop_pct`: Minimum volume drop required to trade.

### Risk Management (Wallet Exposure)
*   `total_wallet_exposure_limit`: Global limit on total wallet exposure (leverage cap).
*   `risk_wel_enforcer_threshold`: Threshold for Per-Position Wallet Exposure Limit enforcer.
*   `risk_twel_enforcer_threshold`: Threshold for Total Wallet Exposure Limit enforcer.
*   `risk_we_excess_allowance_pct`: Allowance for exceeding exposure limits slightly before reducing.

## Live

Configuration specific to live trading.

*   `approved_coins`: List of coins allowed for trading (Long/Short).
*   `auto_gs`: Automatic gradient descent (experimental).
*   `inactive_coin_candle_ttl_minutes`: Time-to-live for candles of inactive coins.
*   `execution_delay_seconds`: Delay between order execution loop.
*   `leverage`: Leverage setting for the exchange.
*   `market_orders_allowed`: Whether market orders are permitted.
*   `max_disk/memory_candles_*`: Limits for candle storage.
*   `max_n_cancellations/creations_per_batch`: Rate limits for order management.
*   `user`: user account identifier (e.g. `bybit_01`).

## Optimize

Configuration for the genetic algorithm optimizer.

*   `bounds`: search space (min/max values) for each bot parameter.
*   `iters`: Total number of iterations (generations * population size).
*   `crossover_probability`: Probability of crossover between individuals.
*   `mutation_probability`: Probability of mutation.
*   `population_size`: Number of individuals in the population.
*   `scoring`: List of metrics to optimize for (e.g., `adg` (average daily gain), `drawdown_worst`).
*   `limits`: Constraints to penalize bad individuals (e.g. `penalize_if profit < 0`).

## PBGUI

Parameters related to the Passivbot GUI (PBGUI) tool, used for managing tasks and configurations.
