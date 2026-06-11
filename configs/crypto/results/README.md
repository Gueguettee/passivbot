# Crypto configs — backtest results

Full-history backtests (`-sd 2019-09-08 -ed now`, $1000 start, Binance native)
for every config in [`configs/crypto/`](../). Each `results/<config>/` folder holds
`analysis.json` (211 metrics), `balance_and_equity.png`, and `pnl_cumsum.png`.

| Config | Role | Gain (USD) | ADG | Max DD (USD) | Sharpe | Gain (BTC-denom) |
| --- | --- | --- | --- | --- | --- | --- |
| `btc_binance` | meta-opt **base** | 1.58× | 0.019%/d | **90%** | 0.003 | 0.22× |
| `btc_binance_grid` | meta-opt **base** (= btc_binance params) | 1.58× | 0.019%/d | **90%** | 0.003 | 0.22× |
| **`btc_binance_grid_rank4`** | **optimized winner** | **2.50×** | 0.038%/d | **29%** | 0.034 | 0.33× |
| `paxg_binance` | meta-opt **base** | 1.77× | 0.130%/d | 47% | 0.032 | 2.28× |
| **`paxg_binance_rank1`** | **optimized winner** | **2.72×** | 0.229%/d | 52% | 0.056 | **3.49×** |

## Notes

- **`btc_binance` and `btc_binance_grid` are identical** in the backtest — same
  bot params (they're the two meta-optimizer *starting points*; they differ only
  in the `optimize` bounds/scoring, which don't affect a plain backtest). Both
  carry a brutal **90% drawdown** — that's the un-optimized baseline.
- **`btc_binance_grid_rank4`** is the BTC strategy to use: the optimizer cut
  drawdown from 90% → 29% and lifted USD gain to 2.50×, at the cost of trading
  less. It still loses measured in BTC (0.33×) — a USD-accumulation grid, not a
  BTC-beater.
- **`paxg_binance_rank1`** is the best risk-adjusted of the crypto set and the
  only one that **beats hold** in BTC terms (3.49×). Caveat: 52% drawdown, and it
  stopped filling after ~2026-02 in backtest (possible stuck position in the tail).

See the original strategy notes for BTC/PAXG in the session that produced these.
Reproduce any of them with `python src/backtest.py configs/crypto/<name>.json`.
