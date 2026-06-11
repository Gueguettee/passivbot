# Original HYPE configs — backtest results

The two `hype_dio` strategies (the configs running live on the box). Each folder
already ships its **result file** (`analysis.json`) and **plot**
(`plot_binance.png` / `plot_binance_logy.png`) alongside the config — they are
*not* duplicated into a `results/` subfolder.

Backtest: Binance HYPE, ~2024-11-29 → 2026-06-07, $1000 start.

| Config | Gain (USD) | ADG | Max DD | Sharpe | Gain (BTC-denom) |
| --- | --- | --- | --- | --- | --- |
| **`hype_dio_masterclas_binance_opti`** | **11.11×** | 0.694%/d | 49% | 0.108 | **18.44×** |
| `hype_dio_masterclass` | 3.15× | 0.330%/d | 65% | 0.030 | 5.37× |

## Notes

- **`hype_dio_masterclas_binance_opti` is the best-performing config in the entire
  repo** — 11× USD and 18× in BTC terms over ~18 months — though on HYPE's short,
  explosive history and with a 49% drawdown. The Binance-optimized variant
  massively outperforms the base `hype_dio_masterclass` (3.15×).
- These were **not re-run this session**: the box's `caches/binance/HYPE_USDT:USDT`
  is owned by the live bot (root), so a non-root backtest hits `Permission denied`
  on the cache lock. The shipped `analysis.json` + `plot_binance.png` are the
  authoritative results; leave the live cache untouched.
