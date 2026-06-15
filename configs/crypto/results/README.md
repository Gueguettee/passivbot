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
| `paxg_binance_paco` † | HL **GOLD** perp, long+short | 2.19× | 0.14%/d | **11.5%** | 0.216 | **3.10×** |
| `paxg_binance_paco_repointed` ‡ | paco params on **Binance PAXG** | 1.03× | 0.05%/d | 15.5% | 0.012 | 1.49× |

† **Different basis** — period 2025-12-22 → 2026-06-14 only (not full history), and
it does **not** trade Binance PAXG: despite the filename it's a third-party ("paco")
config for the **Hyperliquid HIP-3 `xyz:GOLD`** perp, **long + short**. Data is the
Hydromancer 1s→1m set in `caches/stock_ohlcv_src` (no auto-download). Not
comparable row-for-row with the long-only Binance PAXG configs above.

‡ Same 6-month window, but paco's strategy params re-pointed onto **Binance PAXG**
(the instrument the other two trade) for a like-for-like comparison.

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
- **`paxg_binance_paco` is not a Binance PAXG config** — the filename is misleading.
  It's a third-party config trading the **Hyperliquid `xyz:GOLD`** HIP-3 perp,
  long + short, over a 6-month window. That's why its results look nothing like the
  other two PAXG configs: different instrument, different exchange, both directions,
  shorter period.
  - **Data-quality gotcha (fixed):** the first run blew up to a **97% drawdown and
    stopped at 35% completion**. Cause was a single corrupt source candle
    (2026-02-21 02:12:23, all OHLC = 0 in the Hydromancer parquet) — `min(low)`
    aggregation turned that minute into a $0 price, read as a -100% crash →
    liquidation. The converter now filters non-positive prices; the clean run
    completes fully at **2.19× / 11.5% DD / 3.10× BTC**.
  - **Same strategy on the actual PAXG token** (`paxg_binance_paco_repointed`) only
    makes **1.03×** over the same window — so the strong GOLD result is the
    instrument (a trending HL gold perp), not a universally great strategy.

See the original strategy notes for BTC/PAXG in the session that produced these.
Reproduce any of them with `python src/backtest.py configs/crypto/<name>.json`.
