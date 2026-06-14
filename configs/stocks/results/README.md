# Stock-perp configs — backtest results

Backtests for the **Hyperliquid stock-perp** configs in [`configs/stocks/`](../)
(SP500/NVDA HIP-3 `xyz:` coins). Data is the Hydromancer 1s→1m set in
`caches/stock_ohlcv_src` (no auto-download — see the project notes). Each
`results/<config>/` folder holds `analysis.json`, `balance_and_equity.png`, and
`pnl_cumsum.png`.

Periods are short because these perps are newly listed (SP500/NVDA ~2026-03,
effective start ~2026-04-11 after `minimum_coin_age_days`). $1000 start.

| Config | Coin | Gain (USD) | ADG | Max DD | Sharpe |
| --- | --- | --- | --- | --- | --- |
| `config_sp500` | SP500 | 1.095× | 0.168%/d | 7.4% | 0.148 |
| `config_sp500_hl_balanced` | SP500 | 1.109× | 0.196%/d | **3.2%** | 0.246 |
| `config_sp500_hl_opt` | SP500 | 1.100× | 0.180%/d | 3.5% | 0.220 |
| `config_sp500_hl_opt_tuned` | SP500 | 1.116× | 0.207%/d | 4.0% | 0.218 |
| `config_sp500_hl_sharpe` | SP500 | 1.034× | 0.064%/d | 1.3% | 0.244 |
| `config_sp500_david_live` | SP500 | 1.151× | 0.216%/d | 2.2% | 0.326 |
| `config_nvda` | NVDA | 1.191× | 0.219%/d | 26% | 0.053 |
| `config_sp500_legacy_passive` | SP500 | **0.000×** | 0 | **100%** | 0 | ⚠️ |

## Notes

- **`config_sp500_legacy_passive` is degenerate** — it blows up (100% drawdown,
  no profitable trades) on SP500. Its folder has `analysis.json` +
  `balance_and_equity.png` but no `pnl_cumsum.png` (nothing to plot). Don't use it;
  kept only for completeness. Note it shares `base_dir: backtests/config_sp500`
  with `config_sp500`.
- The `hl_*` variants are the tuned SP500 strategies — **`hl_balanced`** has the
  best blend (best return + sub-4% drawdown), **`hl_sharpe`** the steadiest
  (1.3% DD, Sharpe 0.244) but lowest return.
- **`config_nvda`** is the highest USD gain (1.19×) but with a much bigger 26%
  drawdown than the SP500 variants.
- The 4 **Lighter SPY** configs (`config_sp500_lighter`, `config_sp500_opt`,
  `config_sp500_opt_full`, `config_sp500_opt_full_sharpe`) are **not included** —
  no local Lighter data (skipped this session).

Reproduce in Docker (stock data needs `WINDOWS_COMPATIBILITY=1`):

```bash
docker run --rm -e WINDOWS_COMPATIBILITY=1 \
  -v $PWD/configs:/app/configs -v $PWD/caches:/app/caches -v $PWD/backtests:/app/backtests \
  passivbot:latest python src/backtest.py configs/stocks/<config>.json
```
