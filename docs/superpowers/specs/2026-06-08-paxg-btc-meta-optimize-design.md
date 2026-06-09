# PAXG & BTC strategies via meta-optimizer (2026-06-08)

## Goal

Create two freshly-optimized passivbot strategies, each "in the style of" an existing one:

- **PAXG** ← like the **SP500** strategy (reuse its optimize bounds/scoring/limits)
- **BTC** ← like the **HYPE** strategy (reuse its optimize bounds/scoring/limits)

"Optimize fresh": reuse the search space + scoring of the base strategy, but run a new
walk-forward optimization on the new coin via the existing `meta_optimizer/` framework
on the Oracle Cloud instance.

## Decisions (confirmed with user)

| Decision | Choice |
|---|---|
| Deliverable | Optimize fresh for each coin (not just retarget params) |
| Exchange / data | Both on **Binance** native (auto-download, no Hydromancer) |
| Date range | **Max available history** per coin |
| Optimize budget | **Medium**: `iters_per_fold=20000`, `population_size=200` |
| Execution | Existing meta-optimizer on the **Oracle box** (`ubuntu@144.24.236.105`), driven from **WSL** (SSH key `~/.ssh/id_rsa_oracle` lives there) |

Both are crypto perps (`PAXG/USDT:USDT`, `BTC/USDT:USDT` on binanceusdm) — normal data path,
no stock-perp / `ohlcv_source_dir` machinery. Verified Binance listing dates:
**PAXG perp: 2025-03-27** (~14.5 mo), **BTC perp: 2019-09-08** (~6.75 yr).

## Artifacts (4 files, no code)

1. `configs/crypto/paxg_binance.json` — base passivbot config from `configs/stocks/config_sp500_hl_opt.json`,
   retargeted: coin `PAXG` on binance, long-only single position, SP500 optimize bounds +
   scoring `[adg_w_usd, gain_usd, sharpe_ratio_usd]` + limits (DD<0.25, loss/profit<0.1),
   start `2025-03-27`, end `now`.
2. `configs/crypto/btc_binance.json` — base config from `configs/original/hype_dio_masterclass.json`,
   retargeted: coin `BTC` on binance, HYPE optimize bounds + scoring
   `[adg, drawdown_worst, loss_profit_ratio, mdg_w, sharpe_ratio]`, start `2019-09-08`, end `now`.
3. `meta_optimizer/configs/paxg_binance.json` — meta-config: `time_only` walk-forward, 3 rolling
   folds (train 6 / val 2 mo), iters 20000 / pop 200, output `meta_optimize_results/paxg_binance`.
4. `meta_optimizer/configs/btc_binance.json` — same but folds train 18 / val 6 mo, output
   `meta_optimize_results/btc_binance`.

Single-coin strategies ⇒ `time_only` validation (no coin-folds). Robustness weights/thresholds
mirror the HYPE meta-config (`hype_dio_bybit.json`).

## Execution (from WSL)

```bash
# .env already points at the Oracle box; key in WSL ~/.ssh/id_rsa_oracle
./meta_optimizer/deploy.sh deploy                 # rsync repo (incl. new configs) + build
./meta_optimizer/deploy.sh run \
    --config configs/crypto/paxg_binance.json \
    --meta-config meta_optimizer/configs/paxg_binance.json
# (one tmux 'meta_opt' session at a time → run BTC after PAXG finishes)
./meta_optimizer/deploy.sh monitor                # progress
./meta_optimizer/deploy.sh download-best          # rank_1.json = final tuned strategy
```

`deploy.sh run --config <base> --meta-config <meta>`: `--config` becomes cli.py's positional
base config; `--meta-config …` passes through as extra args. The box auto-downloads Binance
history into its own caches.

## Validation done

- Both base configs load + schema-migrate cleanly under passivbot.
- Both meta-configs pass `cli.py … --dry-run` (correct `time_only` scheme, folds, base_config).

## Notes / caveats

- PAXG history is short (~14.5 mo) → its 3 folds are small; results less robust than BTC's.
- The box runs one optimization at a time (tmux `meta_opt`); PAXG then BTC sequentially.
- Final tuned configs come back as `best_configs/rank_1.json` per run; review before any live use.
