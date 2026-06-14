# Hyperliquid OHLCV Collector

> **Vendored into Passivbot** from [`djienne/download_hyperliquid_ohlcv_data`](https://github.com/djienne/download_hyperliquid_ohlcv_data)
> @ `288b370b`. Hyperliquid's public candle API will not serve historical 1m data (it ignores
> `since` and returns only the latest ~1000 candles), so a hyperliquid backtest cannot auto-download
> its history — this collector builds that history from Hydromancer + archive sources instead.
>
> **One change from upstream:** the OHLCV shards are written directly in passivbot's backtest
> source-dir layout so no staging step is needed. Set `OHLCV_SOURCE_DIR` (default `data/ohlcv_source`)
> and `PB_EXCHANGE_DIR` (default `hyperliquid`); shards land at
> `<OHLCV_SOURCE_DIR>/<PB_EXCHANGE_DIR>/1m/<COIN>/YYYY-MM-DD.npy`. Point the backtest config at it:
>
> ```json
> "backtest": { "ohlcv_source_dir": "scripts/hyperliquid_ohlcv_collector/data/ohlcv_source" }
> ```
>
> Raw S3 caches, quality sidecars, `source_index`, status, and reports keep their upstream locations
> (under `DATA_DIR` and as subdirs of the coin dir). See `configs/to_test/hype_x75_dd21.json` for a
> working example.
>
> **Free mode (no S3 creds):** to fetch only the recent tail from the public API and prove the
> write path without touching requester-pays S3, run once with the S3 sources disabled:
>
> ```bash
> docker compose run --rm \
>   -e RUN_ONCE=1 -e ENABLE_HYDROMANCER_CANDLES=0 -e ENABLE_HYDROMANCER_L2=0 \
>   -e ENABLE_HYPERLIQUID_ARCHIVE_L2=0 -e RECENT_HYDROMANCER_REPAIR_DAYS=0 -e ENABLE_API_TAIL=1 \
>   collector
> ```
>
> Full historical backfill (Hydromancer 1s + archive L2) needs requester-pays S3 creds. By default
> the compose file mounts your machine's `~/.aws` (read-only) and the collector resolves the standard
> AWS credential chain — **no `aws.env` required**. On Linux/macOS set `AWS_DIR=$HOME/.aws`.

Dockerized collector for building passivbot-compatible Hyperliquid 1 minute OHLCV history for HYPE from API candles and cached S3 historical sources. It preserves raw requester-pays downloads, quality reports, and per-minute provenance so gap repairs can run without repeatedly redownloading expensive data.

Continuously collects Hyperliquid 1 minute HYPE data into passivbot-compatible daily `.npy` shards:

```text
data/ohlcv_source/hyperliquid/1m/HYPE/YYYY-MM-DD.npy
```

Each shard is a float64 array with columns:

```text
timestamp_ms, open, high, low, close, base_volume
```

Completed UTC days are saved as 1440 rows. The current UTC day grows only to the latest known candle, so it does not invent future candles. Internal missing minutes are filled with previous close and zero volume to keep timestamp continuity.

Each `.npy` shard has a machine-readable quality sidecar:

```text
data/ohlcvs_hyperliquid/HYPE/quality/YYYY-MM-DD.json
```

The sidecar records observed minutes, filled minute ranges, raw missing archive hours, real-vs-zero volume, and `strict_backtest_ok`. A day can be passivbot-loadable and timestamp-continuous while still having `strict_backtest_ok=false` because some source minutes were filled.

Each day also has a per-minute provenance index:

```text
data/ohlcvs_hyperliquid/HYPE/source_index/YYYY-MM-DD.npy
```

The source index is a `uint8[1440]` array with one code per UTC minute: `0=missing`, `1=hydromancer_1s`, `2=api_candleSnapshot`, `3=remote_passivbot_1m`, `4=hydromancer_l2_midpoint_zero_volume`, `5=hyperliquid_archive_l2_midpoint_zero_volume`, `6=filled`, and `7=unknown_legacy`. Complete L2-derived days are reported as synthetic provenance, but they are not automatically redownloaded just to upgrade source quality.

Startup first scans the configured completed history range for missing or invalid daily shards, repairs all detected
holes, then runs a bounded, resumable historical pass before the live API updater. `MAX_HISTORICAL_DAYS_PER_CYCLE`
only limits the normal cursor backfill after holes are repaired.

1. Hydromancer Reservoir 1s candles, aggregated to true 1m OHLCV.
2. Trusted imported 1m shards tagged as `remote_passivbot_1m`, when present.
3. Hydromancer Reservoir 1m L2 book snapshots, reconstructed as midpoint OHLC with volume `0.0`.
4. Official `hyperliquid-archive` L2 book snapshots for older main-dex history, also midpoint OHLC with volume `0.0`.
5. Hyperliquid `candleSnapshot` API for the recent tail.

L2-derived files are explicitly tagged in `.source_report.json`; their volume is not real traded volume. Trusted imported 1m shards are also tagged there and treated as real-volume, full-priority 1m OHLCV so they are not redownloaded from AWS.

Raw source data is also kept locally so future reconstruction changes do not need to redownload S3 data:

```text
data/raw/hyperliquid/hydromancer/hyperliquid/candles_1s/HYPE/YYYY-MM-DD.parquet
data/raw/hyperliquid/hydromancer/hyperliquid/orderbook_1m/HYPE/YYYY-MM-DD.parquet
data/raw/hyperliquid/hyperliquid_archive/market_data_l2Book/HYPE/YYYYMMDD/HH.lz4
```

Hydromancer S3 reads are always persisted as raw parquet before processing, and missing Hydromancer paths are memoized with `.missing.json` markers. Existing raw files and missing markers are reused so the same requester-pays S3 object is not fetched repeatedly; use `FORCE=1` only when you deliberately want to recheck marked-missing S3 paths. With `CACHE_ALL_RAW_L2=1`, L2 raw caching is independent from OHLCV source priority. The `.npy` output still prefers true Hydromancer/API candles, but the collector also stores Hydromancer 1m L2 and official archive L2 when available.

Official archive L2 downloads also keep `.missing.json` markers and record both non-padded and padded hour key attempts when a file is absent. The collector preserves its local canonical raw path and avoids duplicate raw cache files.

`RECENT_HYDROMANCER_REPAIR_DAYS` fills the recent completed tail from Hydromancer 1s candles when the API cannot look back far enough, and caches recent Hydromancer L2 raw files without waiting for the main chronological backfill to reach the current month.

Requester-pays S3 credentials are resolved from the standard AWS chain. By default the compose file
mounts your machine's `~/.aws` read-only (via `${AWS_DIR:-${USERPROFILE}/.aws}`) and the collector
hands those keys to both boto3 and DuckDB (DuckDB's `credential_chain` provider can't carry
`REQUESTER_PAYS`, so explicit keys are bridged in) — so **no `aws.env` is needed** if your machine
already has working AWS credentials (the same ones the `aws` CLI uses). Alternatives still supported:
set `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` in the environment, or copy `aws.env.example` to
`aws.env` (git-ignored) and fill it in. On Linux/macOS point the mount at your home dir with
`AWS_DIR=$HOME/.aws`.

Docker writes all collector state under `/data`, which `docker-compose.yml` bind-mounts to local `./data`. Processed `.npy` shards, quality sidecars, reports, raw S3 cache files, and `.missing.json` markers are visible outside the container.

Runtime status is written under the same mounted data directory:

```text
data/status/collector.json
data/status/hyperliquid_archive_preflight.json
```

`collector.json` shows the current phase, market, date/hour, repair counts, raw cache counters, and last error. The archive preflight report summarizes official archive hour availability learned from local raw files, missing markers, and archive attempts.

Processed HYPE `.npy` snapshots may be tracked in git even though `data/` is ignored. Only the daily OHLCV shards and matching `source_index` shards should be force-added, and incomplete live days should be left untracked. As a rule, commit through at most yesterday-minus-one UTC day; keep today's and yesterday's files out of git.

## Run

```powershell
docker compose up -d --build
docker logs -f hyperliquid-ohlcv-collector
```

## Maintenance

Rebuild quality metadata from local raw cache without S3 redownload:

```powershell
docker compose run --rm --name hyperliquid-ohlcv-maintenance -e REPAIR_METADATA_ONLY=1 -e VERIFY_FULL_HISTORY=1 -e RUN_ONCE=1 collector
```

This also rebuilds missing or invalid `source_index/YYYY-MM-DD.npy` files from local shards and metadata.

Verify the configured completed history range:

```powershell
docker compose run --rm --name hyperliquid-ohlcv-verify -e VERIFY_FULL_HISTORY=1 -e RUN_ONCE=1 collector
```

The full report is written to:

```text
data/ohlcvs_hyperliquid/HYPE/.full_history_report.json
```

The full-history report includes source-index issues, per-day source-code counts, synthetic-source days, unknown legacy provenance, and whether provenance checks passed.

Default settings collect only `HYPE` twice per day. To collect more symbols, edit `SYMBOLS` in `docker-compose.yml`, for example `SYMBOLS=HYPE,BTC,ETH`.

## Stock perps (HIP-3, e.g. SP500)

The same collector handles HIP-3 markets — set the `xyz` dex and use the prefixed coin name.
Hydromancer serves these under `by_dex/xyz/...`, and the shards are written to the passivbot
folder `xyz_SP500` (matching the coin `xyz:SP500`), so a stock-perp backtest can read them with no
extra steps:

```bash
docker compose run --rm \
  -e HL_DEX=xyz -e SYMBOLS=xyz:SP500 -e EARLIEST_DATE=2026-03-18 \
  collector
```

- `HL_DEX=xyz` routes the Hydromancer reads to `by_dex/xyz/...`; `EARLIEST_DATE` should be the
  market's listing date (SP500 listed 2026-03-18).
- Output: `data/ohlcv_source/hyperliquid/1m/xyz_SP500/YYYY-MM-DD.npy` — point
  `backtest.ohlcv_source_dir` at `.../data/ohlcv_source` exactly as for HYPE.
- HIP-3 markets are **not** in the official `hyperliquid-archive` (main-dex only), so their history
  comes from **Hydromancer 1s candles + the API tail only**. The Hydromancer reads are requester-pays
  but use your machine's `~/.aws` creds via the default mount — **no `aws.env` needed** (the free
  API-tail-only mode still works for just the recent tail).
- A single run uses one `HL_DEX`, so collect main-dex coins (`HYPE`) and `xyz` coins in separate
  runs.

## Tests

`test_collector.py` covers the candle-integrity validation in `validate_daily_array` — the OHLC
invariant (`high` below open/close, etc.) and the non-positive-price guard (a `low = 0` simulates a
total crash and would invalidate a backtest). Run it in the image (no extra deps needed):

```bash
docker compose run --rm --no-deps \
  -v "$(pwd)://app" --entrypoint python collector test_collector.py
```

To scan a collected coin for these anomalies, run `validate_daily_array` over its daily shards (see
the one-liner in the test docstring).
