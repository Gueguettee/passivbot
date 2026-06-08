#!/usr/bin/env python3
"""Download Hyperliquid HIP-3 stock-perp candles from Hydromancer S3 and convert
them into a passivbot ``backtest.ohlcv_source_dir`` 1m tree.

Why this exists: HL stock perps (``xyz:`` coins such as SP500 / NVDA) have no usable
auto-download for historical backtests (HL API ~3.5d, Yahoo ~7d). Hydromancer
(docs.hydromancer.xyz) serves complete history as 1-second Parquet candles in a
requester-pays S3 bucket; this script pulls them and aggregates to the 1-minute
``.npz`` shards passivbot's source-dir loader expects.

Requirements
------------
- ``aws`` CLI v2 with credentials configured (the bucket is *requester-pays*, so
  transfer is billed to your AWS account).
- Python deps: ``duckdb numpy pandas`` (NOTE: pyarrow crashes on these Parquets;
  duckdb is used to read/aggregate them).

Usage
-----
    python scripts/download_stock_perp_candles.py \
        --coins xyz:SP500 xyz:NVDA \
        --start 2026-03-05 --end 2026-06-07

Then set ``"ohlcv_source_dir": "caches/stock_ohlcv_src"`` in the backtest config
and run the backtest in Docker with ``-e WINDOWS_COMPATIBILITY=1`` (so the loader's
``:``->``_`` folder-name candidates match the folders written here).
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MIN_MS = 60_000
DAY_MS = 86_400_000
CANDLE_DTYPE = np.dtype(
    [("ts", "<i8"), ("o", "<f4"), ("h", "<f4"), ("l", "<f4"), ("c", "<f4"), ("bv", "<f4")]
)


def daterange(start: str, end: str) -> list[str]:
    d = dt.date.fromisoformat(start)
    last = dt.date.fromisoformat(end)
    out = []
    while d <= last:
        out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def s3_key(bucket: str, dex: str, day: str) -> str:
    return f"s3://{bucket}/by_dex/{dex}/candles/1s/date={day}/candles.parquet"


def download_day(bucket: str, dex: str, day: str, raw_dir: Path) -> tuple[str, bool]:
    dest = raw_dir / f"{day}.parquet"
    if dest.exists() and dest.stat().st_size > 0:
        return day, True
    cmd = [
        "aws", "s3", "cp", s3_key(bucket, dex, day), str(dest),
        "--request-payer", "requester", "--only-show-errors",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        if dest.exists():
            dest.unlink()
        return day, False
    return day, True


def download(bucket: str, dex: str, days: list[str], raw_dir: Path, jobs: int) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    ok = miss = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(download_day, bucket, dex, d, raw_dir) for d in days]
        for f in as_completed(futs):
            day, success = f.result()
            if success:
                ok += 1
            else:
                miss += 1
                print(f"  missing/failed: {day}", file=sys.stderr)
    print(f"downloaded {ok}/{len(days)} day files ({miss} missing) -> {raw_dir}")


def fetch_1m(con: duckdb.DuckDBPyConnection, raw_glob: str, coin: str) -> pd.DataFrame:
    q = f"""
    select (epoch_ms(timestamp)//{MIN_MS})*{MIN_MS} as ts,
           arg_min(open::DOUBLE, timestamp)  as o,
           max(high::DOUBLE)                 as h,
           min(low::DOUBLE)                  as l,
           arg_max(close::DOUBLE, timestamp) as c,
           sum(volume::DOUBLE)               as bv
    from read_parquet('{raw_glob}')
    where coin = '{coin}'
    group by ts order by ts
    """
    return con.execute(q).fetchdf()


def fill_contiguous(df: pd.DataFrame) -> pd.DataFrame:
    full = np.arange(int(df["ts"].iloc[0]), int(df["ts"].iloc[-1]) + MIN_MS, MIN_MS)
    g = df.set_index("ts").reindex(full)
    g["c"] = g["c"].ffill()
    for col in ("o", "h", "l"):
        g[col] = g[col].fillna(g["c"])
    g["bv"] = g["bv"].fillna(0.0)
    return g.reset_index().rename(columns={"index": "ts"})


def convert(raw_dir: Path, out_root: Path, coins: list[str], exchange: str) -> None:
    raw_glob = str(raw_dir / "*.parquet").replace("\\", "/")
    con = duckdb.connect()
    n_raw = con.execute(f"select count(*) from read_parquet('{raw_glob}')").fetchone()[0]
    print(f"raw 1s rows across all files: {n_raw:,}")
    for coin in coins:
        folder = coin.replace("/", "_").replace(":", "_")  # 'xyz:SP500' -> 'xyz_SP500'
        df = fetch_1m(con, raw_glob, coin)
        if df.empty:
            print(f"!! {coin}: NO ROWS (check coin name; parquet uses the 'xyz:' prefix)")
            continue
        n_before = len(df)
        df = fill_contiguous(df)
        assert (np.diff(df["ts"].values) == MIN_MS).all(), f"{coin} not contiguous"
        out_dir = out_root / exchange / "1m" / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        days = df["ts"].values // DAY_MS
        written = 0
        for day in np.unique(days):
            sub = df[days == day]
            arr = np.empty(len(sub), dtype=CANDLE_DTYPE)
            for src, dst in (("ts", "ts"), ("o", "o"), ("h", "h"), ("l", "l"), ("c", "c"), ("bv", "bv")):
                arr[dst] = sub[src].values
            day_str = pd.to_datetime(int(day) * DAY_MS, unit="ms", utc=True).strftime("%Y-%m-%d")
            np.savez(out_dir / f"{day_str}.npz", candles=arr)
            written += 1
        first = pd.to_datetime(int(df["ts"].iloc[0]), unit="ms", utc=True)
        last = pd.to_datetime(int(df["ts"].iloc[-1]), unit="ms", utc=True)
        print(
            f"{coin} -> {out_dir.relative_to(REPO_ROOT)}: {n_before:,} real 1m, "
            f"{len(df) - n_before:,} filled, {written} day files [{first} .. {last}]"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coins", nargs="+", default=["xyz:SP500", "xyz:NVDA"],
                    help="coins as they appear in the parquet 'coin' column (xyz:-prefixed)")
    ap.add_argument("--start", default="2026-03-05", help="first UTC day (YYYY-MM-DD)")
    ap.add_argument("--end", default="2026-06-07", help="last UTC day (YYYY-MM-DD)")
    ap.add_argument("--bucket", default="hydromancer-reservoir")
    ap.add_argument("--dex", default="xyz", help="HIP-3 builder dex prefix in the bucket")
    ap.add_argument("--exchange", default="hyperliquid", help="exchange folder under ohlcv_source_dir")
    ap.add_argument("--raw-dir", default=None, help="scratch dir for downloaded parquets")
    ap.add_argument("--out-root", default=str(REPO_ROOT / "caches" / "stock_ohlcv_src"),
                    help="ohlcv_source_dir root to write 1m .npz into")
    ap.add_argument("--jobs", type=int, default=12, help="parallel S3 downloads")
    ap.add_argument("--skip-download", action="store_true", help="convert already-downloaded parquets only")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else REPO_ROOT / "caches" / "_hydro_raw"
    days = daterange(args.start, args.end)
    print(f"range {args.start}..{args.end} ({len(days)} days), coins={args.coins}")
    if not args.skip_download:
        download(args.bucket, args.dex, days, raw_dir, args.jobs)
    convert(raw_dir, Path(args.out_root), args.coins, args.exchange)


if __name__ == "__main__":
    main()
