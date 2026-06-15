#!/usr/bin/env python3
"""Convert Hydromancer xyz 1s parquet -> passivbot 1m .npz for a HIP-3 coin.

Schema (2026 Hydromancer): columns coin, timestamp (tz), open, high, low, close,
volume, volume_quote, trade_count. Aggregates 1s->1m, fills internal gaps to a
contiguous 1m grid (forward-filled price, bv=0), writes one .npz per UTC day with
the structured dtype the passivbot ohlcv_source_dir loader expects.

Usage: python scripts/_convert_hydro_gold.py <parquet> <coin> <out_dir> <YYYY-MM-DD>
"""
import sys
import numpy as np
import duckdb

DT = np.dtype([("ts", "<i8"), ("o", "<f4"), ("h", "<f4"),
               ("l", "<f4"), ("c", "<f4"), ("bv", "<f4")])


def main():
    parquet, coin, out_dir, day = sys.argv[1:5]
    rows = duckdb.query(f"""
        SELECT (epoch_ms(timestamp) // 60000) * 60000 AS ts,
               arg_min(open, timestamp)  AS o,
               max(high)                 AS h,
               min(low)                  AS l,
               arg_max(close, timestamp) AS c,
               sum(volume)               AS bv
        FROM read_parquet('{parquet}')
        WHERE coin = '{coin}'
          AND open > 0 AND high > 0 AND low > 0 AND close > 0
        GROUP BY ts ORDER BY ts
    """).fetchall()
    if not rows:
        print(f"{day}: no rows for {coin}")
        return 1
    ts = np.array([r[0] for r in rows], dtype=np.int64)
    o = np.array([float(r[1]) for r in rows], dtype=np.float64)
    h = np.array([float(r[2]) for r in rows], dtype=np.float64)
    l = np.array([float(r[3]) for r in rows], dtype=np.float64)
    c = np.array([float(r[4]) for r in rows], dtype=np.float64)
    bv = np.array([float(r[5]) for r in rows], dtype=np.float64)

    # contiguous 1m grid first->last; forward-fill price into gaps, bv=0
    grid = np.arange(ts[0], ts[-1] + 60000, 60000, dtype=np.int64)
    idx = {t: i for i, t in enumerate(ts)}
    out = np.zeros(len(grid), dtype=DT)
    out["ts"] = grid
    last_c = c[0]
    for gi, t in enumerate(grid):
        si = idx.get(int(t))
        if si is None:
            out[gi]["o"] = out[gi]["h"] = out[gi]["l"] = out[gi]["c"] = last_c
            out[gi]["bv"] = 0.0
        else:
            out[gi]["o"], out[gi]["h"], out[gi]["l"], out[gi]["c"] = o[si], h[si], l[si], c[si]
            out[gi]["bv"] = bv[si]
            last_c = c[si]
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{day}.npz")
    np.savez_compressed(path, candles=out)
    print(f"{day}: {len(grid)} candles ({len(ts)} real) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
