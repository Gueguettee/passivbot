"""
Hyperliquid 1-minute OHLCV collector.

Storage (passivbot backtest source-dir layout):
  <OHLCV_SOURCE_DIR>/<PB_EXCHANGE_DIR>/1m/{COIN}/YYYY-MM-DD.npy
  default: data/ohlcv_source/hyperliquid/1m/{COIN}/YYYY-MM-DD.npy
  columns = [timestamp_ms, open, high, low, close, base_volume]
  Point passivbot's backtest.ohlcv_source_dir at <OHLCV_SOURCE_DIR> to consume these.

Source priority:
  1. Hydromancer Reservoir 1s candles -> true OHLCV
  2. Trusted imported remote passivbot 1m shards -> true OHLCV
  3. Hyperliquid candleSnapshot API -> recent true OHLCV tail
  4. Hydromancer Reservoir 1m L2 book -> midpoint OHLC, volume = 0
  5. Official hyperliquid-archive L2 book -> midpoint OHLC, volume = 0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import signal
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("hyperliquid-ohlcv-collector")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get("BASE_URL", "https://api.hyperliquid.xyz/info")
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
# OHLCV shards are written in passivbot's backtest source-dir layout so they can be
# consumed directly via backtest.ohlcv_source_dir (no staging step). Raw S3 caches,
# status, and reports still live under DATA_DIR.
OHLCV_SOURCE_DIR = Path(os.environ.get("OHLCV_SOURCE_DIR", str(DATA_DIR / "ohlcv_source")))
# Exchange directory name; must match passivbot's to_standard_exchange_name (always
# "hyperliquid" for HL spot/perp and HIP-3 coins).
PB_EXCHANGE_DIR = os.environ.get("PB_EXCHANGE_DIR", "hyperliquid")
SYMBOLS = os.environ.get("SYMBOLS", "HYPE")
HL_DEX = os.environ.get("HL_DEX", "hyperliquid")
EARLIEST_DATE = os.environ.get("EARLIEST_DATE", "2024-11-29")
END_DATE = os.environ.get("END_DATE", "")

ENABLE_HYDROMANCER_CANDLES = os.environ.get("ENABLE_HYDROMANCER_CANDLES", "1").lower() not in {"0", "false", "no"}
ENABLE_HYDROMANCER_L2 = os.environ.get("ENABLE_HYDROMANCER_L2", "1").lower() not in {"0", "false", "no"}
ENABLE_HYPERLIQUID_ARCHIVE_L2 = os.environ.get("ENABLE_HYPERLIQUID_ARCHIVE_L2", "1").lower() not in {"0", "false", "no"}
ENABLE_API_TAIL = os.environ.get("ENABLE_API_TAIL", "1").lower() not in {"0", "false", "no"}

RESERVOIR_BUCKET = os.environ.get("RESERVOIR_BUCKET", "hydromancer-reservoir")
RESERVOIR_REGION = os.environ.get("RESERVOIR_REGION", "ap-northeast-1")
ARCHIVE_BUCKET = os.environ.get("ARCHIVE_BUCKET", "hyperliquid-archive")
ARCHIVE_REGION = os.environ.get("ARCHIVE_REGION", "us-east-1")
AWS_ENV_FILE = os.environ.get("AWS_ENV_FILE", "aws.env")

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "43200"))
MAX_HISTORICAL_DAYS_PER_CYCLE = int(os.environ.get("MAX_HISTORICAL_DAYS_PER_CYCLE", "14"))
MAX_ARCHIVE_HOURS_PER_CYCLE = int(os.environ.get("MAX_ARCHIVE_HOURS_PER_CYCLE", "72"))
HISTORICAL_END_LAG_DAYS = int(os.environ.get("HISTORICAL_END_LAG_DAYS", "2"))
KEEP_RAW_ARCHIVE_L2 = os.environ.get("KEEP_RAW_ARCHIVE_L2", "1").lower() not in {"0", "false", "no"}
CACHE_ALL_RAW_L2 = os.environ.get("CACHE_ALL_RAW_L2", "0").lower() not in {"0", "false", "no"}
RAW_L2_REFRESH_LOOKBACK_DAYS = int(os.environ.get("RAW_L2_REFRESH_LOOKBACK_DAYS", "45"))
RECENT_HYDROMANCER_REPAIR_DAYS = int(os.environ.get("RECENT_HYDROMANCER_REPAIR_DAYS", "10"))

INITIAL_LOOKBACK_MINUTES = int(os.environ.get("INITIAL_LOOKBACK_MINUTES", "4990"))
REPAIR_LOOKBACK_MINUTES = int(os.environ.get("REPAIR_LOOKBACK_MINUTES", "2880"))
QUERY_WINDOW_MINUTES = int(os.environ.get("QUERY_WINDOW_MINUTES", "480"))
REQUEST_INTERVAL = float(os.environ.get("REQUEST_INTERVAL", "0.35"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "2"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "8"))
BACKOFF_BASE = float(os.environ.get("BACKOFF_BASE", "2.0"))
BACKOFF_CAP = float(os.environ.get("BACKOFF_CAP", "180.0"))
VERIFY_DAYS = int(os.environ.get("VERIFY_DAYS", "7"))
FORCE = os.environ.get("FORCE", "").strip().lower() in {"1", "true", "yes", "y"}
RUN_ONCE = os.environ.get("RUN_ONCE", "").strip().lower() in {"1", "true", "yes"}
VERIFY_FULL_HISTORY = os.environ.get("VERIFY_FULL_HISTORY", "").strip().lower() in {"1", "true", "yes", "y"}
REPAIR_METADATA_ONLY = os.environ.get("REPAIR_METADATA_ONLY", "").strip().lower() in {"1", "true", "yes", "y"}
REBUILD_FROM_RAW = os.environ.get("REBUILD_FROM_RAW", "").strip().lower() in {"1", "true", "yes", "y"}

MS_PER_MIN = 60_000
CANDLES_PER_DAY = 1440

SOURCE_PRIORITY = {
    "hydromancer_1s": 100,
    "remote_passivbot_1m": 100,
    "api_candleSnapshot": 90,
    "hydromancer_l2_midpoint_zero_volume": 20,
    "hyperliquid_archive_l2_midpoint_zero_volume": 10,
    "missing": 0,
    "unknown": 0,
}

TRUE_VOLUME_SOURCES = {"hydromancer_1s", "remote_passivbot_1m", "api_candleSnapshot"}

SOURCE_INDEX_CODES = {
    "missing": 0,
    "hydromancer_1s": 1,
    "api_candleSnapshot": 2,
    "remote_passivbot_1m": 3,
    "hydromancer_l2_midpoint_zero_volume": 4,
    "hyperliquid_archive_l2_midpoint_zero_volume": 5,
    "filled": 6,
    "unknown_legacy": 7,
}

SOURCE_INDEX_LABELS = {code: label for label, code in SOURCE_INDEX_CODES.items()}
SOURCE_INDEX_MAX_CODE = max(SOURCE_INDEX_LABELS)

shutdown_event = asyncio.Event()
_save_locks: dict[str, threading.Lock] = {}
_status_lock = threading.Lock()
_collector_status: dict[str, Any] = {}


@dataclass(frozen=True)
class Market:
    request_coin: str
    storage_coin: str
    reservoir_coin: str
    l2_coin: str


class ArchiveHourBudget:
    def __init__(self, hours: int):
        self.hours = max(0, int(hours))

    def take(self) -> bool:
        if self.hours <= 0:
            return False
        self.hours -= 1
        return True


def _handle_signal(*_: Any) -> None:
    log.info("Shutdown signal received")
    shutdown_event.set()


def _get_save_lock(coin: str) -> threading.Lock:
    if coin not in _save_locks:
        _save_locks[coin] = threading.Lock()
    return _save_locks[coin]


def _bool_label(value: bool) -> str:
    return "on" if value else "off"


# ---------------------------------------------------------------------------
# Dates, names, and reports
# ---------------------------------------------------------------------------
def symbols_from_env() -> list[str]:
    return [s.strip() for s in SYMBOLS.split(",") if s.strip()]


def markets_from_env() -> list[Market]:
    return [
        Market(
            request_coin=s,
            storage_coin=storage_name(s),
            reservoir_coin=reservoir_coin(s),
            l2_coin=l2_coin_name(s),
        )
        for s in symbols_from_env()
    ]


def storage_name(request_coin: str) -> str:
    # Coin-folder name in the passivbot source-dir layout. Regular coins are
    # uppercased ("HYPE"). HIP-3 markets keep their lowercase dex prefix joined
    # with "_" so the folder matches passivbot's sanitized loader candidate
    # (coin "xyz:SP500" -> "xyz_SP500"). Mirrors reservoir_coin() so both
    # `SYMBOLS=xyz:SP500` and `SYMBOLS=SP500 HL_DEX=xyz` yield the same folder.
    if ":" in request_coin:
        dex, coin = request_coin.split(":", 1)
    elif HL_DEX and HL_DEX != "hyperliquid":
        dex, coin = HL_DEX, request_coin
    else:
        dex, coin = "", request_coin
    coin = coin.replace("/", "_").upper()
    return f"{dex}_{coin}" if dex else coin


def s3_dex() -> str:
    return HL_DEX if HL_DEX else "hyperliquid"


def reservoir_coin(request_coin: str) -> str:
    if ":" in request_coin:
        return request_coin
    if HL_DEX and HL_DEX != "hyperliquid":
        return f"{HL_DEX}:{request_coin}"
    return request_coin


def l2_coin_name(request_coin: str) -> str:
    return request_coin.split(":", 1)[-1]


def date_str_to_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def date_to_str(day: date) -> str:
    return day.strftime("%Y-%m-%d")


def date_str_to_start_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def ts_to_date_str(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def utc_day_start_ms(ts_ms: int) -> int:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return int(datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp() * 1000)


def completed_candle_end_ms() -> int:
    now = int(time.time() * 1000)
    return max(now - (now % MS_PER_MIN) - MS_PER_MIN, 0)


def historical_end_date() -> date:
    if END_DATE:
        return date_str_to_date(END_DATE)
    return (datetime.now(timezone.utc) - timedelta(days=HISTORICAL_END_LAG_DAYS)).date()


def _coin_fs_name(coin: str) -> str:
    # Mirror passivbot's source-dir sanitization so the folder matches a loader
    # candidate on every OS (Windows-safe; HIP-3 "xyz:SP500" -> "xyz_SP500").
    return coin.replace("/", "_").replace(":", "_")


def coin_dir(coin: str) -> Path:
    # Passivbot source-dir layout: <OHLCV_SOURCE_DIR>/<exchange>/1m/<coin>/YYYY-MM-DD.npy
    d = OHLCV_SOURCE_DIR / PB_EXCHANGE_DIR / "1m" / _coin_fs_name(coin)
    d.mkdir(parents=True, exist_ok=True)
    return d


def day_file(coin: str, date_str: str) -> Path:
    return coin_dir(coin) / f"{date_str}.npy"


def source_report_file(coin: str) -> Path:
    return coin_dir(coin) / ".source_report.json"


def gap_report_file(coin: str) -> Path:
    return coin_dir(coin) / ".gap_report.json"


def full_history_report_file(coin: str) -> Path:
    return coin_dir(coin) / ".full_history_report.json"


def source_index_dir(coin: str) -> Path:
    d = coin_dir(coin) / "source_index"
    d.mkdir(parents=True, exist_ok=True)
    return d


def source_index_file(coin: str, date_str: str) -> Path:
    return source_index_dir(coin) / f"{date_str}.npy"


def cursor_report_file(coin: str) -> Path:
    return coin_dir(coin) / ".backfill_cursor.json"


def api_cursor_file(coin: str) -> Path:
    return coin_dir(coin) / ".api_fetched_until"


def raw_dir(*parts: str) -> Path:
    d = DATA_DIR / "raw" / "hyperliquid" / Path(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw_hydromancer_1s_path(market: Market, date_str: str) -> Path:
    return raw_dir("hydromancer", s3_dex(), "candles_1s", market.storage_coin) / f"{date_str}.parquet"


def raw_hydromancer_1s_missing_path(market: Market, date_str: str) -> Path:
    return raw_dir("hydromancer", s3_dex(), "candles_1s", market.storage_coin) / f"{date_str}.missing.json"


def raw_hydromancer_l2_path(market: Market, date_str: str) -> Path:
    return raw_dir("hydromancer", s3_dex(), "orderbook_1m", market.storage_coin) / f"{date_str}.parquet"


def raw_hydromancer_l2_missing_path(market: Market, date_str: str) -> Path:
    return raw_dir("hydromancer", s3_dex(), "orderbook_1m", market.storage_coin) / f"{date_str}.missing.json"


def raw_archive_l2_path(market: Market, day: date, hour: int) -> Path:
    return raw_dir("hyperliquid_archive", "market_data_l2Book", market.storage_coin, day.strftime("%Y%m%d")) / f"{hour}.lz4"


def raw_archive_l2_missing_path(market: Market, day: date, hour: int) -> Path:
    return (
        raw_dir("hyperliquid_archive", "market_data_l2Book", market.storage_coin, day.strftime("%Y%m%d"))
        / f"{hour}.missing.json"
    )


def quality_dir(coin: str) -> Path:
    d = coin_dir(coin) / "quality"
    d.mkdir(parents=True, exist_ok=True)
    return d


def quality_file(coin: str, date_str: str) -> Path:
    return quality_dir(coin) / f"{date_str}.json"


def status_dir() -> Path:
    d = DATA_DIR / "status"
    d.mkdir(parents=True, exist_ok=True)
    return d


def collector_status_file() -> Path:
    return status_dir() / "collector.json"


def archive_preflight_file() -> Path:
    return status_dir() / "hyperliquid_archive_preflight.json"


def posix_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def update_collector_status(**fields: Any) -> None:
    try:
        with _status_lock:
            _collector_status.update(fields)
            _collector_status["updated_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(collector_status_file(), _collector_status)
    except Exception:
        pass


def increment_collector_status(**increments: int) -> None:
    try:
        with _status_lock:
            for key, value in increments.items():
                try:
                    _collector_status[key] = int(_collector_status.get(key, 0) or 0) + int(value)
                except (TypeError, ValueError):
                    _collector_status[key] = int(value)
            _collector_status["updated_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(collector_status_file(), _collector_status)
    except Exception:
        pass


def write_archive_preflight_day(market: Market, day: date, stats: dict[str, Any]) -> None:
    try:
        report = _read_json(archive_preflight_file(), {})
        if not isinstance(report, dict):
            report = {}
        markets = report.setdefault("markets", {})
        if not isinstance(markets, dict):
            markets = {}
            report["markets"] = markets
        market_report = markets.setdefault(market.storage_coin, {})
        if not isinstance(market_report, dict):
            market_report = {}
            markets[market.storage_coin] = market_report
        days = market_report.setdefault("days", {})
        if not isinstance(days, dict):
            days = {}
            market_report["days"] = days

        day_key = day.strftime("%Y%m%d")
        days[day_key] = {
            "date": date_to_str(day),
            "hours_present": sorted(int(h) for h in stats.get("hours_present", [])),
            "hours_downloaded": sorted(int(h) for h in stats.get("hours_downloaded", [])),
            "hours_missing": sorted(int(h) for h in stats.get("hours_missing", [])),
            "hours_error": sorted(int(h) for h in stats.get("hours_error", [])),
            "hours_disabled": sorted(int(h) for h in stats.get("hours_disabled", [])),
            "download_budget_exhausted": bool(stats.get("download_budget_exhausted", False)),
            "archive_cache_complete": bool(stats.get("is_cache_complete", False)),
            "archive_all_hours_available": bool(stats.get("all_hours_available", False)),
            "checked_utc": datetime.now(timezone.utc).isoformat(),
        }
        market_report.update(
            {
                "bucket": ARCHIVE_BUCKET,
                "region": ARCHIVE_REGION,
                "storage_coin": market.storage_coin,
                "l2_coin": market.l2_coin,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        report["updated_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(archive_preflight_file(), report)
    except Exception:
        pass


def get_source_entry(coin: str, date_str: str) -> dict[str, Any]:
    report = _read_json(source_report_file(coin), {})
    entry = report.get(date_str, {})
    return entry if isinstance(entry, dict) else {}


def get_source(coin: str, date_str: str) -> str:
    return str(get_source_entry(coin, date_str).get("source") or "unknown")


def set_source_entry(coin: str, date_str: str, entry: dict[str, Any]) -> None:
    report = _read_json(source_report_file(coin), {})
    old = report.get(date_str, {})
    if isinstance(old, dict):
        old.update(entry)
        entry = old
    report[date_str] = entry
    _write_json(source_report_file(coin), report)


def get_backfill_cursor(coin: str) -> date:
    data = _read_json(cursor_report_file(coin), {})
    raw = data.get("next_date") if isinstance(data, dict) else None
    if raw:
        try:
            return date_str_to_date(str(raw))
        except ValueError:
            pass
    return date_str_to_date(EARLIEST_DATE)


def set_backfill_cursor(coin: str, next_day: date) -> None:
    _write_json(
        cursor_report_file(coin),
        {"next_date": date_to_str(next_day), "updated_utc": datetime.now(timezone.utc).isoformat()},
    )


def source_priority(source: str) -> int:
    return SOURCE_PRIORITY.get(source, 0)


def source_index_code_for_source(source: str) -> int:
    return SOURCE_INDEX_CODES.get(source, SOURCE_INDEX_CODES["unknown_legacy"])


def source_index_counts(index: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    values, raw_counts = np.unique(index.astype(np.uint8, copy=False), return_counts=True)
    for value, count in zip(values, raw_counts):
        label = SOURCE_INDEX_LABELS.get(int(value), f"code_{int(value)}")
        counts[label] = int(count)
    return counts


def read_source_index(coin: str, date_str: str) -> np.ndarray | None:
    fpath = source_index_file(coin, date_str)
    if not fpath.exists():
        return None
    try:
        index = np.load(str(fpath), allow_pickle=False)
    except Exception:
        return None
    if index.shape != (CANDLES_PER_DAY,):
        return None
    return index.astype(np.uint8, copy=False)


def build_source_index(
    date_str: str,
    arr: np.ndarray | None,
    quality: dict[str, Any],
    source: str,
) -> np.ndarray:
    index = np.zeros(CANDLES_PER_DAY, dtype=np.uint8)
    normalized = normalize_existing(arr)
    if normalized is None or normalized.size == 0:
        return index

    rows = min(int(normalized.shape[0]), CANDLES_PER_DAY)
    present = {
        i
        for i in range(rows)
        if np.isfinite(normalized[i, 1])
        and np.isfinite(normalized[i, 2])
        and np.isfinite(normalized[i, 3])
        and np.isfinite(normalized[i, 4])
    }
    if not present:
        return index

    observed = indexes_from_ranges(quality.get("observed_ranges")) & present
    filled = (
        indexes_from_ranges(quality.get("filled_ranges"))
        | indexes_from_ranges(quality.get("filled_forward_ranges"))
        | indexes_from_ranges(quality.get("filled_backward_ranges"))
    ) & present
    source_code = source_index_code_for_source(source)
    if source_code == SOURCE_INDEX_CODES["missing"]:
        source_code = SOURCE_INDEX_CODES["unknown_legacy"]

    unclassified = present - observed - filled
    for idx in observed | unclassified:
        index[idx] = source_code
    for idx in filled:
        index[idx] = SOURCE_INDEX_CODES["filled"]
    return index


def write_source_index(coin: str, date_str: str, index: np.ndarray) -> dict[str, int]:
    fpath = source_index_file(coin, date_str)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = fpath.parent / f"{fpath.stem}.tmp.npy"
    np.save(str(tmp), index.astype(np.uint8, copy=False))
    tmp.replace(fpath)
    return source_index_counts(index)


def rebuild_source_index_for_day(
    coin: str,
    date_str: str,
    arr: np.ndarray | None,
    quality: dict[str, Any] | None = None,
    source: str | None = None,
) -> dict[str, int]:
    quality = quality if isinstance(quality, dict) else read_quality(coin, date_str)
    source = source or str(quality.get("source") or get_source(coin, date_str) or "unknown")
    index = build_source_index(date_str, arr, quality, source)
    return write_source_index(coin, date_str, index)


def source_index_issues(coin: str, date_str: str, arr: np.ndarray | None = None) -> list[str]:
    issues: list[str] = []
    fpath = source_index_file(coin, date_str)
    if not fpath.exists():
        return ["missing source index"]
    try:
        index = np.load(str(fpath), allow_pickle=False)
    except Exception as exc:
        return [f"source index load error: {exc}"]
    if index.shape != (CANDLES_PER_DAY,):
        issues.append(f"source index shape {index.shape}")
        return issues
    try:
        index = index.astype(np.uint8, copy=False)
    except Exception as exc:
        return [f"source index dtype error: {exc}"]
    if index.size and int(index.max()) > SOURCE_INDEX_MAX_CODE:
        issues.append("source index contains unknown source code")

    normalized = normalize_existing(arr)
    if normalized is not None and normalized.size:
        rows = min(int(normalized.shape[0]), CANDLES_PER_DAY)
        present = np.isfinite(normalized[:rows, 1])
        missing_codes = int(np.count_nonzero(present & (index[:rows] == SOURCE_INDEX_CODES["missing"])))
        if missing_codes:
            issues.append(f"source index has {missing_codes} missing code(s) for present rows")
    return issues


def ensure_source_index_for_day(
    coin: str,
    date_str: str,
    arr: np.ndarray | None,
    quality: dict[str, Any] | None = None,
) -> tuple[dict[str, int], list[str]]:
    issues = source_index_issues(coin, date_str, arr)
    if issues:
        try:
            counts = rebuild_source_index_for_day(coin, date_str, arr, quality)
        except Exception as exc:
            return {}, [*issues, f"source index rebuild failed: {exc}"]
        issues = source_index_issues(coin, date_str, arr)
        return counts, issues
    index = read_source_index(coin, date_str)
    return (source_index_counts(index) if index is not None else {}, issues)


def ranges_from_indexes(indexes: set[int], date_str: str) -> list[dict[str, int]]:
    if not indexes:
        return []
    day_start = date_str_to_start_ms(date_str)
    ordered = sorted(i for i in indexes if 0 <= i < CANDLES_PER_DAY)
    if not ordered:
        return []
    ranges: list[dict[str, int]] = []
    start = prev = ordered[0]
    for idx in ordered[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append(
            {
                "start_index": int(start),
                "end_index": int(prev),
                "start_ts": int(day_start + start * MS_PER_MIN),
                "end_ts": int(day_start + prev * MS_PER_MIN),
                "count": int(prev - start + 1),
            }
        )
        start = prev = idx
    ranges.append(
        {
            "start_index": int(start),
            "end_index": int(prev),
            "start_ts": int(day_start + start * MS_PER_MIN),
            "end_ts": int(day_start + prev * MS_PER_MIN),
            "count": int(prev - start + 1),
        }
    )
    return ranges


def indexes_from_ranges(ranges: Any) -> set[int]:
    out: set[int] = set()
    if not isinstance(ranges, list):
        return out
    for item in ranges:
        if not isinstance(item, dict):
            continue
        try:
            start = int(item.get("start_index"))
            end = int(item.get("end_index"))
        except (TypeError, ValueError):
            continue
        if end < start:
            continue
        out.update(range(max(0, start), min(CANDLES_PER_DAY - 1, end) + 1))
    return out


def read_quality(coin: str, date_str: str) -> dict[str, Any]:
    data = _read_json(quality_file(coin, date_str), {})
    return data if isinstance(data, dict) else {}


def observed_indexes_from_existing(existing: np.ndarray | None, existing_quality: dict[str, Any]) -> set[int]:
    observed = indexes_from_ranges(existing_quality.get("observed_ranges"))
    if observed or existing is None:
        return observed
    arr = normalize_existing(existing)
    if arr is None:
        return set()
    return {i for i, row in enumerate(arr[:CANDLES_PER_DAY]) if not np.isnan(row[1])}


def quality_summary(
    coin: str,
    date_str: str,
    source: str,
    arr: np.ndarray,
    observed: set[int],
    filled_forward: set[int],
    filled_backward: set[int],
    source_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_meta = source_meta or {}
    complete_day = date_str_to_start_ms(date_str) < utc_day_start_ms(completed_candle_end_ms())
    expected_rows = CANDLES_PER_DAY if complete_day else int(arr.shape[0])
    observed = {i for i in observed if 0 <= i < arr.shape[0]}
    filled_forward = {i for i in filled_forward if 0 <= i < arr.shape[0]} - observed
    filled_backward = {i for i in filled_backward if 0 <= i < arr.shape[0]} - observed
    filled = filled_forward | filled_backward
    raw_missing_hours = sorted(int(h) for h in source_meta.get("raw_missing_hours", []))
    timestamp_ok = bool(arr.shape[0] <= 1 or np.all(np.diff(arr[:, 0]) == MS_PER_MIN))
    nan_rows = int(np.isnan(arr[:, 1]).sum()) if arr.size else 0
    integrity_issues = ohlc_integrity_issues(arr)
    strict_ok = (
        bool(complete_day)
        and int(arr.shape[0]) == expected_rows
        and timestamp_ok
        and nan_rows == 0
        and not filled
        and not raw_missing_hours
        and not source_meta.get("raw_errors")
        and not source_meta.get("download_budget_exhausted", False)
        and not integrity_issues
    )
    quality = {
        "quality_version": 1,
        "coin": coin,
        "date": date_str,
        "source": source,
        "source_priority": source_priority(source),
        "volume_is_real": source in TRUE_VOLUME_SOURCES,
        "rows": int(arr.shape[0]),
        "expected_rows": int(expected_rows),
        "complete_day": bool(complete_day),
        "observed_minutes": int(len(observed)),
        "filled_minutes": int(len(filled)),
        "raw_missing_minutes": int(len(filled)),
        "raw_missing_hours": raw_missing_hours,
        "timestamp_continuity_ok": timestamp_ok,
        "nan_rows": nan_rows,
        "ohlc_integrity_ok": not bool(integrity_issues),
        "ohlc_issues": integrity_issues,
        "strict_backtest_ok": strict_ok,
        "observed_ranges": ranges_from_indexes(observed, date_str),
        "filled_forward_ranges": ranges_from_indexes(filled_forward, date_str),
        "filled_backward_ranges": ranges_from_indexes(filled_backward, date_str),
        "filled_ranges": [
            {**r, "fill_type": "forward"} for r in ranges_from_indexes(filled_forward, date_str)
        ]
        + [{**r, "fill_type": "backward"} for r in ranges_from_indexes(filled_backward, date_str)],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    for key in (
        "raw_hours_present",
        "raw_hours_downloaded",
        "raw_hours_error",
        "raw_hours_disabled",
        "download_budget_exhausted",
        "archive_cache_complete",
        "archive_all_hours_available",
    ):
        if key in source_meta:
            quality[key] = source_meta[key]
    return quality


def write_quality(coin: str, date_str: str, quality: dict[str, Any]) -> None:
    _write_json(quality_file(coin, date_str), quality)


# ---------------------------------------------------------------------------
# AWS and S3 helpers
# ---------------------------------------------------------------------------
def load_aws_keys() -> tuple[str | None, str | None]:
    key_id = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("PUBLIC_KEY")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("SECRET_KEY")
    if key_id and secret:
        return key_id, secret

    path = Path(AWS_ENV_FILE)
    if path.exists():
        kv: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip().strip('"').strip("'")
        return (
            kv.get("AWS_ACCESS_KEY_ID") or kv.get("PUBLIC_KEY"),
            kv.get("AWS_SECRET_ACCESS_KEY") or kv.get("SECRET_KEY"),
        )
    return None, None


def sql_literal(value: str) -> str:
    return str(value).replace("'", "''")


def resolve_aws_keys() -> tuple[str | None, str | None, str | None]:
    """Return (key_id, secret, session_token) for S3 access.

    Prefers explicit env / aws.env keys; otherwise resolves the standard AWS
    credential chain (env vars, ~/.aws/credentials, SSO, IAM role) via boto3.
    DuckDB's `credential_chain` provider rejects REQUESTER_PAYS (required for the
    Hydromancer bucket), so we always hand DuckDB explicit keys instead.
    """
    key_id, secret = load_aws_keys()
    if key_id and secret:
        return key_id, secret, os.environ.get("AWS_SESSION_TOKEN")
    try:
        import boto3

        creds = boto3.Session().get_credentials()
        if creds is not None:
            frozen = creds.get_frozen_credentials()
            return frozen.access_key, frozen.secret_key, frozen.token
    except Exception as exc:
        log.warning("could not resolve AWS credentials from default chain: %s", exc)
    return None, None, None


def make_duckdb_connection():
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    key_id, secret, session_token = resolve_aws_keys()
    bucket = sql_literal(RESERVOIR_BUCKET)
    region = sql_literal(RESERVOIR_REGION)
    if key_id and secret:
        token_clause = (
            f", SESSION_TOKEN '{sql_literal(session_token)}'" if session_token else ""
        )
        con.execute(
            "CREATE OR REPLACE SECRET reservoir_s3 "
            f"(TYPE s3, KEY_ID '{sql_literal(key_id)}', SECRET '{sql_literal(secret)}'{token_clause}, "
            f"REGION '{region}', REQUESTER_PAYS true, SCOPE 's3://{bucket}');"
        )
    else:
        # No credentials resolvable; create a chain secret without REQUESTER_PAYS
        # (works only for public/non-requester-pays reads; Hydromancer will fail loudly).
        con.execute(
            "CREATE OR REPLACE SECRET reservoir_s3 "
            f"(TYPE s3, PROVIDER credential_chain, REGION '{region}', "
            f"SCOPE 's3://{bucket}');"
        )
    return con


def make_local_duckdb_connection():
    import duckdb

    return duckdb.connect()


def make_s3_client(region: str):
    import boto3
    from botocore.config import Config

    config = Config(
        retries={"max_attempts": MAX_RETRIES, "mode": "adaptive"},
        max_pool_connections=max(MAX_CONCURRENT * 4, 8),
    )

    key_id, secret = load_aws_keys()
    if key_id and secret:
        return boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            config=config,
        )
    return boto3.client("s3", region_name=region, config=config)


def is_missing_s3_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(k in s for k in ("404", "no such key", "nosuchkey", "not found", "does not exist"))


def reservoir_candles_path(date_str: str) -> str:
    return f"s3://{RESERVOIR_BUCKET}/by_dex/{s3_dex()}/candles/1s/date={date_str}/candles.parquet"


def reservoir_l2_path(market: Market, date_str: str) -> str:
    return (
        f"s3://{RESERVOIR_BUCKET}/by_dex/{s3_dex()}/orderbook/1m/perps/"
        f"date={date_str}/{market.l2_coin}.parquet"
    )


def copy_query_to_parquet(con: Any, query: str, target: Path) -> bool:
    tmp = target.parent / f"{target.stem}.tmp.parquet"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"COPY ({query}) TO '{posix_path(tmp)}' (FORMAT parquet)")
        tmp.replace(target)
        return True
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def cache_reservoir_parquet(
    con: Any,
    market: Market,
    date_str: str,
    src: str,
    query: str,
    target: Path,
    missing_path: Path,
    label: str,
) -> bool:
    update_collector_status(
        phase="raw_cache",
        market=market.storage_coin,
        current_date=date_str,
        current_source=label,
    )
    if target.exists():
        increment_collector_status(raw_cache_hits=1)
        return True
    if missing_path.exists() and not FORCE:
        increment_collector_status(missing_marker_hits=1)
        return False

    try:
        copied = copy_query_to_parquet(con, query, target)
    except Exception as exc:
        if is_missing_s3_error(exc):
            _write_json(
                missing_path,
                {
                    "bucket": RESERVOIR_BUCKET,
                    "path": src,
                    "status": "missing",
                    "checked_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            increment_collector_status(missing_objects=1)
            return False
        log.warning("[%s] %s could not save raw %s: %s", market.storage_coin, date_str, label, str(exc)[:220])
        update_collector_status(last_error=f"{market.storage_coin} {date_str} raw {label}: {str(exc)[:220]}")
        return False

    if copied:
        increment_collector_status(raw_downloads=1)
        try:
            missing_path.unlink()
        except OSError:
            pass
        log.info("[%s] %s saved raw %s parquet", market.storage_coin, date_str, label)
    return copied


# ---------------------------------------------------------------------------
# Candle arrays and metadata
# ---------------------------------------------------------------------------
def normalize_existing(existing: np.ndarray | None) -> np.ndarray | None:
    if existing is None or existing.size == 0:
        return None
    if existing.dtype.names:
        names = existing.dtype.names
        if all(n in names for n in ("ts", "o", "h", "l", "c", "bv")):
            return np.column_stack(
                [
                    existing["ts"].astype(np.float64),
                    existing["o"].astype(np.float64),
                    existing["h"].astype(np.float64),
                    existing["l"].astype(np.float64),
                    existing["c"].astype(np.float64),
                    existing["bv"].astype(np.float64),
                ]
            )
        return None
    arr = np.asarray(existing, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 6:
        return None
    return arr


def target_rows_for_day(date_str: str, candles: list[list[float]], existing: np.ndarray | None) -> int:
    day_start = date_str_to_start_ms(date_str)
    if day_start < utc_day_start_ms(completed_candle_end_ms()):
        return CANDLES_PER_DAY
    max_ts = day_start
    if candles:
        max_ts = max(max_ts, max(int(c[0]) for c in candles))
    existing_arr = normalize_existing(existing)
    if existing_arr is not None and existing_arr.size:
        max_ts = max(max_ts, int(float(existing_arr[-1, 0])))
    max_ts = min(max_ts, completed_candle_end_ms())
    return max(1, min(CANDLES_PER_DAY, int((max_ts - day_start) // MS_PER_MIN) + 1))


def build_daily_array(
    candles: list[list[float]],
    date_str: str,
    existing: np.ndarray | None = None,
    existing_quality: dict[str, Any] | None = None,
    source: str = "unknown",
    coin: str = "",
    source_meta: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    day_start = date_str_to_start_ms(date_str)
    rows = target_rows_for_day(date_str, candles, existing)
    arr = np.full((rows, 6), np.nan, dtype=np.float64)
    arr[:, 0] = np.arange(rows, dtype=np.float64) * MS_PER_MIN + day_start

    existing_quality = existing_quality or {}
    observed = observed_indexes_from_existing(existing, existing_quality)
    filled_forward = indexes_from_ranges(existing_quality.get("filled_forward_ranges"))
    filled_backward = indexes_from_ranges(existing_quality.get("filled_backward_ranges"))

    existing_arr = normalize_existing(existing)
    if existing_arr is not None:
        for row in existing_arr:
            idx = int((int(float(row[0])) - day_start) // MS_PER_MIN)
            if 0 <= idx < rows and not np.isnan(row[1]):
                arr[idx, 1:] = row[1:]

    for candle in candles:
        ts = int(candle[0])
        idx = int((ts - day_start) // MS_PER_MIN)
        if 0 <= idx < rows:
            arr[idx, 1:] = candle[1:]
            observed.add(idx)
            filled_forward.discard(idx)
            filled_backward.discard(idx)

    last_close = np.nan
    for i in range(rows):
        if np.isnan(arr[i, 1]):
            if not np.isnan(last_close):
                arr[i, 1:5] = last_close
                arr[i, 5] = 0.0
                if i not in observed:
                    filled_forward.add(i)
                    filled_backward.discard(i)
        else:
            last_close = arr[i, 4]

    for i in range(rows):
        if not np.isnan(arr[i, 1]):
            if i > 0:
                arr[:i, 1:5] = arr[i, 1]
                arr[:i, 5] = 0.0
                for idx in range(i):
                    if idx not in observed:
                        filled_backward.add(idx)
                        filled_forward.discard(idx)
            break

    filled_forward -= observed
    filled_backward -= observed
    quality = quality_summary(
        coin=coin,
        date_str=date_str,
        source=source,
        arr=arr,
        observed=observed,
        filled_forward=filled_forward,
        filled_backward=filled_backward,
        source_meta=source_meta,
    )
    return arr, quality


def group_candles_by_day(candles: list[list[float]]) -> dict[str, list[list[float]]]:
    out: dict[str, list[list[float]]] = {}
    for candle in candles:
        out.setdefault(ts_to_date_str(int(candle[0])), []).append(candle)
    return out


def update_gap_report(coin: str, date_str: str, arr: np.ndarray, quality: dict[str, Any]) -> None:
    report = _read_json(gap_report_file(coin), {})
    report[date_str] = {
        "rows": int(arr.shape[0]),
        "expected_rows": int(quality.get("expected_rows", arr.shape[0])),
        "complete_day": bool(quality.get("complete_day", False)),
        "timestamp_continuity_ok": bool(quality.get("timestamp_continuity_ok", False)),
        "nan_rows": int(quality.get("nan_rows", 0)),
        "missing_raw_minutes_in_last_write": int(quality.get("raw_missing_minutes", 0)),
        "filled_minutes": int(quality.get("filled_minutes", 0)),
        "strict_backtest_ok": bool(quality.get("strict_backtest_ok", False)),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(gap_report_file(coin), report)


def save_day(
    coin: str,
    date_str: str,
    candles: list[list[float]],
    source: str,
    source_meta: dict[str, Any] | None = None,
) -> bool:
    fpath = day_file(coin, date_str)
    update_collector_status(phase="save_day", market=coin, current_date=date_str, current_source=source)
    incoming_priority = source_priority(source)
    existing_source = get_source(coin, date_str)
    existing_priority = source_priority(existing_source)

    if fpath.exists() and not FORCE and existing_priority > incoming_priority:
        log.debug("[%s] %s: keeping %s over %s", coin, date_str, existing_source, source)
        return False

    lock = _get_save_lock(coin)
    with lock:
        existing = None
        existing_quality: dict[str, Any] = {}
        preserve_existing = fpath.exists() and not FORCE and existing_source == source and existing_priority == incoming_priority
        if preserve_existing:
            try:
                existing = np.load(str(fpath), allow_pickle=False)
                existing_quality = read_quality(coin, date_str)
            except Exception:
                log.warning("[%s] Could not read existing %s; rebuilding", coin, fpath.name)

        arr, quality = build_daily_array(
            candles,
            date_str,
            existing,
            existing_quality,
            source=source,
            coin=coin,
            source_meta=source_meta,
        )
        # Clamp OHLC-invalid candles (e.g. low<=0 fake crash, high<open/close) so a
        # corrupt source value can never be persisted as backtest-ready data.
        arr, n_sanitized = sanitize_ohlc(arr)
        if n_sanitized:
            quality["sanitized_minutes"] = int(n_sanitized)
            quality["ohlc_integrity_ok"] = True
            quality["strict_backtest_ok"] = False
            log.warning(
                "[%s] %s sanitized %d OHLC-invalid candle(s) (low<=0 / high<open-close); "
                "marked strict_backtest_ok=false",
                coin,
                date_str,
                n_sanitized,
            )
        if arr.size == 0 or np.isnan(arr[:, 1]).all():
            return False

        tmp = fpath.parent / f"{fpath.stem}.tmp.npy"
        np.save(str(tmp), arr)
        tmp.replace(fpath)
        write_quality(coin, date_str, quality)
        try:
            source_code_counts = write_source_index(coin, date_str, build_source_index(date_str, arr, quality, source))
        except Exception as exc:
            source_code_counts = {}
            log.warning("[%s] could not write source index for %s: %s", coin, date_str, exc)
        update_gap_report(coin, date_str, arr, quality)
        set_source_entry(
            coin,
            date_str,
            {
                "source": source,
                "source_priority": incoming_priority,
                "volume_is_real": source in TRUE_VOLUME_SOURCES,
                "rows": int(arr.shape[0]),
                "raw_rows_in_last_write": int(len(candles)),
                "observed_minutes": int(quality.get("observed_minutes", 0)),
                "filled_minutes": int(quality.get("filled_minutes", 0)),
                "strict_backtest_ok": bool(quality.get("strict_backtest_ok", False)),
                "raw_missing_hours": quality.get("raw_missing_hours", []),
                "source_index_counts": source_code_counts,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return True


def mark_missing(coin: str, date_str: str, reason: str) -> None:
    set_source_entry(
        coin,
        date_str,
        {
            "source": "missing",
            "reason": reason,
            "volume_is_real": False,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Historical source readers
# ---------------------------------------------------------------------------
def hydromancer_1s_candles(con: Any, market: Market, date_str: str) -> list[list[float]] | None:
    local = raw_hydromancer_1s_path(market, date_str)
    coin = sql_literal(market.reservoir_coin)
    if not local.exists():
        src = reservoir_candles_path(date_str)
        extract_query = (
            f"SELECT * FROM read_parquet('{src}') "
            f"WHERE coin = '{coin}' ORDER BY timestamp"
        )
        if not cache_reservoir_parquet(
            con,
            market,
            date_str,
            src,
            extract_query,
            local,
            raw_hydromancer_1s_missing_path(market, date_str),
            "hydromancer 1s",
        ):
            return None
    from_clause = f"read_parquet('{posix_path(local)}')"
    query = f"""
        SELECT
            CAST(epoch_ms(time_bucket(INTERVAL '1 minute', timestamp)) AS BIGINT) AS t,
            first(open ORDER BY timestamp) AS o,
            max(high) AS h,
            min(low) AS l,
            last(close ORDER BY timestamp) AS c,
            sum(volume) AS v
        FROM {from_clause}
        GROUP BY 1
        ORDER BY 1
    """
    try:
        rows = con.execute(query).fetchall()
    except Exception as exc:
        if is_missing_s3_error(exc):
            return None
        log.warning("[%s] %s hydromancer 1s error: %s", market.storage_coin, date_str, str(exc)[:220])
        return None
    return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5] or 0.0)] for r in rows]


def snapshots_to_1m(rows: list[tuple[int, float, float]]) -> list[list[float]]:
    buckets: dict[int, list[float]] = {}
    for ts, bid, ask in rows:
        if bid <= 0 or ask <= 0:
            continue
        minute = int(ts) - (int(ts) % MS_PER_MIN)
        mid = (float(bid) + float(ask)) / 2.0
        if minute not in buckets:
            buckets[minute] = [float(minute), mid, mid, mid, mid, 0.0]
        else:
            bucket = buckets[minute]
            bucket[2] = max(bucket[2], mid)
            bucket[3] = min(bucket[3], mid)
            bucket[4] = mid
    return [buckets[k] for k in sorted(buckets)]


def cache_hydromancer_l2_raw(con: Any, market: Market, date_str: str) -> bool:
    local = raw_hydromancer_l2_path(market, date_str)
    if local.exists():
        return True

    src = reservoir_l2_path(market, date_str)
    return cache_reservoir_parquet(
        con,
        market,
        date_str,
        src,
        f"SELECT * FROM read_parquet('{src}') ORDER BY block_time_ms",
        local,
        raw_hydromancer_l2_missing_path(market, date_str),
        "hydromancer L2",
    )


def hydromancer_l2_candles(con: Any, market: Market, date_str: str) -> list[list[float]] | None:
    local = raw_hydromancer_l2_path(market, date_str)
    if not local.exists() and not cache_hydromancer_l2_raw(con, market, date_str):
        return None
    src = posix_path(local)
    query = f"""
        SELECT
            CAST(block_time_ms AS BIGINT) AS t,
            CAST(bids[1].px AS DOUBLE) AS bid,
            CAST(asks[1].px AS DOUBLE) AS ask
        FROM read_parquet('{src}')
        WHERE bids[1] IS NOT NULL AND asks[1] IS NOT NULL
        ORDER BY 1
    """
    try:
        rows = con.execute(query).fetchall()
    except Exception as exc:
        if is_missing_s3_error(exc):
            return None
        log.warning("[%s] %s hydromancer L2 error: %s", market.storage_coin, date_str, str(exc)[:220])
        return None
    return snapshots_to_1m([(int(r[0]), float(r[1]), float(r[2])) for r in rows])


def extract_l2_snapshot(item: Any, expected_coin: str) -> tuple[int, float, float] | None:
    if not isinstance(item, dict):
        return None
    if isinstance(item.get("raw"), dict):
        raw = item["raw"]
        data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
    else:
        data = item.get("data") if isinstance(item.get("data"), dict) else item
    if isinstance(data.get("data"), dict):
        data = data["data"]
    coin = str(data.get("coin") or item.get("coin") or "")
    if coin and coin.upper() != expected_coin.upper():
        return None
    levels = data.get("levels") or data.get("l2Book") or data.get("book")
    if not isinstance(levels, list) or len(levels) < 2:
        return None
    bids, asks = levels[0], levels[1]
    if not bids or not asks:
        return None
    try:
        bid = float(bids[0]["px"])
        ask = float(asks[0]["px"])
        ts = int(data.get("time") or item.get("time") or data.get("timestamp") or item.get("timestamp"))
        return ts, bid, ask
    except (KeyError, TypeError, ValueError):
        return None


def archive_l2_key(market: Market, day: date, hour: int) -> str:
    return f"market_data/{day.strftime('%Y%m%d')}/{hour}/l2Book/{market.l2_coin}.lz4"


def archive_l2_candidate_keys(market: Market, day: date, hour: int) -> list[str]:
    day_str = day.strftime("%Y%m%d")
    non_padded = f"market_data/{day_str}/{hour}/l2Book/{market.l2_coin}.lz4"
    padded = f"market_data/{day_str}/{hour:02d}/l2Book/{market.l2_coin}.lz4"
    return [non_padded] if non_padded == padded else [non_padded, padded]


def should_trust_archive_missing_marker(_day: date) -> bool:
    return not FORCE


def empty_archive_stats() -> dict[str, Any]:
    return {
        "hours_present": [],
        "hours_downloaded": [],
        "hours_missing": [],
        "hours_error": [],
        "hours_disabled": [],
        "download_budget_exhausted": False,
        "is_cache_complete": True,
        "all_hours_available": False,
    }


def archive_stats_source_meta(stats: dict[str, Any] | None) -> dict[str, Any]:
    if not stats:
        return {}
    return {
        "raw_hours_present": sorted(int(h) for h in stats.get("hours_present", [])),
        "raw_hours_downloaded": sorted(int(h) for h in stats.get("hours_downloaded", [])),
        "raw_missing_hours": sorted(int(h) for h in stats.get("hours_missing", [])),
        "raw_hours_error": sorted(int(h) for h in stats.get("hours_error", [])),
        "raw_hours_disabled": sorted(int(h) for h in stats.get("hours_disabled", [])),
        "raw_errors": bool(stats.get("hours_error")),
        "download_budget_exhausted": bool(stats.get("download_budget_exhausted", False)),
        "archive_cache_complete": bool(stats.get("is_cache_complete", False)),
        "archive_all_hours_available": bool(stats.get("all_hours_available", False)),
    }


def cache_archive_hour_l2_raw(s3_client: Any, market: Market, day: date, hour: int) -> str:
    from botocore.exceptions import ClientError

    keys = archive_l2_candidate_keys(market, day, hour)
    raw_path = raw_archive_l2_path(market, day, hour)
    missing_path = raw_archive_l2_missing_path(market, day, hour)
    update_collector_status(
        phase="official_archive_raw_cache",
        market=market.storage_coin,
        current_date=date_to_str(day),
        current_hour=int(hour),
        current_source="hyperliquid_archive_l2",
    )
    if raw_path.exists():
        increment_collector_status(raw_cache_hits=1)
        return "exists"
    if missing_path.exists() and should_trust_archive_missing_marker(day):
        increment_collector_status(missing_marker_hits=1)
        return "missing"
    if not KEEP_RAW_ARCHIVE_L2:
        return "disabled"

    missing_codes = {"NoSuchKey", "404", "NotFound"}
    attempted_keys: list[str] = []
    try:
        for key in keys:
            attempted_keys.append(key)
            try:
                obj = s3_client.get_object(Bucket=ARCHIVE_BUCKET, Key=key, RequestPayer="requester")
                raw = obj["Body"].read()
                tmp = raw_path.parent / f"{raw_path.stem}.tmp.lz4"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(raw)
                tmp.replace(raw_path)
                try:
                    missing_path.unlink()
                except OSError:
                    pass
                increment_collector_status(raw_downloads=1)
                log.info("[%s] saved raw official archive L2 %s hour %s",
                         market.storage_coin, day.strftime("%Y%m%d"), hour)
                return "downloaded"
            except ClientError as exc:
                code = str(exc.response.get("Error", {}).get("Code", ""))
                if code in missing_codes:
                    continue
                raise
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        log.warning("[%s] official archive error %s: %s", market.storage_coin, attempted_keys[-1] if attempted_keys else keys[0], code or str(exc)[:160])
        update_collector_status(last_error=f"{market.storage_coin} archive {date_to_str(day)} h{hour}: {code or str(exc)[:160]}")
        return "error"
    except Exception as exc:
        log.warning("[%s] official archive error %s: %s", market.storage_coin, attempted_keys[-1] if attempted_keys else keys[0], str(exc)[:160])
        update_collector_status(last_error=f"{market.storage_coin} archive {date_to_str(day)} h{hour}: {str(exc)[:160]}")
        return "error"

    _write_json(
        missing_path,
        {
            "bucket": ARCHIVE_BUCKET,
            "keys": attempted_keys or keys,
            "status": "missing",
            "checked_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    increment_collector_status(missing_objects=1)
    return "missing"


def cache_official_archive_l2_raw_day(
    s3_client: Any, market: Market, day: date, budget: ArchiveHourBudget
) -> dict[str, Any]:
    stats = empty_archive_stats()
    update_collector_status(
        phase="official_archive_day_preflight",
        market=market.storage_coin,
        current_date=date_to_str(day),
        current_source="hyperliquid_archive_l2",
    )
    for hour in range(24):
        raw_path = raw_archive_l2_path(market, day, hour)
        if raw_path.exists():
            stats["hours_present"].append(hour)
            continue
        missing_path = raw_archive_l2_missing_path(market, day, hour)
        if missing_path.exists() and should_trust_archive_missing_marker(day):
            stats["hours_missing"].append(hour)
            continue
        if not budget.take():
            log.info("[%s] Official archive hourly cap reached while caching raw L2 for %s",
                     market.storage_coin, date_to_str(day))
            stats["download_budget_exhausted"] = True
            stats["is_cache_complete"] = False
            write_archive_preflight_day(market, day, stats)
            return stats
        status = cache_archive_hour_l2_raw(s3_client, market, day, hour)
        if status in {"exists", "downloaded"}:
            stats["hours_present"].append(hour)
            if status == "downloaded":
                stats["hours_downloaded"].append(hour)
        elif status == "missing":
            stats["hours_missing"].append(hour)
        elif status == "disabled":
            stats["hours_disabled"].append(hour)
        elif status == "error":
            stats["hours_error"].append(hour)
        if status == "error":
            stats["is_cache_complete"] = False
            write_archive_preflight_day(market, day, stats)
            return stats

    stats["hours_present"] = sorted(set(stats["hours_present"]))
    stats["hours_missing"] = sorted(set(stats["hours_missing"]))
    stats["all_hours_available"] = len(stats["hours_present"]) == 24
    stats["is_cache_complete"] = not stats["hours_error"] and not stats["download_budget_exhausted"]
    if stats["hours_downloaded"] or stats["hours_missing"] or stats["hours_error"]:
        log.info("[%s] %s raw official archive L2 cache: %s",
                 market.storage_coin, date_to_str(day), stats)
    write_archive_preflight_day(market, day, stats)
    return stats


def parse_archive_hour_l2_raw(market: Market, day: date, hour: int) -> list[tuple[int, float, float]] | None:
    try:
        import lz4.frame
    except ImportError as exc:
        log.warning("[%s] lz4 dependency unavailable while parsing official archive: %s", market.storage_coin, exc)
        return None

    key = archive_l2_key(market, day, hour)
    raw_path = raw_archive_l2_path(market, day, hour)
    if not raw_path.exists():
        return None

    raw = raw_path.read_bytes()

    try:
        text = lz4.frame.decompress(raw).decode("utf-8", errors="replace")
    except Exception as exc:
        log.warning("[%s] lz4 decode error %s: %s", market.storage_coin, key, str(exc)[:160])
        return None

    out: list[tuple[int, float, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        snap = extract_l2_snapshot(item, market.l2_coin)
        if snap is not None:
            out.append(snap)
    return out


def official_archive_l2_candles_from_raw(market: Market, day: date) -> list[list[float]] | None:
    snapshots: list[tuple[int, float, float]] = []
    for hour in range(24):
        rows = parse_archive_hour_l2_raw(market, day, hour)
        if rows:
            snapshots.extend(rows)
    if not snapshots:
        return None
    log.info("[%s] %s official archive L2: %s snapshots from %s hour request(s)",
             market.storage_coin, date_to_str(day), len(snapshots), 24)
    return snapshots_to_1m(sorted(snapshots, key=lambda x: x[0]))


def official_archive_l2_candles(
    s3_client: Any, market: Market, day: date, budget: ArchiveHourBudget
) -> list[list[float]] | None:
    stats = cache_official_archive_l2_raw_day(s3_client, market, day, budget)
    if not stats.get("is_cache_complete", False):
        return None
    return official_archive_l2_candles_from_raw(market, day)


# ---------------------------------------------------------------------------
# API tail
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, max_concurrent: int, min_interval: float):
        self._sem = asyncio.Semaphore(max_concurrent)
        self._interval = min_interval
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    async def __aenter__(self) -> None:
        await self._sem.acquire()
        async with self._lock:
            now = time.monotonic()
            wait = self._last_request + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()

    async def __aexit__(self, *_: Any) -> None:
        self._sem.release()


_rate_limiter: RateLimiter | None = None


async def post_info(session: aiohttp.ClientSession, payload: dict[str, Any]) -> Any | None:
    assert _rate_limiter is not None
    backoff = BACKOFF_BASE
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with _rate_limiter:
                async with session.post(BASE_URL, json=payload, timeout=aiohttp.ClientTimeout(total=45)) as resp:
                    body = await resp.text()
                    if resp.status == 429 or resp.status >= 500:
                        wait = min(backoff * (1.0 + random.uniform(0.0, 0.25)), BACKOFF_CAP)
                        log.warning("HTTP %s for %s, retrying in %.1fs", resp.status, payload.get("type"), wait)
                        await asyncio.sleep(wait)
                        backoff = min(backoff * 2.0, BACKOFF_CAP)
                        continue
                    if resp.status != 200:
                        log.error("HTTP %s for %s: %s", resp.status, payload.get("type"), body[:300])
                        return None
                    return json.loads(body)
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
            wait = min(backoff * (1.0 + random.uniform(0.0, 0.25)), BACKOFF_CAP)
            log.warning("Request error for %s (attempt %s/%s): %s; retrying in %.1fs",
                        payload.get("type"), attempt, MAX_RETRIES, exc, wait)
            await asyncio.sleep(wait)
            backoff = min(backoff * 2.0, BACKOFF_CAP)
    return None


async def discover_markets(session: aiohttp.ClientSession) -> list[Market]:
    requested = symbols_from_env()
    if not requested:
        log.error("SYMBOLS is empty")
        return []
    data = await post_info(session, {"type": "meta"})
    universe = data.get("universe", []) if isinstance(data, dict) else []
    known = {str(item.get("name", "")).upper() for item in universe if item.get("name")}
    markets: list[Market] = []
    for raw in requested:
        req = raw.strip()
        base = req.split(":", 1)[-1].upper()
        if ":" not in req and known and base not in known:
            log.warning("[%s] not found in perp meta; still trying configured sources", req)
        markets.append(
            Market(
                request_coin=req,
                storage_coin=storage_name(req),
                reservoir_coin=reservoir_coin(req),
                l2_coin=l2_coin_name(req),
            )
        )
    log.info("Configured markets: %s", ", ".join(m.request_coin for m in markets))
    return markets


async def fetch_api_candles(session: aiohttp.ClientSession, coin: str, start_ms: int, end_ms: int) -> list[list[float]]:
    payload = {
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": "1m", "startTime": int(start_ms), "endTime": int(end_ms)},
    }
    data = await post_info(session, payload)
    if not isinstance(data, list):
        return []
    candles: dict[int, list[float]] = {}
    for item in data:
        try:
            ts = int(item["t"])
            if start_ms <= ts <= end_ms:
                candles[ts] = [
                    float(ts),
                    float(item["o"]),
                    float(item["h"]),
                    float(item["l"]),
                    float(item["c"]),
                    float(item.get("v", 0.0) or 0.0),
                ]
        except (KeyError, TypeError, ValueError):
            continue
    return [candles[ts] for ts in sorted(candles)]


def get_api_cursor(coin: str) -> int | None:
    fpath = api_cursor_file(coin)
    if not fpath.exists():
        return None
    try:
        return int(fpath.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def set_api_cursor(coin: str, ts_ms: int) -> None:
    api_cursor_file(coin).write_text(str(int(ts_ms)), encoding="utf-8")


def find_api_resume_ts(coin: str) -> int | None:
    cursor = get_api_cursor(coin)
    if cursor is not None:
        return cursor
    files = sorted(f for f in coin_dir(coin).glob("*.npy") if ".tmp" not in f.name)
    if not files:
        return None
    try:
        arr = normalize_existing(np.load(str(files[-1]), allow_pickle=False))
        if arr is not None and arr.size:
            return int(arr[-1, 0]) + MS_PER_MIN
    except Exception:
        pass
    return date_str_to_start_ms(files[-1].stem)


def api_start_ms(coin: str) -> int:
    floor = completed_candle_end_ms() - INITIAL_LOOKBACK_MINUTES * MS_PER_MIN
    cursor = find_api_resume_ts(coin)
    if cursor is None:
        return max(floor, date_str_to_start_ms(EARLIEST_DATE))
    return max(cursor - REPAIR_LOOKBACK_MINUTES * MS_PER_MIN, floor, date_str_to_start_ms(EARLIEST_DATE))


async def update_api_tail(session: aiohttp.ClientSession, market: Market) -> None:
    start_ms = api_start_ms(market.storage_coin)
    end_ms = completed_candle_end_ms()
    if start_ms > end_ms:
        return
    update_collector_status(
        phase="api_tail",
        market=market.storage_coin,
        api_start_ms=int(start_ms),
        api_end_ms=int(end_ms),
    )
    total = 0
    last_seen: int | None = None
    window_ms = max(1, QUERY_WINDOW_MINUTES) * MS_PER_MIN
    window_start = start_ms
    while window_start <= end_ms and not shutdown_event.is_set():
        window_end = min(window_start + window_ms - MS_PER_MIN, end_ms)
        update_collector_status(
            phase="api_tail",
            market=market.storage_coin,
            api_window_start_ms=int(window_start),
            api_window_end_ms=int(window_end),
        )
        candles = await fetch_api_candles(session, market.request_coin, window_start, window_end)
        if candles:
            total += len(candles)
            last_seen = max(last_seen or 0, max(int(c[0]) for c in candles))
            for date_str, day_candles in group_candles_by_day(candles).items():
                save_day(market.storage_coin, date_str, day_candles, "api_candleSnapshot")
        window_start = window_end + MS_PER_MIN
    if last_seen is not None:
        set_api_cursor(market.storage_coin, last_seen + MS_PER_MIN)
    if total:
        log.info("[%s] API tail updated +%s candles", market.storage_coin, total)
    update_collector_status(phase="api_tail_complete", market=market.storage_coin, api_candles=int(total))


# ---------------------------------------------------------------------------
# Backfill and verification
# ---------------------------------------------------------------------------
def cleanup_tmp_files(coin: str) -> None:
    for tmp in coin_dir(coin).glob("*.tmp.npy"):
        try:
            tmp.unlink()
            log.info("[%s] cleaned stale temp %s", coin, tmp.name)
        except OSError:
            pass


def verify_recent_files(coin: str, days: int) -> None:
    if days <= 0:
        return
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days - 1)
    warnings = 0
    for offset in range(days):
        date_str = date_to_str(start + timedelta(days=offset))
        fpath = day_file(coin, date_str)
        if not fpath.exists():
            warnings += 1
            log.warning("[%s] missing recent shard %s", coin, fpath.name)
            continue
        try:
            arr = normalize_existing(np.load(str(fpath), allow_pickle=False))
        except Exception as exc:
            warnings += 1
            log.warning("[%s] could not verify %s: %s", coin, fpath.name, exc)
            continue
        if arr is None or arr.size == 0:
            warnings += 1
            log.warning("[%s] invalid shard %s", coin, fpath.name)
            continue
        if arr.shape[0] > 1 and not np.all(np.diff(arr[:, 0]) == MS_PER_MIN):
            warnings += 1
            log.warning("[%s] timestamp gap in %s", coin, fpath.name)
        if date_str < date_to_str(today) and arr.shape[0] != CANDLES_PER_DAY:
            warnings += 1
            log.warning("[%s] completed day %s has %s rows", coin, fpath.name, arr.shape[0])
        quality = read_quality(coin, date_str)
        if int(quality.get("filled_minutes", 0) or 0) > 0:
            log.warning(
                "[%s] %s is timestamp-continuous but has %s filled minute(s)",
                coin,
                fpath.name,
                quality.get("filled_minutes"),
            )
    if warnings == 0:
        log.info("[%s] recent verification OK (%s days)", coin, days)


def ohlc_integrity_issues(arr: np.ndarray) -> list[str]:
    """Per-candle OHLC/price sanity issues for a [ts, o, h, l, c, v] array.

    Returns an empty list for clean data. Note that a zero/negative price (e.g.
    ``low == 0``) is NOT caught by the classic OHLC invariant (0 <= min(o, c) and
    h >= l >= 0) yet simulates a total crash and invalidates backtests, so it is
    flagged separately.
    """
    issues: list[str] = []
    if arr is None or arr.size == 0 or arr.ndim != 2 or arr.shape[1] != 6:
        return issues
    o, h, l, c = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    if np.any(h < np.maximum(o, c)) or np.any(l > np.minimum(o, c)) or np.any(h < l):
        issues.append("OHLC invariant violation")
    if np.any(arr[:, 1:5] <= 0):
        issues.append("non-positive price")
    if np.any(arr[:, 5] < 0):
        issues.append("negative volume")
    return issues


def sanitize_ohlc(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Clamp OHLC-invalid candles into a valid range using their own open/close.

    - high  -> max(open, high, close)
    - low   -> min(open, low, close); a non-positive low becomes min(open, close)
               so a glitchy ``low == 0`` no longer fakes a total crash
    - low   -> min(low, high) so high >= low always holds
    - volume-> max(volume, 0)

    Returns ``(array, n_modified)``; the input is returned unchanged when already
    clean. Candles whose open/close are themselves non-positive cannot be repaired
    from open/close and are left for ``validate_daily_array`` to flag.
    """
    if arr is None or arr.size == 0 or arr.ndim != 2 or arr.shape[1] != 6:
        return arr, 0
    o, h, l, c, v = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]
    oc_max = np.maximum(o, c)
    oc_min = np.minimum(o, c)
    new_h = np.maximum(h, oc_max)
    safe_low = np.where(l > 0, np.minimum(l, oc_min), oc_min)
    new_l = np.minimum(safe_low, new_h)
    new_v = np.maximum(v, 0.0)
    changed = (new_h != h) | (new_l != l) | (new_v != v)
    n = int(np.count_nonzero(changed))
    if n == 0:
        return arr, 0
    out = arr.copy()
    out[:, 2] = new_h
    out[:, 3] = new_l
    out[:, 5] = new_v
    return out, n


def validate_daily_array(date_str: str, arr: np.ndarray) -> list[str]:
    issues: list[str] = []
    arr = normalize_existing(arr)
    if arr is None or arr.size == 0:
        return ["empty or unreadable array"]
    if arr.ndim != 2 or arr.shape[1] != 6:
        issues.append(f"shape {arr.shape}")
        return issues
    day_start = date_str_to_start_ms(date_str)
    if int(arr[0, 0]) != day_start:
        issues.append("first timestamp is not UTC day start")
    if arr.shape[0] > CANDLES_PER_DAY:
        issues.append(f"too many rows {arr.shape[0]}")
    if arr.shape[0] > 1 and not np.all(np.diff(arr[:, 0]) == MS_PER_MIN):
        issues.append("timestamp gap or duplicate")
    if not np.isfinite(arr).all():
        issues.append("NaN or infinite value")
    if arr.shape[0] and int(arr[-1, 0]) > day_start + (CANDLES_PER_DAY - 1) * MS_PER_MIN:
        issues.append("last timestamp outside UTC day")
    if arr.shape[0]:
        issues.extend(ohlc_integrity_issues(arr))
    return issues


def iter_days(start_day: date, end_day: date):
    day = start_day
    while day <= end_day:
        yield day
        day += timedelta(days=1)


def expected_history_start_date() -> date:
    return date_str_to_date(EARLIEST_DATE)


def completed_history_array_issues(date_str: str, arr: np.ndarray) -> list[str]:
    issues = validate_daily_array(date_str, arr)
    normalized = normalize_existing(arr)
    if normalized is not None and normalized.shape[0] != CANDLES_PER_DAY:
        issues.append(f"completed day has {normalized.shape[0]} rows")
    return issues


def historical_day_repair_reasons(
    coin: str,
    date_str: str,
    source_report: dict[str, Any] | None = None,
) -> list[str]:
    reasons: list[str] = []
    source_report = source_report if isinstance(source_report, dict) else _read_json(source_report_file(coin), {})
    source_entry = source_report.get(date_str, {}) if isinstance(source_report, dict) else {}
    if isinstance(source_entry, dict) and source_entry.get("source") == "missing":
        if not FORCE:
            return []
        reasons.append("source report marks day as missing")

    fpath = day_file(coin, date_str)
    if not fpath.exists():
        reasons.append("missing shard")
        return reasons

    try:
        arr = np.load(str(fpath), allow_pickle=False)
        issues = completed_history_array_issues(date_str, arr)
    except Exception as exc:
        issues = [f"load error: {exc}"]
        arr = None
    if arr is not None:
        quality = read_quality(coin, date_str)
        _, index_issues = ensure_source_index_for_day(coin, date_str, arr, quality)
        issues.extend(index_issues)
    reasons.extend(issues)
    return reasons


def historical_repair_candidates(market: Market, end_day: date) -> list[tuple[date, list[str]]]:
    coin = market.storage_coin
    start_day = expected_history_start_date()
    if end_day < start_day:
        return []

    source_report = _read_json(source_report_file(coin), {})
    candidates: list[tuple[date, list[str]]] = []
    for day in iter_days(start_day, end_day):
        date_str = date_to_str(day)
        reasons = historical_day_repair_reasons(coin, date_str, source_report)
        if reasons:
            candidates.append((day, reasons))
    return candidates


def repair_historical_gaps(
    con: Any | None,
    s3_archive: Any | None,
    market: Market,
    end_day: date,
) -> tuple[int, int, bool]:
    candidates = historical_repair_candidates(market, end_day)
    update_collector_status(
        phase="historical_gap_scan",
        market=market.storage_coin,
        repair_candidates=len(candidates),
        expected_end_day=date_to_str(end_day),
    )
    if not candidates:
        return 0, 0, False

    archive_budget = ArchiveHourBudget(max(MAX_ARCHIVE_HOURS_PER_CYCLE, len(candidates) * 24))
    log.info(
        "[%s] historical gap scan found %s repair candidate day(s); filling all gaps before cursor backfill",
        market.storage_coin,
        len(candidates),
    )
    processed = 0
    written = 0
    stopped = False
    for day, reasons in candidates:
        if shutdown_event.is_set():
            break
        if archive_budget.hours <= 0 and ENABLE_HYPERLIQUID_ARCHIVE_L2:
            log.info("official archive hour cap reached during historical gap repair")
            stopped = True
            break

        date_str = date_to_str(day)
        update_collector_status(
            phase="historical_gap_repair",
            market=market.storage_coin,
            current_date=date_str,
            repair_candidates=len(candidates),
            repair_processed=processed,
            repair_written=written,
            repair_reason="; ".join(reasons),
        )
        log.info("[%s] repairing historical gap %s: %s", market.storage_coin, date_str, "; ".join(reasons))
        day_written, day_complete = backfill_one_day(con, s3_archive, market, day, archive_budget)
        if day_written:
            written += 1
        processed += 1
        if not day_complete:
            log.info(
                "[%s] deferring %s until next cycle because raw cache is incomplete",
                market.storage_coin,
                date_str,
            )
            stopped = True
            break

    if processed:
        update_collector_status(
            phase="historical_gap_repair_complete",
            market=market.storage_coin,
            repair_candidates=len(candidates),
            repair_processed=processed,
            repair_written=written,
            repair_remaining=0 if not stopped else max(0, len(candidates) - processed),
        )
        log.info(
            "[%s] historical gap repair processed=%s wrote=%s remaining_candidates=%s",
            market.storage_coin,
            processed,
            written,
            0 if not stopped else max(0, len(candidates) - processed),
        )
    return processed, written, stopped


def archive_stats_from_local_raw(market: Market, day: date) -> dict[str, Any]:
    stats = empty_archive_stats()
    for hour in range(24):
        if raw_archive_l2_path(market, day, hour).exists():
            stats["hours_present"].append(hour)
        else:
            stats["hours_missing"].append(hour)
    stats["hours_present"] = sorted(set(stats["hours_present"]))
    stats["hours_missing"] = sorted(set(stats["hours_missing"]))
    stats["all_hours_available"] = len(stats["hours_present"]) == 24
    stats["is_cache_complete"] = True
    return stats


def quality_from_existing_array(coin: str, date_str: str, source: str, arr: np.ndarray) -> dict[str, Any]:
    arr = normalize_existing(arr)
    if arr is None:
        arr = np.empty((0, 6), dtype=np.float64)
    observed = {i for i in range(min(int(arr.shape[0]), CANDLES_PER_DAY)) if arr.shape[0] and not np.isnan(arr[i, 1])}
    gap = _read_json(gap_report_file(coin), {}).get(date_str, {})
    filled_count = int(gap.get("filled_minutes", gap.get("missing_raw_minutes_in_last_write", 0)) or 0) if isinstance(gap, dict) else 0
    filled_forward: set[int] = set()
    if filled_count > 0 and arr.shape[0] > 0:
        # Legacy reports did not store exact filled indexes; mark the count as an unknown tail range.
        start = max(0, min(int(arr.shape[0]), CANDLES_PER_DAY) - filled_count)
        filled_forward = set(range(start, min(int(arr.shape[0]), CANDLES_PER_DAY)))
        observed -= filled_forward
    return quality_summary(
        coin=coin,
        date_str=date_str,
        source=source,
        arr=arr,
        observed=observed,
        filled_forward=filled_forward,
        filled_backward=set(),
    )


def rebuild_quality_from_raw(con: Any | None, market: Market, date_str: str, arr: np.ndarray) -> dict[str, Any]:
    source = get_source(market.storage_coin, date_str)
    day = date_str_to_date(date_str)
    candles: list[list[float]] | None = None
    source_meta: dict[str, Any] = {}

    if source == "hydromancer_1s" and con is not None and raw_hydromancer_1s_path(market, date_str).exists():
        candles = hydromancer_1s_candles(con, market, date_str)
    elif source == "hydromancer_l2_midpoint_zero_volume" and con is not None and raw_hydromancer_l2_path(market, date_str).exists():
        candles = hydromancer_l2_candles(con, market, date_str)
    elif source == "hyperliquid_archive_l2_midpoint_zero_volume":
        stats = archive_stats_from_local_raw(market, day)
        source_meta = archive_stats_source_meta(stats)
        candles = official_archive_l2_candles_from_raw(market, day)

    if candles:
        rebuilt, quality = build_daily_array(
            candles,
            date_str,
            existing=None,
            existing_quality={},
            source=source,
            coin=market.storage_coin,
            source_meta=source_meta,
        )
        if REBUILD_FROM_RAW:
            tmp = day_file(market.storage_coin, date_str).parent / f"{date_str}.tmp.npy"
            np.save(str(tmp), rebuilt)
            tmp.replace(day_file(market.storage_coin, date_str))
        return quality

    return quality_from_existing_array(market.storage_coin, date_str, source, arr)


def repair_metadata_for_market(market: Market) -> dict[str, Any]:
    update_collector_status(phase="repair_metadata", market=market.storage_coin)
    con = None
    try:
        con = make_local_duckdb_connection()
    except Exception as exc:
        log.warning("[%s] metadata repair running without DuckDB raw parquet reads: %s", market.storage_coin, exc)

    repaired = 0
    partial = 0
    for fpath in sorted(coin_dir(market.storage_coin).glob("*.npy")):
        if ".tmp" in fpath.name:
            continue
        date_str = fpath.stem
        update_collector_status(phase="repair_metadata", market=market.storage_coin, current_date=date_str)
        try:
            arr = np.load(str(fpath), allow_pickle=False)
        except Exception as exc:
            log.warning("[%s] could not repair metadata for %s: %s", market.storage_coin, fpath.name, exc)
            continue
        quality = rebuild_quality_from_raw(con, market, date_str, arr)
        write_quality(market.storage_coin, date_str, quality)
        normalized = normalize_existing(arr)
        try:
            source_code_counts = rebuild_source_index_for_day(
                market.storage_coin,
                date_str,
                normalized if normalized is not None else arr,
                quality,
                str(quality.get("source") or get_source(market.storage_coin, date_str) or "unknown"),
            )
        except Exception as exc:
            source_code_counts = {}
            log.warning("[%s] could not repair source index for %s: %s", market.storage_coin, fpath.name, exc)
        update_gap_report(market.storage_coin, date_str, normalized if normalized is not None else arr, quality)
        entry = get_source_entry(market.storage_coin, date_str)
        entry.update(
            {
                "observed_minutes": int(quality.get("observed_minutes", 0)),
                "filled_minutes": int(quality.get("filled_minutes", 0)),
                "strict_backtest_ok": bool(quality.get("strict_backtest_ok", False)),
                "raw_missing_hours": quality.get("raw_missing_hours", []),
                "source_index_counts": source_code_counts,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        set_source_entry(market.storage_coin, date_str, entry)
        repaired += 1
        if not quality.get("strict_backtest_ok", False):
            partial += 1
    report = {"coin": market.storage_coin, "repaired_days": repaired, "partial_days": partial}
    update_collector_status(
        phase="repair_metadata_complete",
        market=market.storage_coin,
        metadata_repaired_days=repaired,
        metadata_partial_days=partial,
    )
    log.info("[%s] metadata repair complete: %s", market.storage_coin, report)
    return report


def verify_full_history_for_market(market: Market) -> dict[str, Any]:
    coin = market.storage_coin
    update_collector_status(phase="verify_full_history", market=coin)
    files = sorted(f for f in coin_dir(coin).glob("*.npy") if ".tmp" not in f.name)
    expected_start = expected_history_start_date()
    expected_end = historical_end_date()
    cursor_end = min(get_backfill_cursor(coin) - timedelta(days=1), expected_end)
    start_day = min((date_str_to_date(f.stem) for f in files), default=None)
    missing_files: list[str] = []
    array_issues: dict[str, list[str]] = {}
    missing_quality: list[str] = []
    partial_days: dict[str, dict[str, Any]] = {}
    source_index_issues_report: dict[str, list[str]] = {}
    source_index_summary: dict[str, dict[str, int]] = {}
    synthetic_source_days: dict[str, dict[str, int]] = {}
    unknown_source_days: dict[str, dict[str, int]] = {}

    if expected_end >= expected_start:
        for day in iter_days(expected_start, expected_end):
            date_str = date_to_str(day)
            fpath = day_file(coin, date_str)
            if not fpath.exists():
                missing_files.append(date_str)
                continue
            try:
                arr = np.load(str(fpath), allow_pickle=False)
                issues = completed_history_array_issues(date_str, arr)
                if issues:
                    array_issues[date_str] = issues
            except Exception as exc:
                array_issues[date_str] = [f"load error: {exc}"]
                continue
            quality = read_quality(coin, date_str)
            counts, index_issues = ensure_source_index_for_day(coin, date_str, arr, quality)
            if index_issues:
                source_index_issues_report[date_str] = index_issues
            if counts:
                source_index_summary[date_str] = counts
                synthetic_counts = {
                    key: int(counts.get(key, 0))
                    for key in (
                        "hydromancer_l2_midpoint_zero_volume",
                        "hyperliquid_archive_l2_midpoint_zero_volume",
                        "filled",
                    )
                    if int(counts.get(key, 0) or 0) > 0
                }
                if synthetic_counts:
                    synthetic_source_days[date_str] = synthetic_counts
                unknown_count = int(counts.get("unknown_legacy", 0) or 0)
                if unknown_count:
                    unknown_source_days[date_str] = {"unknown_legacy": unknown_count}
            if not quality:
                missing_quality.append(date_str)
            elif not quality.get("strict_backtest_ok", False):
                partial_days[date_str] = {
                    "source": quality.get("source"),
                    "filled_minutes": quality.get("filled_minutes", 0),
                    "raw_missing_hours": quality.get("raw_missing_hours", []),
                    "volume_is_real": quality.get("volume_is_real", False),
                }

    source_report = _read_json(source_report_file(coin), {})
    missing_source_days = sorted(
        k
        for k, v in source_report.items()
        if isinstance(v, dict)
        and v.get("source") == "missing"
        and date_to_str(expected_start) <= k <= date_to_str(expected_end)
    )
    cursor_lag_days = max(0, (expected_end - cursor_end).days)
    report = {
        "coin": coin,
        "start_day": date_to_str(start_day) if start_day else None,
        "expected_start_day": date_to_str(expected_start),
        "expected_end_day": date_to_str(expected_end),
        "cursor_completed_through": date_to_str(cursor_end),
        "cursor_lag_days": cursor_lag_days,
        "file_count": len(files),
        "missing_files": missing_files,
        "array_issues": array_issues,
        "missing_quality": missing_quality,
        "partial_days": partial_days,
        "source_index_issues": source_index_issues_report,
        "source_index_summary": source_index_summary,
        "synthetic_source_days": synthetic_source_days,
        "unknown_source_days": unknown_source_days,
        "provenance_ok": not source_index_issues_report,
        "missing_source_days": missing_source_days,
        "timestamp_continuous_ok": not missing_files and not array_issues,
        "strict_backtest_ok": (
            not missing_files
            and not array_issues
            and not missing_quality
            and not partial_days
            and not source_index_issues_report
            and not missing_source_days
        ),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(full_history_report_file(coin), report)
    update_collector_status(
        phase="verify_full_history_complete",
        market=coin,
        missing_files=len(missing_files),
        array_issue_days=len(array_issues),
        source_index_issue_days=len(source_index_issues_report),
        strict_backtest_ok=bool(report["strict_backtest_ok"]),
    )
    if report["timestamp_continuous_ok"]:
        log.info(
            "[%s] full-history timestamp continuity OK; partial_days=%s cursor_lag_days=%s",
            coin,
            len(partial_days),
            cursor_lag_days,
        )
    else:
        log.warning(
            "[%s] full-history verification found gaps/issues: missing=%s issues=%s cursor_lag_days=%s",
            coin,
            len(missing_files),
            len(array_issues),
            cursor_lag_days,
        )
    return report


def has_true_ohlcv(coin: str, date_str: str) -> bool:
    return day_file(coin, date_str).exists() and get_source(coin, date_str) in TRUE_VOLUME_SOURCES


def should_fetch_hydromancer_1s(market: Market, date_str: str) -> bool:
    if not ENABLE_HYDROMANCER_CANDLES:
        return False
    if FORCE:
        return True
    existing_source = get_source(market.storage_coin, date_str)
    if (
        existing_source == "hydromancer_1s"
        and
        not raw_hydromancer_1s_path(market, date_str).exists()
        and not raw_hydromancer_1s_missing_path(market, date_str).exists()
    ):
        return True
    return source_priority(existing_source) < source_priority("hydromancer_1s")


def backfill_one_day(
    con: Any | None,
    s3_archive: Any | None,
    market: Market,
    day: date,
    archive_budget: ArchiveHourBudget,
) -> tuple[bool, bool]:
    date_str = date_to_str(day)
    update_collector_status(phase="backfill_one_day", market=market.storage_coin, current_date=date_str)
    has_true = has_true_ohlcv(market.storage_coin, date_str) and not FORCE
    written = False

    if con is not None and should_fetch_hydromancer_1s(market, date_str):
        candles = hydromancer_1s_candles(con, market, date_str)
        if candles:
            if source_priority(get_source(market.storage_coin, date_str)) < source_priority("hydromancer_1s") or FORCE:
                written = save_day(market.storage_coin, date_str, candles, "hydromancer_1s")
            log.info("[%s] %s hydromancer 1s candles: %s rows", market.storage_coin, date_str, len(candles))
            has_true = True

    if CACHE_ALL_RAW_L2 and ENABLE_HYDROMANCER_L2 and con is not None:
        cache_hydromancer_l2_raw(con, market, date_str)

    if not has_true and not written and ENABLE_HYDROMANCER_L2 and con is not None:
        candles = hydromancer_l2_candles(con, market, date_str)
        if candles:
            written = save_day(market.storage_coin, date_str, candles, "hydromancer_l2_midpoint_zero_volume")
            log.info("[%s] %s hydromancer L2 midpoint: %s rows", market.storage_coin, date_str, len(candles))

    archive_complete = True
    archive_stats: dict[str, Any] | None = None
    if ENABLE_HYPERLIQUID_ARCHIVE_L2 and s3_archive is not None:
        if CACHE_ALL_RAW_L2 or (not has_true and not written):
            archive_stats = cache_official_archive_l2_raw_day(s3_archive, market, day, archive_budget)
            archive_complete = bool(archive_stats.get("is_cache_complete", False))
        if archive_complete and not has_true and not written:
            candles = official_archive_l2_candles_from_raw(market, day)
            if candles:
                written = save_day(
                    market.storage_coin,
                    date_str,
                    candles,
                    "hyperliquid_archive_l2_midpoint_zero_volume",
                    archive_stats_source_meta(archive_stats),
                )
                log.info("[%s] %s official archive L2 midpoint: %s rows",
                         market.storage_coin, date_str, len(candles))

    if not has_true and not written and archive_complete:
        mark_missing(market.storage_coin, date_str, "no configured source returned data")
    return written, archive_complete


def repair_recent_hydromancer(con: Any | None, market: Market, end_day: date) -> int:
    if con is None or RECENT_HYDROMANCER_REPAIR_DAYS <= 0:
        return 0
    start_day = max(
        date_str_to_date(EARLIEST_DATE),
        end_day - timedelta(days=RECENT_HYDROMANCER_REPAIR_DAYS - 1),
    )
    repaired = 0
    day = start_day
    while day <= end_day and not shutdown_event.is_set():
        date_str = date_to_str(day)
        if should_fetch_hydromancer_1s(market, date_str):
            candles = hydromancer_1s_candles(con, market, date_str)
            if candles:
                saved = False
                if source_priority(get_source(market.storage_coin, date_str)) < source_priority("hydromancer_1s") or FORCE:
                    saved = save_day(market.storage_coin, date_str, candles, "hydromancer_1s")
                if saved:
                    repaired += 1
                log.info("[%s] %s recent hydromancer 1s repair: %s rows",
                         market.storage_coin, date_str, len(candles))
        if CACHE_ALL_RAW_L2 and ENABLE_HYDROMANCER_L2:
            cache_hydromancer_l2_raw(con, market, date_str)
        day += timedelta(days=1)
    if repaired:
        log.info("[%s] recent hydromancer repair wrote=%s day(s)", market.storage_coin, repaired)
    return repaired


def historical_backfill(markets: list[Market]) -> None:
    if not (ENABLE_HYDROMANCER_CANDLES or ENABLE_HYDROMANCER_L2 or ENABLE_HYPERLIQUID_ARCHIVE_L2):
        return

    update_collector_status(phase="historical_backfill_start")
    con = None
    s3_archive = None
    if ENABLE_HYDROMANCER_CANDLES or ENABLE_HYDROMANCER_L2:
        try:
            con = make_duckdb_connection()
        except Exception as exc:
            log.error("DuckDB/S3 init failed; hydromancer sources disabled this cycle: %s", exc)
    if ENABLE_HYPERLIQUID_ARCHIVE_L2:
        try:
            s3_archive = make_s3_client(ARCHIVE_REGION)
        except Exception as exc:
            log.error("boto3 official archive init failed this cycle: %s", exc)

    archive_budget = ArchiveHourBudget(MAX_ARCHIVE_HOURS_PER_CYCLE)
    end_day = historical_end_date()
    for market in markets:
        update_collector_status(
            phase="historical_backfill",
            market=market.storage_coin,
            expected_end_day=date_to_str(end_day),
        )
        cleanup_tmp_files(market.storage_coin)
        repair_recent_hydromancer(con, market, end_day)

        processed = 0
        written = 0
        repaired, repaired_written, repair_stopped = repair_historical_gaps(
            con,
            s3_archive,
            market,
            end_day,
        )
        processed += repaired
        written += repaired_written
        if repair_stopped:
            update_collector_status(
                phase="historical_gap_repair_deferred",
                market=market.storage_coin,
                repair_processed=processed,
                repair_written=written,
                next_date=date_to_str(get_backfill_cursor(market.storage_coin)),
            )
            log.info(
                "[%s] historical cycle processed=%s wrote=%s next=%s",
                market.storage_coin,
                processed,
                written,
                date_to_str(get_backfill_cursor(market.storage_coin)),
            )
            continue

        cursor = get_backfill_cursor(market.storage_coin)
        if CACHE_ALL_RAW_L2 and RAW_L2_REFRESH_LOOKBACK_DAYS > 0:
            refresh = max(date_str_to_date(EARLIEST_DATE), end_day - timedelta(days=RAW_L2_REFRESH_LOOKBACK_DAYS))
            if cursor > refresh:
                cursor = refresh
        if cursor > end_day:
            log.info("[%s] historical backfill current through %s", market.storage_coin, date_to_str(end_day))
            update_collector_status(
                phase="historical_backfill_current",
                market=market.storage_coin,
                current_through=date_to_str(end_day),
            )
            continue

        day = cursor
        while day <= end_day and processed < MAX_HISTORICAL_DAYS_PER_CYCLE and not shutdown_event.is_set():
            if archive_budget.hours <= 0 and ENABLE_HYPERLIQUID_ARCHIVE_L2:
                log.info("official archive hour cap reached for this cycle")
                break
            update_collector_status(
                phase="historical_cursor_backfill",
                market=market.storage_coin,
                current_date=date_to_str(day),
                repair_processed=processed,
                repair_written=written,
            )
            day_written, day_complete = backfill_one_day(con, s3_archive, market, day, archive_budget)
            if day_written:
                written += 1
            if not day_complete:
                log.info("[%s] deferring %s until next cycle because raw cache is incomplete",
                         market.storage_coin, date_to_str(day))
                break
            processed += 1
            day += timedelta(days=1)
            set_backfill_cursor(market.storage_coin, day)
        log.info("[%s] historical cycle processed=%s wrote=%s next=%s",
                 market.storage_coin, processed, written, date_to_str(day))
        update_collector_status(
            phase="historical_backfill_complete",
            market=market.storage_coin,
            repair_processed=processed,
            repair_written=written,
            next_date=date_to_str(day),
        )


async def run_api_cycle(session: aiohttp.ClientSession, markets: list[Market]) -> None:
    if not ENABLE_API_TAIL:
        return
    await asyncio.gather(*[update_api_tail(session, market) for market in markets])


async def run() -> None:
    global _rate_limiter
    _rate_limiter = RateLimiter(MAX_CONCURRENT, REQUEST_INTERVAL)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    update_collector_status(
        phase="starting",
        symbols=SYMBOLS,
        data_dir=str(DATA_DIR.resolve()),
        earliest=EARLIEST_DATE,
    )
    log.info("Hyperliquid OHLCV Collector starting")
    log.info("  data dir: %s", DATA_DIR.resolve())
    log.info("  symbols: %s", SYMBOLS)
    log.info("  dex: %s", s3_dex())
    log.info("  earliest: %s", EARLIEST_DATE)
    log.info("  sources: hydromancer_1s=%s hydromancer_l2=%s official_l2=%s api_tail=%s",
             _bool_label(ENABLE_HYDROMANCER_CANDLES),
             _bool_label(ENABLE_HYDROMANCER_L2),
             _bool_label(ENABLE_HYPERLIQUID_ARCHIVE_L2),
             _bool_label(ENABLE_API_TAIL))
    log.info("  raw L2 cache all: %s; refresh lookback: %s days",
             _bool_label(CACHE_ALL_RAW_L2), RAW_L2_REFRESH_LOOKBACK_DAYS)
    log.info("  recent hydromancer repair: %s days", RECENT_HYDROMANCER_REPAIR_DAYS)
    log.info("  modes: repair_metadata=%s rebuild_from_raw=%s verify_full_history=%s",
             _bool_label(REPAIR_METADATA_ONLY), _bool_label(REBUILD_FROM_RAW), _bool_label(VERIFY_FULL_HISTORY))
    log.info("  caps: historical_days=%s official_archive_download_hours=%s poll=%ss",
             MAX_HISTORICAL_DAYS_PER_CYCLE, MAX_ARCHIVE_HOURS_PER_CYCLE, POLL_INTERVAL)

    if REPAIR_METADATA_ONLY or VERIFY_FULL_HISTORY:
        markets = markets_from_env()
        if REPAIR_METADATA_ONLY:
            for market in markets:
                await asyncio.to_thread(repair_metadata_for_market, market)
        if VERIFY_FULL_HISTORY:
            for market in markets:
                await asyncio.to_thread(verify_full_history_for_market, market)
        log.info("Collector stopped after maintenance mode")
        update_collector_status(phase="stopped_after_maintenance")
        return

    async with aiohttp.ClientSession(headers={"Content-Type": "application/json"}) as session:
        markets = await discover_markets(session)
        if not markets:
            update_collector_status(phase="stopped_no_markets")
            return
        while not shutdown_event.is_set():
            await asyncio.to_thread(historical_backfill, markets)
            await run_api_cycle(session, markets)
            for market in markets:
                verify_recent_files(market.storage_coin, VERIFY_DAYS)
            if RUN_ONCE or shutdown_event.is_set():
                break
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=POLL_INTERVAL)
            except asyncio.TimeoutError:
                pass

    log.info("Collector stopped")
    update_collector_status(phase="stopped")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
