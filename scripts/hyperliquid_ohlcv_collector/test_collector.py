"""Tests for OHLCV candle-integrity validation (collector.validate_daily_array).

Regression tests for archive anomalies found 2026-06: two candles with a
``high`` slightly below open/close, and a ``low = 0`` on 2026-01-08. A zero low
simulates a total crash and would invalidate a backtest, yet it is NOT caught by
the classic OHLC invariant (0 <= min(open, close) and high >= 0), so it needs an
explicit non-positive-price check.

Run inside the collector image (which has numpy), mounting this dir over /app:

    docker compose run --rm --no-deps -v "<dir>://app" \
        --entrypoint python collector test_collector.py

Also collectible by pytest if pytest + numpy are available.
"""
import numpy as np

import collector

DAY = "2026-01-08"  # the date of the real low=0 anomaly


def make_valid_day(date_str: str = DAY, price: float = 100.0) -> np.ndarray:
    """A full, valid UTC day: 1440 contiguous 1m candles, flat positive price."""
    start = collector.date_str_to_start_ms(date_str)
    n = collector.CANDLES_PER_DAY
    arr = np.empty((n, 6), dtype=np.float64)
    arr[:, 0] = start + np.arange(n, dtype=np.int64) * collector.MS_PER_MIN
    arr[:, 1:5] = price  # open, high, low, close
    arr[:, 5] = 1.0  # volume
    return arr


def test_clean_day_has_no_issues():
    assert collector.validate_daily_array(DAY, make_valid_day()) == []


def test_high_below_open_close_is_flagged():
    arr = make_valid_day()
    arr[10, 2] = 99.0  # high below open/close
    arr[1000, 2] = 99.5  # a second high anomaly
    issues = collector.validate_daily_array(DAY, arr)
    assert "OHLC invariant violation" in issues, issues


def test_low_zero_is_flagged():
    # The real 2026-01-08 anomaly: low = 0 => fake total crash.
    arr = make_valid_day()
    arr[123, 3] = 0.0
    issues = collector.validate_daily_array(DAY, arr)
    assert "non-positive price" in issues, issues


def test_negative_low_is_flagged():
    arr = make_valid_day()
    arr[7, 3] = -5.0
    issues = collector.validate_daily_array(DAY, arr)
    assert "non-positive price" in issues, issues


def test_ohlc_integrity_issues_detects_each_anomaly():
    clean = make_valid_day()
    assert collector.ohlc_integrity_issues(clean) == []

    high_bad = make_valid_day()
    high_bad[3, 2] = 99.0
    assert "OHLC invariant violation" in collector.ohlc_integrity_issues(high_bad)

    low_zero = make_valid_day()
    low_zero[3, 3] = 0.0
    assert "non-positive price" in collector.ohlc_integrity_issues(low_zero)


def test_quality_summary_clean_day_is_backtest_ready():
    arr = make_valid_day()
    q = collector.quality_summary(
        "HYPE", DAY, "hydromancer_1s", arr,
        observed=set(range(collector.CANDLES_PER_DAY)),
        filled_forward=set(), filled_backward=set(),
    )
    assert q["strict_backtest_ok"] is True, q


def test_quality_summary_low_zero_not_backtest_ready():
    arr = make_valid_day()
    arr[123, 3] = 0.0  # low = 0 => fake total crash
    q = collector.quality_summary(
        "HYPE", DAY, "hydromancer_1s", arr,
        observed=set(range(collector.CANDLES_PER_DAY)),
        filled_forward=set(), filled_backward=set(),
    )
    assert q["strict_backtest_ok"] is False, "low=0 must not be backtest-ready"


def test_sanitize_ohlc_fixes_low_zero():
    arr = make_valid_day()  # flat at 100
    arr[5, 3] = 0.0  # low = 0
    fixed, n = collector.sanitize_ohlc(arr)
    assert n >= 1
    assert collector.ohlc_integrity_issues(fixed) == []
    assert float(fixed[:, 3].min()) > 0  # no zero low survives
    assert float(fixed[5, 3]) == 100.0  # clamped to min(open, close)


def test_sanitize_ohlc_fixes_high_below_open_close():
    arr = make_valid_day()
    arr[5, 2] = 99.0  # high below open/close
    fixed, n = collector.sanitize_ohlc(arr)
    assert n >= 1
    assert collector.ohlc_integrity_issues(fixed) == []
    assert float(fixed[5, 2]) == 100.0  # clamped up to max(open, close)


def test_sanitize_ohlc_leaves_clean_day_untouched():
    arr = make_valid_day()
    fixed, n = collector.sanitize_ohlc(arr)
    assert n == 0
    assert np.array_equal(fixed, arr)


def test_save_day_sanitizes_and_flags_low_zero():
    import shutil

    coin = "ZZTEST_INTEGRITY"
    start = collector.date_str_to_start_ms(DAY)
    candles = [
        [start + i * collector.MS_PER_MIN, 100.0, 100.0, 100.0, 100.0, 1.0]
        for i in range(collector.CANDLES_PER_DAY)
    ]
    candles[50][3] = 0.0  # low = 0
    try:
        wrote = collector.save_day(coin, DAY, candles, "hydromancer_1s")
        assert wrote is True
        arr = np.load(str(collector.day_file(coin, DAY)), allow_pickle=False)
        assert collector.validate_daily_array(DAY, arr) == [], "saved shard must be OHLC-clean"
        assert float(arr[:, 3].min()) > 0, "low=0 must be sanitized away"
        q = collector.read_quality(coin, DAY)
        assert q.get("strict_backtest_ok") is False, "sanitized day must not be backtest-ready"
    finally:
        shutil.rmtree(collector.coin_dir(coin), ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 - test runner reports any failure
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
