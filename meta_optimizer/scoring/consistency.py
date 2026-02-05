"""
Consistency metrics for robustness scoring.

Measures how stable a strategy's performance is across different folds/scenarios.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ConsistencyMetrics:
    """Metrics measuring performance consistency across folds."""
    # Coefficient of variation (std/mean) - lower is better
    cv_adg: float
    cv_sharpe: float

    # Percentage of profitable folds - higher is better
    profitable_fold_pct: float

    # Inter-quartile range normalized by median - lower is better
    iqr_normalized_adg: float

    # Standard deviation of key metrics
    std_adg: float
    std_sharpe: float

    # Range (max - min) of metrics
    range_adg: float
    range_sharpe: float

    # Overall consistency score (0-1, higher is better)
    consistency_score: float

    def to_dict(self) -> dict:
        return {
            "cv_adg": self.cv_adg,
            "cv_sharpe": self.cv_sharpe,
            "profitable_fold_pct": self.profitable_fold_pct,
            "iqr_normalized_adg": self.iqr_normalized_adg,
            "std_adg": self.std_adg,
            "std_sharpe": self.std_sharpe,
            "range_adg": self.range_adg,
            "range_sharpe": self.range_sharpe,
            "consistency_score": self.consistency_score,
        }


def compute_consistency_metrics(
    fold_results: List[Dict[str, Any]],
    adg_key: str = "adg_pnl",
    sharpe_key: str = "sharpe_ratio_pnl",
) -> ConsistencyMetrics:
    """
    Compute consistency metrics from validation results across folds.

    Args:
        fold_results: List of metrics dictionaries, one per fold
        adg_key: Key for average daily gain metric
        sharpe_key: Key for Sharpe ratio metric

    Returns:
        ConsistencyMetrics dataclass
    """
    if not fold_results:
        return ConsistencyMetrics(
            cv_adg=float("inf"),
            cv_sharpe=float("inf"),
            profitable_fold_pct=0.0,
            iqr_normalized_adg=float("inf"),
            std_adg=float("inf"),
            std_sharpe=float("inf"),
            range_adg=float("inf"),
            range_sharpe=float("inf"),
            consistency_score=0.0,
        )

    # Extract values (handling different key patterns)
    adg_values = []
    sharpe_values = []

    for result in fold_results:
        # Try different key patterns
        adg = _get_metric(result, adg_key)
        sharpe = _get_metric(result, sharpe_key)

        if adg is not None:
            adg_values.append(adg)
        if sharpe is not None:
            sharpe_values.append(sharpe)

    if not adg_values:
        adg_values = [0.0]
    if not sharpe_values:
        sharpe_values = [0.0]

    adg_arr = np.array(adg_values)
    sharpe_arr = np.array(sharpe_values)

    # Coefficient of variation
    cv_adg = _safe_cv(adg_arr)
    cv_sharpe = _safe_cv(sharpe_arr)

    # Profitable fold percentage
    profitable_fold_pct = np.mean(adg_arr > 0)

    # IQR normalized
    iqr_normalized_adg = _safe_iqr_normalized(adg_arr)

    # Standard deviations
    std_adg = np.std(adg_arr) if len(adg_arr) > 1 else 0.0
    std_sharpe = np.std(sharpe_arr) if len(sharpe_arr) > 1 else 0.0

    # Ranges
    range_adg = np.max(adg_arr) - np.min(adg_arr) if len(adg_arr) > 1 else 0.0
    range_sharpe = np.max(sharpe_arr) - np.min(sharpe_arr) if len(sharpe_arr) > 1 else 0.0

    # Compute overall consistency score
    consistency_score = _compute_consistency_score(
        cv_adg=cv_adg,
        cv_sharpe=cv_sharpe,
        profitable_fold_pct=profitable_fold_pct,
        iqr_normalized_adg=iqr_normalized_adg,
    )

    return ConsistencyMetrics(
        cv_adg=cv_adg,
        cv_sharpe=cv_sharpe,
        profitable_fold_pct=profitable_fold_pct,
        iqr_normalized_adg=iqr_normalized_adg,
        std_adg=std_adg,
        std_sharpe=std_sharpe,
        range_adg=range_adg,
        range_sharpe=range_sharpe,
        consistency_score=consistency_score,
    )


def _get_metric(result: Dict[str, Any], key: str) -> Optional[float]:
    """Get metric value, trying various key patterns."""
    # Direct key
    if key in result:
        val = result[key]
        if isinstance(val, (int, float)) and not np.isnan(val):
            return float(val)

    # With _mean suffix
    mean_key = f"{key}_mean"
    if mean_key in result:
        val = result[mean_key]
        if isinstance(val, (int, float)) and not np.isnan(val):
            return float(val)

    # Nested in metrics
    if "metrics" in result:
        return _get_metric(result["metrics"], key)

    return None


def _safe_cv(arr: np.ndarray) -> float:
    """Compute coefficient of variation safely."""
    if len(arr) < 2:
        return 0.0
    mean = np.mean(arr)
    if abs(mean) < 1e-10:
        return float("inf") if np.std(arr) > 1e-10 else 0.0
    return abs(np.std(arr) / mean)


def _safe_iqr_normalized(arr: np.ndarray) -> float:
    """Compute IQR normalized by median safely."""
    if len(arr) < 4:
        return 0.0
    q75, q25 = np.percentile(arr, [75, 25])
    median = np.median(arr)
    if abs(median) < 1e-10:
        return float("inf") if (q75 - q25) > 1e-10 else 0.0
    return abs((q75 - q25) / median)


def _compute_consistency_score(
    cv_adg: float,
    cv_sharpe: float,
    profitable_fold_pct: float,
    iqr_normalized_adg: float,
) -> float:
    """
    Compute overall consistency score (0-1, higher is better).

    Components:
    - Low CV is good (transformed: 1 / (1 + cv))
    - High profitable % is good (direct)
    - Low IQR is good (transformed: 1 / (1 + iqr))
    """
    # Transform CV values (lower is better -> higher score)
    cv_score = 0.5 * (1 / (1 + cv_adg)) + 0.5 * (1 / (1 + cv_sharpe))

    # IQR score
    iqr_score = 1 / (1 + iqr_normalized_adg)

    # Combine scores
    # Weights: profitable_pct 40%, CV 35%, IQR 25%
    score = (
        0.40 * profitable_fold_pct +
        0.35 * cv_score +
        0.25 * iqr_score
    )

    return float(np.clip(score, 0, 1))


def compute_fold_agreement(
    fold_results: List[Dict[str, Any]],
    metric_key: str = "adg_pnl",
    agreement_threshold: float = 0.5,
) -> float:
    """
    Compute the agreement rate between fold pairs.

    Measures how often different folds agree on profitability direction.

    Args:
        fold_results: List of metrics per fold
        metric_key: Metric to compare
        agreement_threshold: Min correlation for agreement

    Returns:
        Agreement rate (0-1)
    """
    if len(fold_results) < 2:
        return 1.0  # Single fold always agrees with itself

    values = []
    for result in fold_results:
        val = _get_metric(result, metric_key)
        if val is not None:
            values.append(val)

    if len(values) < 2:
        return 1.0

    # Count pairs that agree on sign (both positive or both negative)
    n_pairs = 0
    n_agree = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            n_pairs += 1
            if (values[i] > 0) == (values[j] > 0):
                n_agree += 1

    return n_agree / n_pairs if n_pairs > 0 else 1.0
