"""
Degradation metrics for detecting overfitting.

Compares in-sample (training) vs out-of-sample (validation) performance.
Large degradation indicates overfitting.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class DegradationMetrics:
    """Metrics measuring performance degradation from train to validation."""
    # Ratio of validation to training performance (closer to 1.0 is better)
    adg_ratio: float
    sharpe_ratio: float
    drawdown_ratio: float  # val_dd / train_dd (higher is worse)

    # Absolute degradation
    adg_degradation: float  # train_adg - val_adg
    sharpe_degradation: float

    # Percentage degradation
    adg_degradation_pct: float  # (train - val) / train * 100
    sharpe_degradation_pct: float

    # Whether strategy went from profitable to unprofitable
    profitability_flip: bool

    # Overall degradation score (0-1, higher is better = less degradation)
    degradation_score: float

    def to_dict(self) -> dict:
        return {
            "adg_ratio": self.adg_ratio,
            "sharpe_ratio": self.sharpe_ratio,
            "drawdown_ratio": self.drawdown_ratio,
            "adg_degradation": self.adg_degradation,
            "sharpe_degradation": self.sharpe_degradation,
            "adg_degradation_pct": self.adg_degradation_pct,
            "sharpe_degradation_pct": self.sharpe_degradation_pct,
            "profitability_flip": self.profitability_flip,
            "degradation_score": self.degradation_score,
        }


def compute_degradation_metrics(
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    adg_key: str = "adg_pnl",
    sharpe_key: str = "sharpe_ratio_pnl",
    drawdown_key: str = "drawdown_worst_usd",
) -> DegradationMetrics:
    """
    Compute degradation metrics comparing training and validation performance.

    Args:
        train_metrics: Metrics from training (in-sample) backtest
        val_metrics: Metrics from validation (out-of-sample) backtest
        adg_key: Key for ADG metric
        sharpe_key: Key for Sharpe ratio metric
        drawdown_key: Key for drawdown metric

    Returns:
        DegradationMetrics dataclass
    """
    # Extract values
    train_adg = _get_metric(train_metrics, adg_key, 0.0)
    val_adg = _get_metric(val_metrics, adg_key, 0.0)

    train_sharpe = _get_metric(train_metrics, sharpe_key, 0.0)
    val_sharpe = _get_metric(val_metrics, sharpe_key, 0.0)

    train_dd = _get_metric(train_metrics, drawdown_key, 1.0)
    val_dd = _get_metric(val_metrics, drawdown_key, 1.0)

    # Compute ratios (validation / training)
    adg_ratio = _safe_ratio(val_adg, train_adg)
    sharpe_ratio_val = _safe_ratio(val_sharpe, train_sharpe)
    drawdown_ratio = _safe_ratio(val_dd, train_dd)  # Higher means worse DD in validation

    # Absolute degradation
    adg_degradation = train_adg - val_adg
    sharpe_degradation = train_sharpe - val_sharpe

    # Percentage degradation
    adg_degradation_pct = _safe_pct_degradation(train_adg, val_adg)
    sharpe_degradation_pct = _safe_pct_degradation(train_sharpe, val_sharpe)

    # Profitability flip
    profitability_flip = (train_adg > 0) and (val_adg <= 0)

    # Compute overall degradation score
    degradation_score = _compute_degradation_score(
        adg_ratio=adg_ratio,
        sharpe_ratio=sharpe_ratio_val,
        profitability_flip=profitability_flip,
    )

    return DegradationMetrics(
        adg_ratio=adg_ratio,
        sharpe_ratio=sharpe_ratio_val,
        drawdown_ratio=drawdown_ratio,
        adg_degradation=adg_degradation,
        sharpe_degradation=sharpe_degradation,
        adg_degradation_pct=adg_degradation_pct,
        sharpe_degradation_pct=sharpe_degradation_pct,
        profitability_flip=profitability_flip,
        degradation_score=degradation_score,
    )


def compute_aggregate_degradation(
    train_val_pairs: List[tuple[Dict[str, Any], Dict[str, Any]]],
    adg_key: str = "adg_pnl",
    sharpe_key: str = "sharpe_ratio_pnl",
) -> DegradationMetrics:
    """
    Compute aggregate degradation across multiple train/val pairs.

    Args:
        train_val_pairs: List of (train_metrics, val_metrics) tuples
        adg_key: Key for ADG metric
        sharpe_key: Key for Sharpe ratio metric

    Returns:
        Aggregated DegradationMetrics
    """
    if not train_val_pairs:
        return DegradationMetrics(
            adg_ratio=0.0,
            sharpe_ratio=0.0,
            drawdown_ratio=1.0,
            adg_degradation=0.0,
            sharpe_degradation=0.0,
            adg_degradation_pct=0.0,
            sharpe_degradation_pct=0.0,
            profitability_flip=True,
            degradation_score=0.0,
        )

    # Compute individual degradation metrics
    individual_metrics = [
        compute_degradation_metrics(train, val, adg_key, sharpe_key)
        for train, val in train_val_pairs
    ]

    # Aggregate by averaging
    adg_ratios = [m.adg_ratio for m in individual_metrics]
    sharpe_ratios = [m.sharpe_ratio for m in individual_metrics]
    drawdown_ratios = [m.drawdown_ratio for m in individual_metrics]
    adg_degradations = [m.adg_degradation for m in individual_metrics]
    sharpe_degradations = [m.sharpe_degradation for m in individual_metrics]
    adg_degradation_pcts = [m.adg_degradation_pct for m in individual_metrics]
    sharpe_degradation_pcts = [m.sharpe_degradation_pct for m in individual_metrics]
    profitability_flips = [m.profitability_flip for m in individual_metrics]
    degradation_scores = [m.degradation_score for m in individual_metrics]

    return DegradationMetrics(
        adg_ratio=np.mean(adg_ratios),
        sharpe_ratio=np.mean(sharpe_ratios),
        drawdown_ratio=np.mean(drawdown_ratios),
        adg_degradation=np.mean(adg_degradations),
        sharpe_degradation=np.mean(sharpe_degradations),
        adg_degradation_pct=np.mean(adg_degradation_pcts),
        sharpe_degradation_pct=np.mean(sharpe_degradation_pcts),
        profitability_flip=any(profitability_flips),  # True if any fold flipped
        degradation_score=np.mean(degradation_scores),
    )


def _get_metric(metrics: Dict[str, Any], key: str, default: float) -> float:
    """Get metric value with fallback patterns."""
    # Direct key
    if key in metrics:
        val = metrics[key]
        if isinstance(val, (int, float)) and not np.isnan(val):
            return float(val)

    # With _mean suffix
    mean_key = f"{key}_mean"
    if mean_key in metrics:
        val = metrics[mean_key]
        if isinstance(val, (int, float)) and not np.isnan(val):
            return float(val)

    # Nested in metrics
    if "metrics" in metrics:
        return _get_metric(metrics["metrics"], key, default)

    return default


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Compute ratio safely, handling edge cases."""
    if abs(denominator) < 1e-10:
        if abs(numerator) < 1e-10:
            return 1.0  # Both near zero, no degradation
        return 0.0 if numerator < 0 else float("inf")

    ratio = numerator / denominator

    # Handle negative to positive transitions
    if denominator > 0 and numerator < 0:
        return numerator / denominator  # Will be negative, indicating flip

    return ratio


def _safe_pct_degradation(train_val: float, val_val: float) -> float:
    """Compute percentage degradation safely."""
    if abs(train_val) < 1e-10:
        return 0.0 if abs(val_val) < 1e-10 else -100.0 if val_val > 0 else 100.0

    return ((train_val - val_val) / abs(train_val)) * 100


def _compute_degradation_score(
    adg_ratio: float,
    sharpe_ratio: float,
    profitability_flip: bool,
) -> float:
    """
    Compute overall degradation score (0-1, higher is better).

    A ratio of 1.0 means no degradation (best).
    Ratios < 1.0 mean validation is worse than training.
    Ratios > 1.0 mean validation is better (possible but rare).
    """
    # Heavy penalty for profitability flip
    if profitability_flip:
        return 0.0

    # Score based on how close ratios are to 1.0
    # Score = 1 for ratio = 1, decreasing as ratio deviates

    def ratio_to_score(ratio: float) -> float:
        if ratio <= 0:
            return 0.0
        if ratio > 2:
            return 1.0  # Cap at 2x improvement
        if ratio >= 1:
            return 1.0  # No penalty for improvement
        # Linear penalty for degradation
        return ratio

    adg_score = ratio_to_score(adg_ratio)
    sharpe_score = ratio_to_score(sharpe_ratio)

    # Weighted average
    return 0.6 * adg_score + 0.4 * sharpe_score


def is_likely_overfit(
    degradation: DegradationMetrics,
    max_degradation_ratio: float = 0.5,
) -> bool:
    """
    Determine if metrics suggest overfitting.

    Args:
        degradation: DegradationMetrics from compute_degradation_metrics
        max_degradation_ratio: Maximum allowed val/train ratio

    Returns:
        True if likely overfit
    """
    # Definite overfit if profitable strategy becomes unprofitable
    if degradation.profitability_flip:
        return True

    # Overfit if performance drops more than threshold
    if degradation.adg_ratio < max_degradation_ratio:
        return True

    if degradation.sharpe_ratio < max_degradation_ratio:
        return True

    return False
