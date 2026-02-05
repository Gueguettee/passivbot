"""
Robustness scorer - combines consistency and degradation metrics.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .consistency import compute_consistency_metrics, ConsistencyMetrics
from .degradation import (
    compute_degradation_metrics,
    compute_aggregate_degradation,
    DegradationMetrics,
)


@dataclass
class RobustnessScore:
    """Complete robustness assessment for a strategy."""
    # Component scores (0-1, higher is better)
    consistency_score: float
    degradation_score: float
    worst_case_score: float
    stability_score: float

    # Overall robustness score
    overall_score: float

    # Component details
    consistency_metrics: ConsistencyMetrics
    degradation_metrics: DegradationMetrics

    # Worst fold metrics
    worst_fold_adg: float
    worst_fold_sharpe: float
    worst_fold_id: int

    # Mean performance across folds
    mean_adg: float
    mean_sharpe: float

    # Flags
    passes_thresholds: bool
    failure_reasons: List[str]

    def to_dict(self) -> dict:
        return {
            "consistency_score": self.consistency_score,
            "degradation_score": self.degradation_score,
            "worst_case_score": self.worst_case_score,
            "stability_score": self.stability_score,
            "overall_score": self.overall_score,
            "consistency_metrics": self.consistency_metrics.to_dict(),
            "degradation_metrics": self.degradation_metrics.to_dict(),
            "worst_fold_adg": self.worst_fold_adg,
            "worst_fold_sharpe": self.worst_fold_sharpe,
            "worst_fold_id": self.worst_fold_id,
            "mean_adg": self.mean_adg,
            "mean_sharpe": self.mean_sharpe,
            "passes_thresholds": self.passes_thresholds,
            "failure_reasons": self.failure_reasons,
        }


class RobustnessScorer:
    """
    Score strategies on robustness across validation folds.

    Robustness Score = weighted combination of:
    - Consistency (40%): Low variance across folds
    - Degradation (30%): Train vs validation performance ratio
    - Worst-case (20%): Performance in worst fold
    - Stability (10%): Parameter sensitivity (placeholder)
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize scorer with weights and thresholds.

        Args:
            weights: Component weights (must sum to 1.0)
            thresholds: Minimum requirements for passing
        """
        self.weights = weights or {
            "consistency": 0.4,
            "degradation": 0.3,
            "worst_case": 0.2,
            "stability": 0.1,
        }

        self.thresholds = thresholds or {
            "min_profitable_folds_pct": 0.8,
            "max_degradation_ratio": 0.5,
            "max_cv_adg": 1.0,
            "min_worst_fold_adg": 0.0,
        }

        # Validate weights
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    def score(
        self,
        validation_results: List[Dict[str, Any]],
        training_results: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> RobustnessScore:
        """
        Compute robustness score for a strategy.

        Args:
            validation_results: List of metrics from validation backtests (one per fold)
            training_results: Optional list of training metrics (for degradation calc)
            config: Optional bot config (for stability scoring)

        Returns:
            RobustnessScore with all metrics
        """
        if not validation_results:
            return self._empty_score()

        # Compute consistency metrics
        consistency_metrics = compute_consistency_metrics(validation_results)

        # Compute degradation metrics
        if training_results and len(training_results) == len(validation_results):
            train_val_pairs = list(zip(training_results, validation_results))
            degradation_metrics = compute_aggregate_degradation(train_val_pairs)
        else:
            # Without training data, use placeholder
            degradation_metrics = DegradationMetrics(
                adg_ratio=1.0,
                sharpe_ratio=1.0,
                drawdown_ratio=1.0,
                adg_degradation=0.0,
                sharpe_degradation=0.0,
                adg_degradation_pct=0.0,
                sharpe_degradation_pct=0.0,
                profitability_flip=False,
                degradation_score=1.0,
            )

        # Compute worst-case metrics
        adg_values = [self._get_adg(r) for r in validation_results]
        sharpe_values = [self._get_sharpe(r) for r in validation_results]

        worst_fold_idx = int(np.argmin(adg_values))
        worst_fold_adg = adg_values[worst_fold_idx]
        worst_fold_sharpe = sharpe_values[worst_fold_idx]

        mean_adg = np.mean(adg_values)
        mean_sharpe = np.mean(sharpe_values)

        # Compute component scores
        consistency_score = consistency_metrics.consistency_score
        degradation_score = degradation_metrics.degradation_score
        worst_case_score = self._compute_worst_case_score(worst_fold_adg, worst_fold_sharpe)
        stability_score = self._compute_stability_score(config) if config else 0.5

        # Compute overall score
        overall_score = (
            self.weights["consistency"] * consistency_score +
            self.weights["degradation"] * degradation_score +
            self.weights["worst_case"] * worst_case_score +
            self.weights["stability"] * stability_score
        )

        # Check thresholds
        passes_thresholds, failure_reasons = self._check_thresholds(
            consistency_metrics=consistency_metrics,
            degradation_metrics=degradation_metrics,
            worst_fold_adg=worst_fold_adg,
        )

        return RobustnessScore(
            consistency_score=consistency_score,
            degradation_score=degradation_score,
            worst_case_score=worst_case_score,
            stability_score=stability_score,
            overall_score=overall_score,
            consistency_metrics=consistency_metrics,
            degradation_metrics=degradation_metrics,
            worst_fold_adg=worst_fold_adg,
            worst_fold_sharpe=worst_fold_sharpe,
            worst_fold_id=worst_fold_idx,
            mean_adg=mean_adg,
            mean_sharpe=mean_sharpe,
            passes_thresholds=passes_thresholds,
            failure_reasons=failure_reasons,
        )

    def _get_adg(self, result: Dict[str, Any]) -> float:
        """Extract ADG from result dict."""
        for key in ["adg_pnl", "adg_pnl_mean", "adg"]:
            if key in result:
                val = result[key]
                if isinstance(val, (int, float)) and not np.isnan(val):
                    return float(val)
        if "metrics" in result:
            return self._get_adg(result["metrics"])
        return 0.0

    def _get_sharpe(self, result: Dict[str, Any]) -> float:
        """Extract Sharpe ratio from result dict."""
        for key in ["sharpe_ratio_pnl", "sharpe_ratio_pnl_mean", "sharpe_ratio"]:
            if key in result:
                val = result[key]
                if isinstance(val, (int, float)) and not np.isnan(val):
                    return float(val)
        if "metrics" in result:
            return self._get_sharpe(result["metrics"])
        return 0.0

    def _compute_worst_case_score(
        self,
        worst_adg: float,
        worst_sharpe: float,
    ) -> float:
        """
        Compute worst-case score (0-1).

        Higher score = better worst-case performance.
        """
        # ADG component: 0 if negative, scales up to 1 for ADG >= 0.002 (0.2%/day)
        if worst_adg <= 0:
            adg_score = 0.0
        else:
            adg_score = min(1.0, worst_adg / 0.002)

        # Sharpe component: 0 if negative, scales up to 1 for Sharpe >= 1.5
        if worst_sharpe <= 0:
            sharpe_score = 0.0
        else:
            sharpe_score = min(1.0, worst_sharpe / 1.5)

        return 0.6 * adg_score + 0.4 * sharpe_score

    def _compute_stability_score(self, config: Dict[str, Any]) -> float:
        """
        Compute stability score based on parameter sensitivity.

        This is a placeholder - full implementation would perturb parameters
        and measure how much performance changes.
        """
        # For now, return neutral score
        return 0.5

    def _check_thresholds(
        self,
        consistency_metrics: ConsistencyMetrics,
        degradation_metrics: DegradationMetrics,
        worst_fold_adg: float,
    ) -> tuple[bool, List[str]]:
        """Check if strategy passes all thresholds."""
        failure_reasons = []

        # Check profitable folds percentage
        if consistency_metrics.profitable_fold_pct < self.thresholds["min_profitable_folds_pct"]:
            failure_reasons.append(
                f"Profitable folds ({consistency_metrics.profitable_fold_pct:.1%}) "
                f"< minimum ({self.thresholds['min_profitable_folds_pct']:.1%})"
            )

        # Check degradation ratio
        if degradation_metrics.adg_ratio < self.thresholds["max_degradation_ratio"]:
            failure_reasons.append(
                f"Degradation ratio ({degradation_metrics.adg_ratio:.2f}) "
                f"< maximum allowed ({self.thresholds['max_degradation_ratio']:.2f})"
            )

        # Check CV of ADG
        if consistency_metrics.cv_adg > self.thresholds["max_cv_adg"]:
            failure_reasons.append(
                f"CV of ADG ({consistency_metrics.cv_adg:.2f}) "
                f"> maximum allowed ({self.thresholds['max_cv_adg']:.2f})"
            )

        # Check worst fold ADG
        if worst_fold_adg < self.thresholds["min_worst_fold_adg"]:
            failure_reasons.append(
                f"Worst fold ADG ({worst_fold_adg:.6f}) "
                f"< minimum ({self.thresholds['min_worst_fold_adg']:.6f})"
            )

        return len(failure_reasons) == 0, failure_reasons

    def _empty_score(self) -> RobustnessScore:
        """Return empty score for invalid input."""
        return RobustnessScore(
            consistency_score=0.0,
            degradation_score=0.0,
            worst_case_score=0.0,
            stability_score=0.0,
            overall_score=0.0,
            consistency_metrics=ConsistencyMetrics(
                cv_adg=float("inf"),
                cv_sharpe=float("inf"),
                profitable_fold_pct=0.0,
                iqr_normalized_adg=float("inf"),
                std_adg=0.0,
                std_sharpe=0.0,
                range_adg=0.0,
                range_sharpe=0.0,
                consistency_score=0.0,
            ),
            degradation_metrics=DegradationMetrics(
                adg_ratio=0.0,
                sharpe_ratio=0.0,
                drawdown_ratio=1.0,
                adg_degradation=0.0,
                sharpe_degradation=0.0,
                adg_degradation_pct=0.0,
                sharpe_degradation_pct=0.0,
                profitability_flip=True,
                degradation_score=0.0,
            ),
            worst_fold_adg=0.0,
            worst_fold_sharpe=0.0,
            worst_fold_id=0,
            mean_adg=0.0,
            mean_sharpe=0.0,
            passes_thresholds=False,
            failure_reasons=["No validation results"],
        )
