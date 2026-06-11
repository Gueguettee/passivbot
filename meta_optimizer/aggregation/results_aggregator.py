"""
Results aggregation utilities.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple folds."""
    # Mean values
    mean_adg: float
    mean_sharpe: float
    mean_drawdown: float
    mean_loss_profit_ratio: float

    # Standard deviations
    std_adg: float
    std_sharpe: float

    # Quantiles
    median_adg: float
    q25_adg: float
    q75_adg: float

    # Extremes
    min_adg: float
    max_adg: float
    min_sharpe: float
    max_sharpe: float

    # Counts
    n_folds: int
    n_profitable_folds: int

    def to_dict(self) -> dict:
        return {
            "mean_adg": self.mean_adg,
            "mean_sharpe": self.mean_sharpe,
            "mean_drawdown": self.mean_drawdown,
            "mean_loss_profit_ratio": self.mean_loss_profit_ratio,
            "std_adg": self.std_adg,
            "std_sharpe": self.std_sharpe,
            "median_adg": self.median_adg,
            "q25_adg": self.q25_adg,
            "q75_adg": self.q75_adg,
            "min_adg": self.min_adg,
            "max_adg": self.max_adg,
            "min_sharpe": self.min_sharpe,
            "max_sharpe": self.max_sharpe,
            "n_folds": self.n_folds,
            "n_profitable_folds": self.n_profitable_folds,
        }


class ResultsAggregator:
    """
    Aggregate validation results across folds.
    """

    def __init__(
        self,
        adg_key: str = "adg_pnl",
        sharpe_key: str = "sharpe_ratio_pnl",
        drawdown_key: str = "drawdown_worst_usd",
        lpr_key: str = "loss_profit_ratio",
    ):
        self.adg_key = adg_key
        self.sharpe_key = sharpe_key
        self.drawdown_key = drawdown_key
        self.lpr_key = lpr_key

    def aggregate(
        self,
        fold_results: List[Dict[str, Any]],
    ) -> AggregatedMetrics:
        """
        Aggregate metrics across folds.

        Args:
            fold_results: List of metrics dictionaries, one per fold

        Returns:
            AggregatedMetrics with summary statistics
        """
        if not fold_results:
            return self._empty_metrics()

        # Extract values
        adg_values = [self._get_metric(r, self.adg_key, 0.0) for r in fold_results]
        sharpe_values = [self._get_metric(r, self.sharpe_key, 0.0) for r in fold_results]
        drawdown_values = [self._get_metric(r, self.drawdown_key, 1.0) for r in fold_results]
        lpr_values = [self._get_metric(r, self.lpr_key, 1.0) for r in fold_results]

        adg_arr = np.array(adg_values)
        sharpe_arr = np.array(sharpe_values)

        return AggregatedMetrics(
            mean_adg=np.mean(adg_arr),
            mean_sharpe=np.mean(sharpe_arr),
            mean_drawdown=np.mean(drawdown_values),
            mean_loss_profit_ratio=np.mean(lpr_values),
            std_adg=np.std(adg_arr) if len(adg_arr) > 1 else 0.0,
            std_sharpe=np.std(sharpe_arr) if len(sharpe_arr) > 1 else 0.0,
            median_adg=np.median(adg_arr),
            q25_adg=np.percentile(adg_arr, 25) if len(adg_arr) >= 4 else np.min(adg_arr),
            q75_adg=np.percentile(adg_arr, 75) if len(adg_arr) >= 4 else np.max(adg_arr),
            min_adg=np.min(adg_arr),
            max_adg=np.max(adg_arr),
            min_sharpe=np.min(sharpe_arr),
            max_sharpe=np.max(sharpe_arr),
            n_folds=len(fold_results),
            n_profitable_folds=int(np.sum(adg_arr > 0)),
        )

    def aggregate_by_config(
        self,
        validation_matrix: Dict[str, Dict[int, Dict[str, Any]]],
    ) -> Dict[str, AggregatedMetrics]:
        """
        Aggregate metrics for each config across all its validation folds.

        Args:
            validation_matrix: {config_hash: {fold_id: metrics}}

        Returns:
            {config_hash: AggregatedMetrics}
        """
        aggregated = {}
        for config_hash, fold_metrics in validation_matrix.items():
            fold_results = list(fold_metrics.values())
            aggregated[config_hash] = self.aggregate(fold_results)
        return aggregated

    def _get_metric(
        self,
        result: Dict[str, Any],
        key: str,
        default: float,
    ) -> float:
        """Extract metric value with fallback patterns."""
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
            return self._get_metric(result["metrics"], key, default)

        return default

    def _empty_metrics(self) -> AggregatedMetrics:
        """Return empty metrics for invalid input."""
        return AggregatedMetrics(
            mean_adg=0.0,
            mean_sharpe=0.0,
            mean_drawdown=1.0,
            mean_loss_profit_ratio=1.0,
            std_adg=0.0,
            std_sharpe=0.0,
            median_adg=0.0,
            q25_adg=0.0,
            q75_adg=0.0,
            min_adg=0.0,
            max_adg=0.0,
            min_sharpe=0.0,
            max_sharpe=0.0,
            n_folds=0,
            n_profitable_folds=0,
        )


def merge_pareto_fronts(
    pareto_fronts: List[List[Dict[str, Any]]],
    objective_keys: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Merge multiple Pareto fronts into a single combined front.

    Args:
        pareto_fronts: List of Pareto fronts (each is a list of member dicts)
        objective_keys: Keys to use for Pareto dominance (default: adg_pnl, sharpe_ratio_pnl)

    Returns:
        Combined Pareto front
    """
    if objective_keys is None:
        objective_keys = ["adg_pnl", "sharpe_ratio_pnl"]

    # Collect all members
    all_members = []
    for front in pareto_fronts:
        all_members.extend(front)

    if not all_members:
        return []

    # Extract objectives
    def get_objectives(member: Dict[str, Any]) -> List[float]:
        objectives = []
        metrics = member.get("metrics", member.get("analysis", member))
        for key in objective_keys:
            val = metrics.get(key, metrics.get(f"{key}_mean", 0.0))
            objectives.append(val if isinstance(val, (int, float)) else 0.0)
        return objectives

    # Find non-dominated solutions
    n = len(all_members)
    dominated = [False] * n

    for i in range(n):
        if dominated[i]:
            continue
        obj_i = get_objectives(all_members[i])

        for j in range(n):
            if i == j or dominated[j]:
                continue
            obj_j = get_objectives(all_members[j])

            # Check if j dominates i
            if _dominates(obj_j, obj_i):
                dominated[i] = True
                break

    return [m for i, m in enumerate(all_members) if not dominated[i]]


def _dominates(obj_a: List[float], obj_b: List[float]) -> bool:
    """
    Check if solution A dominates solution B (for maximization).

    A dominates B if:
    - A is at least as good as B in all objectives
    - A is strictly better than B in at least one objective
    """
    at_least_as_good = all(a >= b for a, b in zip(obj_a, obj_b))
    strictly_better = any(a > b for a, b in zip(obj_a, obj_b))
    return at_least_as_good and strictly_better
