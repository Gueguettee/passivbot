"""
Ranking utilities for sorting configs by robustness.
"""

from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass

from meta_optimizer.scoring.robustness_scorer import RobustnessScore


@dataclass
class RankingCriteria:
    """Criteria for ranking configurations."""
    # Primary sort key
    primary_key: str = "overall_score"

    # Secondary sort keys (for tie-breaking)
    secondary_keys: List[str] = None

    # Whether to require passing thresholds
    require_passing: bool = True

    # Whether higher is better for each key
    higher_is_better: Dict[str, bool] = None

    def __post_init__(self):
        if self.secondary_keys is None:
            self.secondary_keys = ["mean_adg", "mean_sharpe", "worst_fold_adg"]

        if self.higher_is_better is None:
            self.higher_is_better = {
                "overall_score": True,
                "consistency_score": True,
                "degradation_score": True,
                "worst_case_score": True,
                "mean_adg": True,
                "mean_sharpe": True,
                "worst_fold_adg": True,
                "profitable_fold_pct": True,
            }


def rank_configs_by_robustness(
    configs: Dict[str, Dict[str, Any]],
    scores: Dict[str, RobustnessScore],
    criteria: Optional[RankingCriteria] = None,
) -> List[tuple[str, RobustnessScore, int]]:
    """
    Rank configurations by robustness score.

    Args:
        configs: Dictionary of {config_hash: config}
        scores: Dictionary of {config_hash: RobustnessScore}
        criteria: Ranking criteria to use

    Returns:
        List of (config_hash, score, rank) tuples, sorted by rank
    """
    if criteria is None:
        criteria = RankingCriteria()

    # Filter to passing configs if required
    if criteria.require_passing:
        valid_hashes = [h for h, s in scores.items() if s.passes_thresholds]
    else:
        valid_hashes = list(scores.keys())

    # Build sort key
    def sort_key(config_hash: str) -> tuple:
        score = scores[config_hash]
        keys = []

        # Primary key
        primary_val = _get_score_value(score, criteria.primary_key)
        if criteria.higher_is_better.get(criteria.primary_key, True):
            primary_val = -primary_val  # Negate for descending sort
        keys.append(primary_val)

        # Secondary keys
        for key in criteria.secondary_keys:
            val = _get_score_value(score, key)
            if criteria.higher_is_better.get(key, True):
                val = -val
            keys.append(val)

        return tuple(keys)

    # Sort
    sorted_hashes = sorted(valid_hashes, key=sort_key)

    # Assign ranks
    ranked = []
    for rank, config_hash in enumerate(sorted_hashes, start=1):
        ranked.append((config_hash, scores[config_hash], rank))

    # Add non-passing configs at the end if not requiring passing
    if not criteria.require_passing:
        non_passing = [h for h in scores.keys() if h not in valid_hashes]
        for config_hash in non_passing:
            ranked.append((config_hash, scores[config_hash], len(ranked) + 1))

    return ranked


def _get_score_value(score: RobustnessScore, key: str) -> float:
    """Extract value from RobustnessScore by key name."""
    if hasattr(score, key):
        return getattr(score, key)

    # Check in consistency metrics
    if hasattr(score.consistency_metrics, key):
        return getattr(score.consistency_metrics, key)

    # Check in degradation metrics
    if hasattr(score.degradation_metrics, key):
        return getattr(score.degradation_metrics, key)

    return 0.0


def filter_top_n(
    ranked: List[tuple[str, RobustnessScore, int]],
    n: int = 10,
    min_score: float = 0.0,
) -> List[tuple[str, RobustnessScore, int]]:
    """
    Filter to top N configs that meet minimum score.

    Args:
        ranked: Ranked list from rank_configs_by_robustness
        n: Maximum number to return
        min_score: Minimum overall score required

    Returns:
        Filtered list of top configs
    """
    filtered = [
        (h, s, r) for h, s, r in ranked
        if s.overall_score >= min_score
    ]
    return filtered[:n]


def group_by_score_tier(
    ranked: List[tuple[str, RobustnessScore, int]],
    tiers: List[float] = None,
) -> Dict[str, List[tuple[str, RobustnessScore, int]]]:
    """
    Group ranked configs into score tiers.

    Args:
        ranked: Ranked list from rank_configs_by_robustness
        tiers: Score thresholds for tiers (default: [0.8, 0.6, 0.4, 0.2])

    Returns:
        Dictionary of tier name to list of configs
    """
    if tiers is None:
        tiers = [0.8, 0.6, 0.4, 0.2]

    tier_names = ["excellent", "good", "fair", "poor", "very_poor"]
    groups = {name: [] for name in tier_names}

    for item in ranked:
        config_hash, score, rank = item
        overall = score.overall_score

        assigned = False
        for i, threshold in enumerate(tiers):
            if overall >= threshold:
                groups[tier_names[i]].append(item)
                assigned = True
                break

        if not assigned:
            groups["very_poor"].append(item)

    return groups


def compare_configs(
    config_a_hash: str,
    config_b_hash: str,
    scores: Dict[str, RobustnessScore],
) -> Dict[str, Any]:
    """
    Compare two configurations on all metrics.

    Args:
        config_a_hash: First config hash
        config_b_hash: Second config hash
        scores: Dictionary of scores

    Returns:
        Comparison dictionary
    """
    score_a = scores.get(config_a_hash)
    score_b = scores.get(config_b_hash)

    if not score_a or not score_b:
        return {"error": "One or both configs not found"}

    comparison = {
        "config_a": config_a_hash,
        "config_b": config_b_hash,
        "metrics": {},
    }

    # Compare main scores
    metrics_to_compare = [
        "overall_score",
        "consistency_score",
        "degradation_score",
        "worst_case_score",
        "mean_adg",
        "mean_sharpe",
        "worst_fold_adg",
    ]

    for metric in metrics_to_compare:
        val_a = _get_score_value(score_a, metric)
        val_b = _get_score_value(score_b, metric)

        comparison["metrics"][metric] = {
            "config_a": val_a,
            "config_b": val_b,
            "difference": val_a - val_b,
            "winner": "a" if val_a > val_b else "b" if val_b > val_a else "tie",
        }

    # Overall winner
    a_wins = sum(1 for m in comparison["metrics"].values() if m["winner"] == "a")
    b_wins = sum(1 for m in comparison["metrics"].values() if m["winner"] == "b")
    comparison["overall_winner"] = "a" if a_wins > b_wins else "b" if b_wins > a_wins else "tie"

    return comparison
