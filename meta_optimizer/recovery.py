"""
Recovery script for meta-optimizer runs that failed mid-validation (e.g., disk full).

Reads existing training results and completed validations, completes missing
validations, then scores and ranks all configs.

Usage:
    python meta_optimizer/recovery.py meta_optimize_results/2026-02-11_201855/
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Add passivbot src to path
PASSIVBOT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PASSIVBOT_ROOT / "src"))
sys.path.insert(0, str(PASSIVBOT_ROOT))

from meta_optimizer.config import MetaOptimizerConfig
from meta_optimizer.splitting.combined_splitter import (
    CombinedFold,
    CombinedSplitter,
    create_splitter_from_config,
)
from meta_optimizer.runners.optimization_runner import extract_configs_from_pareto
from meta_optimizer.runners.validation_runner import (
    ValidationRunner,
    compute_config_hash,
)
from meta_optimizer.scoring.robustness_scorer import RobustnessScorer, RobustnessScore
from meta_optimizer.reporting.report_generator import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("recovery")


@dataclass
class RankedConfig:
    """A configuration with its robustness score and metadata."""
    config: Dict[str, Any]
    config_hash: str
    robustness_score: RobustnessScore
    rank: int
    source_folds: List[int]

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "config_hash": self.config_hash,
            "overall_score": self.robustness_score.overall_score,
            "mean_adg": self.robustness_score.mean_adg,
            "mean_sharpe": self.robustness_score.mean_sharpe,
            "passes_thresholds": self.robustness_score.passes_thresholds,
            "source_folds": self.source_folds,
            "config": self.config,
            "robustness_details": self.robustness_score.to_dict(),
        }


def load_meta_config(results_dir: Path) -> MetaOptimizerConfig:
    """Load the meta config used for this run."""
    config_path = results_dir / "meta_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No meta_config.json found in {results_dir}")
    return MetaOptimizerConfig.load(str(config_path))


def load_base_config(meta_config: MetaOptimizerConfig) -> Dict[str, Any]:
    """Load the base passivbot config."""
    config_path = Path(meta_config.base_config)
    if not config_path.is_absolute():
        config_path = PASSIVBOT_ROOT / config_path

    try:
        from config_utils import load_config
        return load_config(str(config_path))
    except ImportError:
        with open(config_path) as f:
            return json.load(f)


def regenerate_folds(
    meta_config: MetaOptimizerConfig,
    base_config: Dict[str, Any],
) -> List[CombinedFold]:
    """Regenerate the fold structure from config (deterministic)."""
    splitter = create_splitter_from_config(meta_config.validation_scheme)

    start_date = base_config.get("backtest", {}).get("start_date", "2021-11-01")
    end_date = base_config.get("backtest", {}).get("end_date", "now")

    approved_coins = base_config.get("live", {}).get("approved_coins", [])
    if isinstance(approved_coins, dict):
        coins = approved_coins.get("long", [])
    else:
        coins = []

    if isinstance(splitter, CombinedSplitter):
        return splitter.generate_folds(start_date, end_date, coins)
    else:
        raise ValueError(f"Expected CombinedSplitter, got {type(splitter)}")


def collect_pareto_configs(
    results_dir: Path,
    n_folds: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[int]]]:
    """
    Collect and deduplicate Pareto configs from all training folds.

    Returns:
        (all_configs, config_sources) where:
        - all_configs: {config_hash: config_dict}
        - config_sources: {config_hash: [fold_ids]}
    """
    all_configs = {}
    config_sources = {}

    for fold_id in range(n_folds):
        pareto_dir = (
            results_dir / "fold_results" / f"fold_{fold_id}"
            / "training" / "optimization_results" / "pareto"
        )

        if not pareto_dir.exists():
            logger.warning(f"No pareto dir for fold {fold_id}: {pareto_dir}")
            continue

        pareto_files = list(pareto_dir.glob("*.json"))
        logger.info(f"Fold {fold_id}: found {len(pareto_files)} Pareto configs")

        for pf in pareto_files:
            try:
                with open(pf) as f:
                    config = json.load(f)

                # Extract the actual config (may be nested)
                extracted = extract_configs_from_pareto([{"config": config}])
                if not extracted:
                    continue

                actual_config = extracted[0]
                config_hash = compute_config_hash(actual_config)

                if config_hash not in all_configs:
                    all_configs[config_hash] = actual_config
                    config_sources[config_hash] = []

                config_sources[config_hash].append(fold_id)

            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to load {pf}: {e}")

    logger.info(f"Collected {len(all_configs)} unique configs from {n_folds} folds")
    return all_configs, config_sources


def read_existing_validations(
    results_dir: Path,
    fold_id: int,
    config_hashes: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Read existing validation metrics for a fold from disk.

    Returns:
        {config_hash: metrics_dict} for configs that have valid results
    """
    val_dir = results_dir / "fold_results" / f"fold_{fold_id}" / "validation"
    if not val_dir.exists():
        return {}

    metrics = {}
    config_dirs = list(val_dir.iterdir())
    logger.info(f"Fold {fold_id}: scanning {len(config_dirs)} validation result dirs")

    for config_dir in config_dirs:
        if not config_dir.is_dir():
            continue

        # Extract config hash from directory name (format: config_HASH8)
        dir_name = config_dir.name
        if not dir_name.startswith("config_"):
            continue
        short_hash = dir_name.replace("config_", "")

        # Find matching full hash
        matching_hash = None
        for full_hash in config_hashes:
            if full_hash.startswith(short_hash):
                matching_hash = full_hash
                break

        if matching_hash is None:
            continue

        # Find analysis.json
        analysis_files = list(config_dir.rglob("analysis.json"))
        if not analysis_files:
            continue

        # Take most recent if multiple
        if len(analysis_files) > 1:
            latest = max(analysis_files, key=lambda p: p.stat().st_mtime)
        else:
            latest = analysis_files[0]

        try:
            with open(latest) as f:
                result_metrics = json.load(f)
            metrics[matching_hash] = result_metrics
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to read {latest}: {e}")

    logger.info(f"Fold {fold_id}: read {len(metrics)} existing validation results")
    return metrics


async def run_missing_validations(
    results_dir: Path,
    fold: CombinedFold,
    configs: Dict[str, Dict[str, Any]],
    base_config: Dict[str, Any],
    max_parallel: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Run validation backtests for a fold where results don't exist yet.

    Returns:
        {config_hash: metrics_dict}
    """
    runner = ValidationRunner(
        base_config=base_config,
        output_base_dir=str(results_dir),
    )

    fold_dir = results_dir / "fold_results" / f"fold_{fold.fold_id}" / "validation"
    fold_dir.mkdir(parents=True, exist_ok=True)

    config_list = list(configs.values())
    hash_list = list(configs.keys())

    logger.info(
        f"Running validation for {len(configs)} configs on fold {fold.fold_id} "
        f"(max {max_parallel} parallel)"
    )

    results = await runner.run_validations_parallel(
        configs=config_list,
        config_hashes=hash_list,
        fold=fold,
        output_dir=fold_dir,
        max_parallel=max_parallel,
    )

    metrics = {}
    for result in results:
        if result.metrics and "error" not in result.metrics:
            metrics[result.config_hash] = result.metrics
        else:
            logger.warning(
                f"Validation failed for {result.config_hash[:8]} on fold {fold.fold_id}"
            )

    return metrics


async def main():
    parser = argparse.ArgumentParser(
        description="Recover a meta-optimizer run that failed mid-validation"
    )
    parser.add_argument(
        "results_dir",
        help="Path to the results directory (e.g., meta_optimize_results/2026-02-11_201855/)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=3,
        help="Max parallel backtests for validation (default: 3)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip running missing validations; score with available data only",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    start_time = time.time()

    # Step 1: Load configs
    logger.info("=" * 60)
    logger.info("META-OPTIMIZER RECOVERY")
    logger.info("=" * 60)
    logger.info(f"Results dir: {results_dir}")

    meta_config = load_meta_config(results_dir)
    base_config = load_base_config(meta_config)
    logger.info(f"Loaded meta config: {meta_config.base_config}")

    # Step 2: Regenerate fold structure
    folds = regenerate_folds(meta_config, base_config)
    logger.info(f"Regenerated {len(folds)} folds:")
    for fold in folds:
        logger.info(f"  {fold}")

    # Step 3: Collect Pareto configs from training
    all_configs, config_sources = collect_pareto_configs(results_dir, len(folds))
    config_hashes = list(all_configs.keys())
    logger.info(f"Total unique configs to validate: {len(all_configs)}")

    # Step 4: Build validation matrix
    # Read existing validations and run missing ones
    validation_matrix: Dict[str, Dict[int, Dict[str, Any]]] = {
        h: {} for h in config_hashes
    }

    for fold in folds:
        fold_val_dir = (
            results_dir / "fold_results" / f"fold_{fold.fold_id}" / "validation"
        )

        # Check if this fold has existing results
        existing = read_existing_validations(
            results_dir, fold.fold_id, config_hashes
        )

        if existing:
            logger.info(
                f"Fold {fold.fold_id}: {len(existing)}/{len(all_configs)} "
                f"configs already validated"
            )
            for config_hash, metrics in existing.items():
                validation_matrix[config_hash][fold.fold_id] = metrics

            # Check if fold is complete
            if len(existing) >= len(all_configs) * 0.95:  # 95% threshold
                logger.info(f"Fold {fold.fold_id}: considered complete")
                continue

        if args.skip_validation:
            logger.info(f"Fold {fold.fold_id}: skipping validation (--skip-validation)")
            continue

        # Run missing validations
        missing_configs = {
            h: c for h, c in all_configs.items()
            if h not in existing
        }

        if missing_configs:
            logger.info(
                f"Fold {fold.fold_id}: running {len(missing_configs)} missing validations"
            )
            new_metrics = await run_missing_validations(
                results_dir=results_dir,
                fold=fold,
                configs=missing_configs,
                base_config=base_config,
                max_parallel=args.max_parallel,
            )

            for config_hash, metrics in new_metrics.items():
                validation_matrix[config_hash][fold.fold_id] = metrics

            logger.info(
                f"Fold {fold.fold_id}: completed {len(new_metrics)} new validations"
            )

    # Step 5: Score robustness
    logger.info("=" * 60)
    logger.info("Scoring robustness")
    logger.info("=" * 60)

    scorer = RobustnessScorer(
        weights={
            "consistency": meta_config.robustness_weights.consistency,
            "degradation": meta_config.robustness_weights.degradation,
            "worst_case": meta_config.robustness_weights.worst_case,
            "stability": meta_config.robustness_weights.stability,
        },
        thresholds={
            "min_profitable_folds_pct": meta_config.robustness_thresholds.min_profitable_folds_pct,
            "max_degradation_ratio": meta_config.robustness_thresholds.max_degradation_ratio,
            "max_cv_adg": meta_config.robustness_thresholds.max_cv_adg,
            "min_worst_fold_adg": meta_config.robustness_thresholds.min_worst_fold_adg,
        },
    )

    robustness_scores: Dict[str, RobustnessScore] = {}
    for config_hash, fold_metrics in validation_matrix.items():
        validation_results = [m for m in fold_metrics.values() if m]
        score = scorer.score(
            validation_results=validation_results,
            config=all_configs.get(config_hash),
        )
        robustness_scores[config_hash] = score

    # Count passing configs
    passing = sum(1 for s in robustness_scores.values() if s.passes_thresholds)
    logger.info(f"Scored {len(robustness_scores)} configs, {passing} passing thresholds")

    # Step 6: Rank and export
    logger.info("=" * 60)
    logger.info("Ranking and exporting")
    logger.info("=" * 60)

    sorted_hashes = sorted(
        robustness_scores.keys(),
        key=lambda h: (robustness_scores[h].overall_score, robustness_scores[h].mean_adg),
        reverse=True,
    )

    ranked_configs = []
    for rank, config_hash in enumerate(sorted_hashes, start=1):
        sources = config_sources.get(config_hash, [])
        ranked_configs.append(RankedConfig(
            config=all_configs[config_hash],
            config_hash=config_hash,
            robustness_score=robustness_scores[config_hash],
            rank=rank,
            source_folds=sources,
        ))

    # Save results
    # Summary
    elapsed = time.time() - start_time
    summary = {
        "recovery": True,
        "output_dir": str(results_dir),
        "total_configs_evaluated": len(all_configs),
        "total_folds": len(folds),
        "folds_with_validation": sum(
            1 for fold in folds
            if any(fold.fold_id in v for v in validation_matrix.values())
        ),
        "configs_passing_thresholds": passing,
        "elapsed_seconds": elapsed,
        "top_configs": [rc.to_dict() for rc in ranked_configs[:meta_config.output_settings.top_n_configs]],
    }

    summary_path = results_dir / "recovery_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # All ranked configs
    all_ranked_path = results_dir / "all_ranked_configs.json"
    with open(all_ranked_path, "w") as f:
        json.dump([rc.to_dict() for rc in ranked_configs], f, indent=2)
    logger.info(f"All ranked configs saved to {all_ranked_path}")

    # Top configs as individual files
    best_configs_dir = results_dir / "best_configs"
    best_configs_dir.mkdir(exist_ok=True)

    top_n = meta_config.output_settings.top_n_configs
    for rc in ranked_configs[:top_n]:
        config_path = best_configs_dir / f"rank_{rc.rank}.json"
        with open(config_path, "w") as f:
            json.dump(rc.config, f, indent=2)

    logger.info(f"Top {top_n} configs saved to {best_configs_dir}")

    # Generate reports
    try:
        ranked_dicts = [rc.to_dict() for rc in ranked_configs]
        report_gen = ReportGenerator(results_dir)
        report_gen.generate_summary_report(
            ranked_dicts,
            meta_config.to_dict(),
            elapsed,
        )
        text_summary = report_gen.generate_text_summary(ranked_dicts)
        report_gen.save_text_report(text_summary)
        print("\n" + text_summary + "\n")
        logger.info(f"Reports saved to {results_dir / 'reports'}")
    except Exception as e:
        logger.warning(f"Failed to generate reports: {e}")

    # Print top 10
    logger.info("=" * 60)
    logger.info("TOP 10 CONFIGURATIONS")
    logger.info("=" * 60)
    for rc in ranked_configs[:10]:
        status = "PASS" if rc.robustness_score.passes_thresholds else "FAIL"
        n_folds_validated = sum(
            1 for fold_id in range(len(folds))
            if validation_matrix[rc.config_hash].get(fold_id)
        )
        logger.info(
            f"  Rank {rc.rank:3d} [{status}] "
            f"score={rc.robustness_score.overall_score:.4f} "
            f"adg={rc.robustness_score.mean_adg:.6f} "
            f"sharpe={rc.robustness_score.mean_sharpe:.3f} "
            f"folds={n_folds_validated}/{len(folds)} "
            f"hash={rc.config_hash[:8]}"
        )

    logger.info(f"\nRecovery completed in {elapsed:.1f}s")
    logger.info(f"Results saved to {results_dir}")


if __name__ == "__main__":
    asyncio.run(main())
