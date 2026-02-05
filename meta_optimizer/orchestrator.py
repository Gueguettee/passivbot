"""
Meta-Optimizer Orchestrator - coordinates the entire optimization workflow.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from meta_optimizer.config import MetaOptimizerConfig
from meta_optimizer.splitting.time_splitter import TimeSplitter, TimeFold
from meta_optimizer.splitting.coin_splitter import CoinSplitter, CoinFold
from meta_optimizer.splitting.combined_splitter import (
    CombinedSplitter,
    CombinedFold,
    create_splitter_from_config,
)
from meta_optimizer.runners.optimization_runner import (
    OptimizationRunner,
    OptimizationResult,
    extract_configs_from_pareto,
    deduplicate_configs,
)
from meta_optimizer.runners.validation_runner import (
    ValidationRunner,
    ValidationResult,
    compute_config_hash,
)
from meta_optimizer.scoring.robustness_scorer import RobustnessScorer, RobustnessScore

logger = logging.getLogger(__name__)

# Passivbot root directory
PASSIVBOT_ROOT = Path(__file__).parent.parent


@dataclass
class RankedConfig:
    """A configuration with its robustness score and metadata."""
    config: Dict[str, Any]
    config_hash: str
    robustness_score: RobustnessScore
    rank: int
    source_folds: List[int]  # Which optimization folds this config came from

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


@dataclass
class MetaOptimizationResult:
    """Complete results from a meta-optimization run."""
    output_dir: Path
    ranked_configs: List[RankedConfig]
    total_configs_evaluated: int
    total_folds: int
    elapsed_seconds: float
    config_used: MetaOptimizerConfig

    def to_dict(self) -> dict:
        return {
            "output_dir": str(self.output_dir),
            "total_configs_evaluated": self.total_configs_evaluated,
            "total_folds": self.total_folds,
            "elapsed_seconds": self.elapsed_seconds,
            "top_configs": [rc.to_dict() for rc in self.ranked_configs],
        }


class MetaOptimizer:
    """
    Orchestrates the meta-optimization workflow.

    Workflow:
    1. Generate data splits (time, coin, or combined)
    2. Run optimization on each training fold
    3. Collect unique configs from all Pareto fronts
    4. Validate each config on all validation folds
    5. Score robustness for each config
    6. Rank and export results
    """

    def __init__(
        self,
        config: MetaOptimizerConfig,
        resume_from: Optional[Path] = None,
    ):
        self.config = config
        self.resume_from = resume_from

        # Initialize paths
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.output_dir = Path(config.output_dir) / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load base passivbot config
        self.base_config = self._load_base_config()

        # Initialize components
        self.splitter = create_splitter_from_config(config.validation_scheme)
        self.scorer = RobustnessScorer(
            weights={
                "consistency": config.robustness_weights.consistency,
                "degradation": config.robustness_weights.degradation,
                "worst_case": config.robustness_weights.worst_case,
                "stability": config.robustness_weights.stability,
            },
            thresholds={
                "min_profitable_folds_pct": config.robustness_thresholds.min_profitable_folds_pct,
                "max_degradation_ratio": config.robustness_thresholds.max_degradation_ratio,
                "max_cv_adg": config.robustness_thresholds.max_cv_adg,
                "min_worst_fold_adg": config.robustness_thresholds.min_worst_fold_adg,
            },
        )

        # State for checkpointing
        self.state = {
            "phase": "init",
            "folds": [],
            "optimization_results": {},
            "all_configs": {},
            "validation_matrix": {},
            "robustness_scores": {},
        }

    def _load_base_config(self) -> Dict[str, Any]:
        """Load the base Passivbot configuration."""
        config_path = Path(self.config.base_config)
        if not config_path.is_absolute():
            config_path = PASSIVBOT_ROOT / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Base config not found: {config_path}")

        # Try to use Passivbot's config loader
        try:
            import sys
            sys.path.insert(0, str(PASSIVBOT_ROOT / "src"))
            from config_utils import load_config
            return load_config(str(config_path))
        except ImportError:
            # Fallback to basic JSON loading
            with open(config_path) as f:
                return json.load(f)

    async def run(self) -> MetaOptimizationResult:
        """
        Run the complete meta-optimization workflow.

        Returns:
            MetaOptimizationResult with ranked configs
        """
        import time
        start_time = time.time()

        logger.info("Starting meta-optimization")
        logger.info(f"Output directory: {self.output_dir}")

        # Save config
        self.config.save(str(self.output_dir / "meta_config.json"))

        # Phase 1: Generate folds
        logger.info("Phase 1: Generating data splits")
        folds = self._generate_folds()
        self.state["folds"] = [f.to_dict() for f in folds]
        self._save_checkpoint()

        logger.info(f"Generated {len(folds)} folds")
        for fold in folds:
            logger.info(f"  {fold}")

        # Phase 2: Run optimization on each fold
        logger.info("Phase 2: Running optimization on training folds")
        optimization_results = await self._run_optimizations(folds)

        # Phase 3: Collect unique configs
        logger.info("Phase 3: Collecting unique configurations")
        all_configs = self._collect_unique_configs(optimization_results)
        logger.info(f"Collected {len(all_configs)} unique configurations")

        # Phase 4: Validate configs on all folds
        logger.info("Phase 4: Validating configurations on all folds")
        validation_matrix = await self._run_validations(all_configs, folds)

        # Phase 5: Score robustness
        logger.info("Phase 5: Scoring robustness")
        robustness_scores = self._score_robustness(all_configs, validation_matrix)

        # Phase 6: Rank and export
        logger.info("Phase 6: Ranking and exporting results")
        ranked_configs = self._rank_configs(all_configs, robustness_scores)

        elapsed = time.time() - start_time

        # Save results
        result = MetaOptimizationResult(
            output_dir=self.output_dir,
            ranked_configs=ranked_configs[:self.config.output_settings.top_n_configs],
            total_configs_evaluated=len(all_configs),
            total_folds=len(folds),
            elapsed_seconds=elapsed,
            config_used=self.config,
        )

        self._save_results(result, ranked_configs)

        logger.info(f"Meta-optimization complete in {elapsed:.1f}s")
        logger.info(f"Found {len(ranked_configs)} robust configurations")

        return result

    def _generate_folds(self) -> List[Union[TimeFold, CoinFold, CombinedFold]]:
        """Generate data folds based on validation scheme."""
        # Extract date range and coins from base config
        start_date = self.base_config.get("backtest", {}).get("start_date", "2021-01-01")
        end_date = self.base_config.get("backtest", {}).get("end_date", "now")

        # Get coins list
        approved_coins = self.base_config.get("live", {}).get("approved_coins", [])
        if isinstance(approved_coins, dict):
            # Use long coins as representative
            coins = approved_coins.get("long", [])
        elif isinstance(approved_coins, list):
            coins = approved_coins
        else:
            coins = []

        # Generate folds based on splitter type
        if isinstance(self.splitter, TimeSplitter):
            return self.splitter.generate_folds(start_date, end_date)
        elif isinstance(self.splitter, CoinSplitter):
            return self.splitter.generate_folds(coins)
        elif isinstance(self.splitter, CombinedSplitter):
            return self.splitter.generate_folds(start_date, end_date, coins)
        else:
            raise ValueError(f"Unknown splitter type: {type(self.splitter)}")

    async def _run_optimizations(
        self,
        folds: List[Union[TimeFold, CoinFold, CombinedFold]],
    ) -> Dict[int, OptimizationResult]:
        """Run optimization on each training fold."""
        runner = OptimizationRunner(
            base_config=self.base_config,
            iters_per_fold=self.config.optimization_settings.iters_per_fold,
            population_size=self.config.optimization_settings.population_size,
            n_cpus=self.config.optimization_settings.n_cpus,
            output_base_dir=str(self.output_dir),
        )

        results = {}
        for fold in folds:
            fold_dir = self.output_dir / "fold_results" / f"fold_{fold.fold_id}" / "training"
            result = await runner.run_optimization(fold, fold_dir)
            results[fold.fold_id] = result

            self.state["optimization_results"][fold.fold_id] = result.to_dict()
            self._save_checkpoint()

            logger.info(
                f"Fold {fold.fold_id}: {len(result.pareto_members)} Pareto members, "
                f"{result.elapsed_seconds:.1f}s"
            )

        return results

    def _collect_unique_configs(
        self,
        optimization_results: Dict[int, OptimizationResult],
    ) -> Dict[str, Dict[str, Any]]:
        """Collect and deduplicate configs from all Pareto fronts."""
        all_configs = {}
        config_sources = {}  # Track which folds each config came from

        for fold_id, result in optimization_results.items():
            configs = extract_configs_from_pareto(result.pareto_members)

            for config in configs:
                config_hash = compute_config_hash(config)

                if config_hash not in all_configs:
                    all_configs[config_hash] = config
                    config_sources[config_hash] = []

                config_sources[config_hash].append(fold_id)

        self.state["all_configs"] = {h: {"config": c, "sources": config_sources[h]}
                                      for h, c in all_configs.items()}
        self._save_checkpoint()

        return all_configs

    async def _run_validations(
        self,
        configs: Dict[str, Dict[str, Any]],
        folds: List[Union[TimeFold, CoinFold, CombinedFold]],
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Validate each config on all validation folds.

        Returns:
            Matrix of {config_hash: {fold_id: metrics}}
        """
        runner = ValidationRunner(
            base_config=self.base_config,
            output_base_dir=str(self.output_dir),
        )

        validation_matrix = {h: {} for h in configs.keys()}

        for fold in folds:
            logger.info(f"Validating {len(configs)} configs on fold {fold.fold_id}")

            fold_dir = self.output_dir / "fold_results" / f"fold_{fold.fold_id}" / "validation"

            # Run validations in parallel
            config_list = list(configs.values())
            hash_list = list(configs.keys())

            results = await runner.run_validations_parallel(
                configs=config_list,
                config_hashes=hash_list,
                fold=fold,
                output_dir=fold_dir,
                max_parallel=self.config.execution_settings.n_parallel_backtests,
            )

            # Store results
            for result in results:
                validation_matrix[result.config_hash][fold.fold_id] = result.metrics

            self.state["validation_matrix"] = validation_matrix
            self._save_checkpoint()

        return validation_matrix

    def _score_robustness(
        self,
        configs: Dict[str, Dict[str, Any]],
        validation_matrix: Dict[str, Dict[int, Dict[str, Any]]],
    ) -> Dict[str, RobustnessScore]:
        """Score robustness for each config."""
        scores = {}

        for config_hash, fold_metrics in validation_matrix.items():
            # Convert fold metrics to list
            validation_results = list(fold_metrics.values())

            # Score
            score = self.scorer.score(
                validation_results=validation_results,
                config=configs.get(config_hash),
            )
            scores[config_hash] = score

            self.state["robustness_scores"][config_hash] = score.to_dict()

        self._save_checkpoint()
        return scores

    def _rank_configs(
        self,
        configs: Dict[str, Dict[str, Any]],
        scores: Dict[str, RobustnessScore],
    ) -> List[RankedConfig]:
        """Rank configs by robustness score."""
        # Sort by overall score (descending)
        sorted_hashes = sorted(
            scores.keys(),
            key=lambda h: scores[h].overall_score,
            reverse=True,
        )

        ranked = []
        for rank, config_hash in enumerate(sorted_hashes, start=1):
            source_folds = self.state["all_configs"].get(config_hash, {}).get("sources", [])

            ranked.append(RankedConfig(
                config=configs[config_hash],
                config_hash=config_hash,
                robustness_score=scores[config_hash],
                rank=rank,
                source_folds=source_folds,
            ))

        return ranked

    def _save_results(
        self,
        result: MetaOptimizationResult,
        ranked_configs: List[RankedConfig],
    ) -> None:
        """Save final results."""
        # Save summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save all ranked configs
        all_ranked_path = self.output_dir / "all_ranked_configs.json"
        with open(all_ranked_path, "w") as f:
            json.dump([rc.to_dict() for rc in ranked_configs], f, indent=2)

        # Save top configs as individual files
        best_configs_dir = self.output_dir / "best_configs"
        best_configs_dir.mkdir(exist_ok=True)

        for rc in ranked_configs[:self.config.output_settings.top_n_configs]:
            config_path = best_configs_dir / f"rank_{rc.rank}.json"
            with open(config_path, "w") as f:
                json.dump(rc.config, f, indent=2)

        logger.info(f"Results saved to {self.output_dir}")

    def _save_checkpoint(self) -> None:
        """Save current state for resume."""
        checkpoint_path = self.output_dir / "checkpoint.json"
        checkpoint = {
            "config": self.config.to_dict(),
            "state": self.state,
            "timestamp": datetime.now().isoformat(),
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def save_checkpoint(self) -> None:
        """Public method for saving checkpoint on interrupt."""
        self._save_checkpoint()
        logger.info(f"Checkpoint saved to {self.output_dir / 'checkpoint.json'}")
