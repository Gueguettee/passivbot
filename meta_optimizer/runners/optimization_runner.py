"""
Optimization runner - wraps Passivbot optimizer via subprocess.
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from meta_optimizer.splitting.time_splitter import TimeFold
from meta_optimizer.splitting.coin_splitter import CoinFold
from meta_optimizer.splitting.combined_splitter import CombinedFold

logger = logging.getLogger(__name__)

# Path to passivbot root
PASSIVBOT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class OptimizationResult:
    """Results from an optimization run."""
    fold_id: int
    pareto_members: List[Dict[str, Any]]
    results_dir: Path
    elapsed_seconds: float
    iterations_completed: int
    best_metrics: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "pareto_count": len(self.pareto_members),
            "results_dir": str(self.results_dir),
            "elapsed_seconds": self.elapsed_seconds,
            "iterations_completed": self.iterations_completed,
            "best_metrics": self.best_metrics,
        }


class OptimizationRunner:
    """
    Run Passivbot optimizer as subprocess for a given fold.

    This wrapper:
    1. Creates a temporary config file for the fold
    2. Launches optimize.py as subprocess
    3. Monitors progress
    4. Collects Pareto front results
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        iters_per_fold: int = 20000,
        population_size: int = 150,
        n_cpus: Optional[int] = None,
        output_base_dir: str = "meta_optimize_results",
    ):
        self.base_config = base_config
        self.iters_per_fold = iters_per_fold
        self.population_size = population_size
        self.n_cpus = n_cpus
        self.output_base_dir = Path(output_base_dir)

    async def run_optimization(
        self,
        fold: Union[TimeFold, CoinFold, CombinedFold],
        fold_output_dir: Path,
    ) -> OptimizationResult:
        """
        Run optimization for a single fold.

        Args:
            fold: The fold to optimize on
            fold_output_dir: Directory to store results

        Returns:
            OptimizationResult with Pareto members and metrics
        """
        import time
        start_time = time.time()

        # Create fold-specific config
        fold_config = self._create_fold_config(fold)

        # Write temporary config file
        config_path = fold_output_dir / "fold_config.json"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(fold_config, f, indent=2)

        logger.info(f"Running optimization for fold {fold.fold_id}")
        logger.debug(f"Config written to: {config_path}")

        # Build command
        cmd = self._build_command(config_path, fold_output_dir)
        logger.debug(f"Command: {' '.join(cmd)}")

        # Run optimizer
        logger.info(f"Starting optimizer subprocess: {' '.join(cmd)}")
        logger.info(f"Working directory: {PASSIVBOT_ROOT}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PASSIVBOT_ROOT),
        )

        # Monitor and log output from both streams
        iterations_completed = 0
        output_lines = []

        async def read_stream(stream, prefix):
            nonlocal iterations_completed
            while True:
                line = await stream.readline()
                if not line:
                    break
                line_str = line.decode().strip()
                if line_str:
                    output_lines.append(f"{prefix}: {line_str}")
                    logger.debug(f"[optimizer {prefix}] {line_str}")
                    # Parse iteration count from output
                    if "gen" in line_str.lower() or "iter" in line_str.lower():
                        try:
                            parts = line_str.split()
                            for part in parts:
                                if part.isdigit():
                                    iterations_completed = max(iterations_completed, int(part))
                                    break
                        except:
                            pass

        # Read both stdout and stderr concurrently
        await asyncio.gather(
            read_stream(process.stdout, "stdout"),
            read_stream(process.stderr, "stderr"),
        )

        return_code = await process.wait()

        if return_code != 0:
            logger.error(f"Optimizer process exited with code {return_code}")
            logger.error(f"Last 20 lines of output:")
            for line in output_lines[-20:]:
                logger.error(f"  {line}")
        else:
            logger.info(f"Optimizer subprocess completed successfully")

        elapsed = time.time() - start_time
        logger.info(f"Fold {fold.fold_id} optimization completed in {elapsed:.1f}s")

        # Find and load results
        pareto_members, best_metrics = self._load_results(fold_output_dir)

        return OptimizationResult(
            fold_id=fold.fold_id,
            pareto_members=pareto_members,
            results_dir=fold_output_dir,
            elapsed_seconds=elapsed,
            iterations_completed=iterations_completed,
            best_metrics=best_metrics,
        )

    def _create_fold_config(
        self,
        fold: Union[TimeFold, CoinFold, CombinedFold],
    ) -> Dict[str, Any]:
        """Create config for fold training data."""
        config = json.loads(json.dumps(self.base_config))  # Deep copy

        # Apply fold-specific settings
        if isinstance(fold, TimeFold):
            config["backtest"]["start_date"] = fold.train_start
            config["backtest"]["end_date"] = fold.train_end

        elif isinstance(fold, CoinFold):
            config["live"]["approved_coins"] = {
                "long": fold.train_coins,
                "short": fold.train_coins,
            }

        elif isinstance(fold, CombinedFold):
            config["backtest"]["start_date"] = fold.train_start
            config["backtest"]["end_date"] = fold.train_end
            config["live"]["approved_coins"] = {
                "long": fold.train_coins,
                "short": fold.train_coins,
            }

        # Apply optimization settings
        config["optimize"]["iters"] = self.iters_per_fold
        config["optimize"]["population_size"] = self.population_size
        if self.n_cpus:
            config["optimize"]["n_cpus"] = self.n_cpus

        return config

    def _build_command(self, config_path: Path, output_dir: Path) -> List[str]:
        """Build subprocess command."""
        cmd = [
            sys.executable,
            str(PASSIVBOT_ROOT / "src" / "optimize.py"),
            str(config_path),
        ]

        return cmd

    def _load_results(
        self,
        output_dir: Path,
    ) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Load Pareto front results from output directory."""
        pareto_members = []
        best_metrics = {}

        # First check if results were already copied to fold output directory
        fold_results_dir = output_dir / "optimization_results"
        fold_pareto_dir = fold_results_dir / "pareto"

        if fold_pareto_dir.exists() and fold_pareto_dir.is_dir():
            # Load from already-copied results (resume scenario)
            pareto_files = list(fold_pareto_dir.glob("*.json"))
            if pareto_files:
                logger.info(f"Found {len(pareto_files)} existing Pareto configs in {fold_pareto_dir}")
                for pf in pareto_files:
                    try:
                        with open(pf) as f:
                            config = json.load(f)
                        pareto_members.append({
                            "config": config,
                            "config_hash": pf.stem,
                        })
                    except (json.JSONDecodeError, Exception) as e:
                        logger.warning(f"Failed to load pareto config {pf}: {e}")
                        continue

                if pareto_members:
                    logger.info(f"Loaded {len(pareto_members)} Pareto members from existing results")
                    # Extract metrics from first member
                    first = pareto_members[0]
                    if "config" in first and isinstance(first["config"], dict):
                        if "analysis" in first["config"]:
                            best_metrics = first["config"]["analysis"]
                    return pareto_members, best_metrics

        # Find the optimizer results directory for fresh results
        # Passivbot creates directories like: optimize_results/YYYY-MM-DDTHH_MM_SS_...
        optimize_results_dir = PASSIVBOT_ROOT / "optimize_results"
        if not optimize_results_dir.exists():
            logger.warning(f"No optimize_results directory found")
            return pareto_members, best_metrics

        # Find most recent results directory
        result_dirs = sorted(optimize_results_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if not result_dirs:
            logger.warning(f"No result directories found in {optimize_results_dir}")
            return pareto_members, best_metrics

        latest_dir = result_dirs[0]
        logger.debug(f"Loading results from: {latest_dir}")

        # Copy results to fold output directory first
        fold_results_dir = output_dir / "optimization_results"
        if latest_dir.exists():
            try:
                shutil.copytree(latest_dir, fold_results_dir, dirs_exist_ok=True)
            except Exception as e:
                logger.warning(f"Failed to copy results: {e}")

        # Try to load Pareto configs from pareto/ folder (new format)
        pareto_folder = latest_dir / "pareto"
        if pareto_folder.exists() and pareto_folder.is_dir():
            pareto_files = list(pareto_folder.glob("*.json"))
            logger.info(f"Found {len(pareto_files)} Pareto config files in {pareto_folder}")

            for pf in pareto_files:
                try:
                    with open(pf) as f:
                        config = json.load(f)
                    # Wrap the config in a member dict for consistency
                    pareto_members.append({
                        "config": config,
                        "config_hash": pf.stem,  # filename without extension
                    })
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Failed to load pareto config {pf}: {e}")
                    continue

        # Fallback: try legacy optimizer.pareto file format
        if not pareto_members:
            pareto_file = latest_dir / "optimizer.pareto"
            if pareto_file.exists():
                with open(pareto_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                member = json.loads(line)
                                pareto_members.append(member)
                            except json.JSONDecodeError:
                                continue

        logger.info(f"Loaded {len(pareto_members)} Pareto members total")

        # Extract best metrics (from first Pareto member if available)
        if pareto_members:
            first = pareto_members[0]
            if "metrics" in first:
                best_metrics = first["metrics"]
            elif "analysis" in first:
                best_metrics = first["analysis"]
            elif "config" in first and "analysis" in first.get("config", {}):
                best_metrics = first["config"]["analysis"]

        return pareto_members, best_metrics


def extract_configs_from_pareto(pareto_members: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract bot configs from Pareto members.

    Args:
        pareto_members: List of Pareto member dictionaries

    Returns:
        List of bot configuration dictionaries (full configs, not just bot section)
    """
    configs = []
    for member in pareto_members:
        if "config" in member:
            config = member["config"]
            # If config has full structure (backtest, bot, etc.), use it directly
            if isinstance(config, dict) and ("bot" in config or "backtest" in config):
                configs.append(config)
            else:
                configs.append(member["config"])
        elif "bot" in member:
            # Legacy format - just bot section
            configs.append({"bot": member["bot"]})
        elif "backtest" in member and "bot" in member:
            # Full config stored directly in member
            configs.append(member)
    return configs


def deduplicate_configs(
    configs: List[Dict[str, Any]],
    tolerance: float = 1e-6,
) -> List[Dict[str, Any]]:
    """
    Remove duplicate configs based on parameter values.

    Args:
        configs: List of bot configurations
        tolerance: Tolerance for floating point comparison

    Returns:
        Deduplicated list of configs
    """
    unique = []
    seen_hashes = set()

    for config in configs:
        # Create a hashable representation
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hash(config_str)

        if config_hash not in seen_hashes:
            seen_hashes.add(config_hash)
            unique.append(config)

    return unique
