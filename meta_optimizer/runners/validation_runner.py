"""
Validation runner - runs backtests on validation folds.
"""

import asyncio
import json
import logging
import sys
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
class ValidationResult:
    """Results from a validation backtest."""
    config_hash: str
    fold_id: int
    metrics: Dict[str, float]
    is_profitable: bool
    elapsed_seconds: float
    results_dir: Optional[Path] = None

    def to_dict(self) -> dict:
        return {
            "config_hash": self.config_hash,
            "fold_id": self.fold_id,
            "metrics": self.metrics,
            "is_profitable": self.is_profitable,
            "elapsed_seconds": self.elapsed_seconds,
            "results_dir": str(self.results_dir) if self.results_dir else None,
        }


class ValidationRunner:
    """
    Run backtests on validation folds via subprocess.

    This wrapper:
    1. Creates config with validation period/coins
    2. Launches backtest.py as subprocess
    3. Collects metrics from analysis.json
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        output_base_dir: str = "meta_optimize_results",
        disable_plotting: bool = True,
    ):
        self.base_config = base_config
        self.output_base_dir = Path(output_base_dir)
        self.disable_plotting = disable_plotting

    async def run_validation(
        self,
        config: Dict[str, Any],
        fold: Union[TimeFold, CoinFold, CombinedFold],
        config_hash: str,
        output_dir: Path,
    ) -> ValidationResult:
        """
        Run validation backtest for a config on a fold.

        Args:
            config: Bot configuration to validate
            fold: Validation fold (defines period and/or coins)
            config_hash: Unique identifier for this config
            output_dir: Directory for results

        Returns:
            ValidationResult with metrics
        """
        import time
        start_time = time.time()

        # Create output directory first
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create validation config (with unique base_dir for backtest output)
        val_config = self._create_validation_config(config, fold, output_dir)

        # Write config to file
        config_path = output_dir / f"val_config_{config_hash}_{fold.fold_id}.json"
        with open(config_path, "w") as f:
            json.dump(val_config, f, indent=2)

        logger.debug(f"Running validation for config {config_hash[:8]} on fold {fold.fold_id}")

        # Build and run command
        cmd = self._build_command(config_path)
        logger.info(f"Backtest cmd: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PASSIVBOT_ROOT),
        )

        # Read output from both streams
        output_lines = []

        async def read_stream(stream, prefix):
            while True:
                line = await stream.readline()
                if not line:
                    break
                line_str = line.decode().strip()
                if line_str:
                    output_lines.append(f"{prefix}: {line_str}")
                    logger.debug(f"[backtest {prefix}] {line_str}")

        await asyncio.gather(
            read_stream(process.stdout, "stdout"),
            read_stream(process.stderr, "stderr"),
        )

        return_code = await process.wait()
        elapsed = time.time() - start_time

        if return_code != 0:
            logger.error(f"Backtest failed for config {config_hash[:8]} fold {fold.fold_id} (code {return_code})")
            for line in output_lines[-10:]:
                logger.error(f"  {line}")

        # Load metrics from config-specific results directory
        metrics = self._load_metrics(output_dir)

        # Determine if profitable
        adg = metrics.get("adg_pnl", metrics.get("adg_pnl_mean", 0))
        is_profitable = adg > 0

        return ValidationResult(
            config_hash=config_hash,
            fold_id=fold.fold_id,
            metrics=metrics,
            is_profitable=is_profitable,
            elapsed_seconds=elapsed,
            results_dir=output_dir,
        )

    async def run_validations_parallel(
        self,
        configs: List[Dict[str, Any]],
        config_hashes: List[str],
        fold: Union[TimeFold, CoinFold, CombinedFold],
        output_dir: Path,
        max_parallel: int = 4,
    ) -> List[ValidationResult]:
        """
        Run multiple validations in parallel.

        Args:
            configs: List of configs to validate
            config_hashes: Corresponding hashes
            fold: Validation fold
            output_dir: Base output directory
            max_parallel: Maximum parallel backtests

        Returns:
            List of ValidationResult
        """
        semaphore = asyncio.Semaphore(max_parallel)

        async def run_with_semaphore(config, config_hash):
            async with semaphore:
                cfg_output_dir = output_dir / f"config_{config_hash[:8]}"
                return await self.run_validation(config, fold, config_hash, cfg_output_dir)

        tasks = [
            run_with_semaphore(cfg, hash_)
            for cfg, hash_ in zip(configs, config_hashes)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Validation failed for config {config_hashes[i][:8]}: {result}")
                # Create a failed result
                valid_results.append(ValidationResult(
                    config_hash=config_hashes[i],
                    fold_id=fold.fold_id,
                    metrics={"error": str(result)},
                    is_profitable=False,
                    elapsed_seconds=0,
                ))
            else:
                valid_results.append(result)

        return valid_results

    def _create_validation_config(
        self,
        bot_config: Dict[str, Any],
        fold: Union[TimeFold, CoinFold, CombinedFold],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Create config for validation backtest."""
        # Start with base config
        config = json.loads(json.dumps(self.base_config))

        # Apply bot parameters
        if "bot" in bot_config:
            config["bot"] = bot_config["bot"]
        else:
            config["bot"] = bot_config

        # Direct backtest output to this config's unique directory
        # This prevents race conditions when running parallel backtests
        config["backtest"]["base_dir"] = str(output_dir / "backtests")

        # Apply validation fold settings
        if isinstance(fold, TimeFold):
            config["backtest"]["start_date"] = fold.val_start
            config["backtest"]["end_date"] = fold.val_end

        elif isinstance(fold, CoinFold):
            config["live"]["approved_coins"] = {
                "long": fold.val_coins,
                "short": fold.val_coins,
            }

        elif isinstance(fold, CombinedFold):
            config["backtest"]["start_date"] = fold.val_start
            config["backtest"]["end_date"] = fold.val_end
            config["live"]["approved_coins"] = {
                "long": fold.val_coins,
                "short": fold.val_coins,
            }

        return config

    def _build_command(self, config_path: Path) -> List[str]:
        """Build backtest command."""
        cmd = [
            sys.executable,
            str(PASSIVBOT_ROOT / "src" / "backtest.py"),
            str(config_path),
        ]

        if self.disable_plotting:
            cmd.append("-dp")  # disable plotting

        return cmd

    def _load_metrics(self, output_dir: Path) -> Dict[str, float]:
        """Load metrics from backtest output in the config-specific directory."""
        # Search for analysis.json in the config's own backtests directory
        backtests_dir = output_dir / "backtests"

        if not backtests_dir.exists():
            logger.warning(f"No backtests directory found at {backtests_dir}")
            return {}

        # Recursively find all analysis.json files
        analysis_files = list(backtests_dir.rglob("analysis.json"))

        if not analysis_files:
            logger.warning(f"No analysis.json found under {backtests_dir}")
            return {}

        # If multiple (shouldn't happen), take the most recent
        if len(analysis_files) > 1:
            latest = max(analysis_files, key=lambda p: p.stat().st_mtime)
        else:
            latest = analysis_files[0]

        try:
            with open(latest) as f:
                metrics = json.load(f)
            logger.info(f"Loaded {len(metrics)} metrics from {latest}")
            return metrics
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse {latest}: {e}")
            return {}


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute a hash for a config to use as identifier."""
    import hashlib
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]
