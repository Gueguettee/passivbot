#!/usr/bin/env python3
"""
Meta-Optimizer CLI - Entry point for robust strategy discovery.

Usage:
    python meta_optimizer/cli.py configs/template.json
    python meta_optimizer/cli.py configs/template.json --meta-config meta_config.json
    python meta_optimizer/cli.py configs/template.json --quick-test
    python meta_optimizer/cli.py --resume meta_optimize_results/20250205_123456/
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_optimizer.config import (
    MetaOptimizerConfig,
    get_default_config,
    get_quick_test_config,
    get_production_config,
    merge_configs,
)
from meta_optimizer.orchestrator import MetaOptimizer


def setup_logging(level: str = "info") -> None:
    """Configure logging for the meta-optimizer."""
    log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    logging.basicConfig(
        level=log_levels.get(level.lower(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Meta-Optimizer for Passivbot: Find robust, generalizable strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s configs/template.json
  %(prog)s configs/template.json --meta-config meta_config.json
  %(prog)s configs/template.json --quick-test
  %(prog)s --resume meta_optimize_results/20250205_123456/
        """,
    )

    parser.add_argument(
        "base_config",
        nargs="?",
        help="Path to base Passivbot configuration file",
    )

    parser.add_argument(
        "--meta-config", "-m",
        help="Path to meta-optimizer configuration file",
    )

    parser.add_argument(
        "--quick-test", "-q",
        action="store_true",
        help="Run quick test with reduced iterations and folds",
    )

    parser.add_argument(
        "--production", "-p",
        action="store_true",
        help="Use production settings (more folds, more iterations)",
    )

    parser.add_argument(
        "--resume", "-r",
        help="Resume from checkpoint directory",
    )

    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for results",
    )

    parser.add_argument(
        "--time-folds",
        type=int,
        help="Number of time-based folds",
    )

    parser.add_argument(
        "--coin-folds",
        type=int,
        help="Number of coin-based folds",
    )

    parser.add_argument(
        "--iters",
        type=int,
        help="Iterations per optimization fold",
    )

    parser.add_argument(
        "--population-size",
        type=int,
        help="Population size for optimizer",
    )

    parser.add_argument(
        "--n-cpus",
        type=int,
        help="Number of CPUs for parallel processing",
    )

    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration and exit without running",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> MetaOptimizerConfig:
    """Build meta-optimizer configuration from arguments."""
    # Start with appropriate base config
    if args.quick_test:
        config = get_quick_test_config()
    elif args.production:
        config = get_production_config()
    else:
        config = get_default_config()

    # Load and merge meta-config file if provided
    if args.meta_config:
        meta_config_path = Path(args.meta_config)
        if meta_config_path.exists():
            with open(meta_config_path) as f:
                override_data = json.load(f)
            config = merge_configs(config, override_data.get("meta_optimization", override_data))
        else:
            raise FileNotFoundError(f"Meta-config file not found: {args.meta_config}")

    # Apply command-line overrides
    if args.base_config:
        config.base_config = args.base_config

    if args.output_dir:
        config.output_dir = args.output_dir

    if args.time_folds:
        config.validation_scheme.time_folds.n_folds = args.time_folds

    if args.coin_folds:
        config.validation_scheme.coin_folds.n_folds = args.coin_folds

    if args.iters:
        config.optimization_settings.iters_per_fold = args.iters

    if args.population_size:
        config.optimization_settings.population_size = args.population_size

    if args.n_cpus:
        config.optimization_settings.n_cpus = args.n_cpus

    config.execution_settings.log_level = args.log_level

    return config


def validate_config(config: MetaOptimizerConfig) -> None:
    """Validate configuration before running."""
    # Check base config exists
    base_config_path = Path(config.base_config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {config.base_config}")

    # Check output directory is writable
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate fold counts
    if config.validation_scheme.time_folds.n_folds < 1:
        raise ValueError("time_folds.n_folds must be at least 1")

    if config.validation_scheme.coin_folds.n_folds < 1:
        raise ValueError("coin_folds.n_folds must be at least 1")


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Handle resume mode
    if args.resume:
        checkpoint_dir = Path(args.resume)
        checkpoint_file = checkpoint_dir / "checkpoint.json"
        if not checkpoint_file.exists():
            print(f"Error: Checkpoint not found at {checkpoint_file}")
            return 1

        with open(checkpoint_file) as f:
            checkpoint = json.load(f)

        config = MetaOptimizerConfig.from_dict(checkpoint["config"])
        setup_logging(config.execution_settings.log_level)

        logger = logging.getLogger("meta_optimizer")
        logger.info(f"Resuming from checkpoint: {checkpoint_dir}")

        optimizer = MetaOptimizer(config, resume_from=checkpoint_dir)
        await optimizer.run()
        return 0

    # Require base_config in normal mode
    if not args.base_config:
        print("Error: base_config is required (unless using --resume)")
        print("Usage: python meta_optimizer/cli.py configs/template.json")
        return 1

    # Build configuration
    try:
        config = build_config(args)
        validate_config(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    setup_logging(config.execution_settings.log_level)
    logger = logging.getLogger("meta_optimizer")

    # Show configuration in dry-run mode
    if args.dry_run:
        print("Meta-Optimizer Configuration:")
        print("=" * 50)
        print(json.dumps(config.to_dict(), indent=2))
        return 0

    # Log startup info
    logger.info("Meta-Optimizer for Passivbot")
    logger.info(f"Base config: {config.base_config}")
    logger.info(f"Validation scheme: {config.validation_scheme.type}")
    logger.info(f"Time folds: {config.validation_scheme.time_folds.n_folds}")
    logger.info(f"Coin folds: {config.validation_scheme.coin_folds.n_folds}")
    logger.info(f"Iterations per fold: {config.optimization_settings.iters_per_fold}")
    logger.info(f"Output directory: {config.output_dir}")

    # Create and run optimizer
    optimizer = MetaOptimizer(config)

    try:
        results = await optimizer.run()
        logger.info(f"Optimization complete. Results saved to: {results.output_dir}")
        logger.info(f"Top {len(results.ranked_configs)} robust configurations found")
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Saving checkpoint...")
        optimizer.save_checkpoint()
        return 130
    except Exception as e:
        logger.exception(f"Error during optimization: {e}")
        return 1


def run():
    """Wrapper for asyncio.run()."""
    return asyncio.run(main())


if __name__ == "__main__":
    sys.exit(run())
