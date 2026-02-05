"""
Configuration schema and utilities for the meta-optimizer.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path


@dataclass
class TimeFoldConfig:
    """Configuration for time-based (walk-forward) validation."""
    n_folds: int = 3
    train_months: int = 18
    val_months: int = 6
    mode: str = "rolling"  # "rolling" or "expanding"
    gap_months: int = 0  # Gap between train and val to avoid lookahead


@dataclass
class CoinFoldConfig:
    """Configuration for coin-based cross-validation."""
    n_folds: int = 2
    stratify_by: str = "market_cap"  # "market_cap", "random", or "alphabetical"
    min_coins_per_fold: int = 5


@dataclass
class ValidationScheme:
    """Overall validation scheme configuration."""
    type: str = "combined"  # "time_only", "coin_only", or "combined"
    time_folds: TimeFoldConfig = field(default_factory=TimeFoldConfig)
    coin_folds: CoinFoldConfig = field(default_factory=CoinFoldConfig)


@dataclass
class OptimizationSettings:
    """Settings for each optimization run."""
    iters_per_fold: int = 20000
    population_size: int = 150
    n_cpus: Optional[int] = None  # None = use all available
    early_stopping_patience: int = 5000
    early_stopping_min_improvement: float = 0.001


@dataclass
class RobustnessWeights:
    """Weights for robustness score components."""
    consistency: float = 0.4
    degradation: float = 0.3
    worst_case: float = 0.2
    stability: float = 0.1

    def __post_init__(self):
        total = self.consistency + self.degradation + self.worst_case + self.stability
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Robustness weights must sum to 1.0, got {total}")


@dataclass
class RobustnessThresholds:
    """Thresholds for filtering strategies."""
    min_profitable_folds_pct: float = 0.8  # At least 80% of folds must be profitable
    max_degradation_ratio: float = 0.5  # Out-of-sample can't be less than 50% of in-sample
    max_cv_adg: float = 1.0  # Coefficient of variation of ADG across folds
    min_worst_fold_adg: float = 0.0  # Worst fold must be at least break-even


@dataclass
class OutputSettings:
    """Settings for output generation."""
    top_n_configs: int = 10
    export_all_pareto: bool = True
    generate_report: bool = True
    save_validation_details: bool = True


@dataclass
class ExecutionSettings:
    """Settings for execution control."""
    n_parallel_backtests: int = 4
    checkpoint_interval_minutes: int = 30
    resume_from_checkpoint: bool = True
    verbose: bool = True
    log_level: str = "info"  # "debug", "info", "warning", "error"


@dataclass
class MetaOptimizerConfig:
    """Complete meta-optimizer configuration."""
    base_config: str = "configs/template.json"
    output_dir: str = "meta_optimize_results"
    validation_scheme: ValidationScheme = field(default_factory=ValidationScheme)
    optimization_settings: OptimizationSettings = field(default_factory=OptimizationSettings)
    robustness_weights: RobustnessWeights = field(default_factory=RobustnessWeights)
    robustness_thresholds: RobustnessThresholds = field(default_factory=RobustnessThresholds)
    output_settings: OutputSettings = field(default_factory=OutputSettings)
    execution_settings: ExecutionSettings = field(default_factory=ExecutionSettings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaOptimizerConfig":
        """Create from dictionary."""
        # Handle nested dataclasses
        if "validation_scheme" in data and isinstance(data["validation_scheme"], dict):
            vs = data["validation_scheme"]
            if "time_folds" in vs and isinstance(vs["time_folds"], dict):
                vs["time_folds"] = TimeFoldConfig(**vs["time_folds"])
            if "coin_folds" in vs and isinstance(vs["coin_folds"], dict):
                vs["coin_folds"] = CoinFoldConfig(**vs["coin_folds"])
            data["validation_scheme"] = ValidationScheme(**vs)

        if "optimization_settings" in data and isinstance(data["optimization_settings"], dict):
            data["optimization_settings"] = OptimizationSettings(**data["optimization_settings"])

        if "robustness_weights" in data and isinstance(data["robustness_weights"], dict):
            data["robustness_weights"] = RobustnessWeights(**data["robustness_weights"])

        if "robustness_thresholds" in data and isinstance(data["robustness_thresholds"], dict):
            data["robustness_thresholds"] = RobustnessThresholds(**data["robustness_thresholds"])

        if "output_settings" in data and isinstance(data["output_settings"], dict):
            data["output_settings"] = OutputSettings(**data["output_settings"])

        if "execution_settings" in data and isinstance(data["execution_settings"], dict):
            data["execution_settings"] = ExecutionSettings(**data["execution_settings"])

        return cls(**data)

    @classmethod
    def load(cls, path: str) -> "MetaOptimizerConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data.get("meta_optimization", data))


def get_default_config() -> MetaOptimizerConfig:
    """Get default meta-optimizer configuration."""
    return MetaOptimizerConfig()


def get_quick_test_config() -> MetaOptimizerConfig:
    """Get configuration for quick testing."""
    return MetaOptimizerConfig(
        validation_scheme=ValidationScheme(
            type="time_only",
            time_folds=TimeFoldConfig(n_folds=2, train_months=12, val_months=3),
        ),
        optimization_settings=OptimizationSettings(
            iters_per_fold=5000,
            population_size=50,
        ),
        output_settings=OutputSettings(
            top_n_configs=5,
            generate_report=False,
        ),
    )


def get_production_config() -> MetaOptimizerConfig:
    """Get configuration for production use."""
    return MetaOptimizerConfig(
        validation_scheme=ValidationScheme(
            type="combined",
            time_folds=TimeFoldConfig(n_folds=4, train_months=24, val_months=6, mode="expanding"),
            coin_folds=CoinFoldConfig(n_folds=3),
        ),
        optimization_settings=OptimizationSettings(
            iters_per_fold=50000,
            population_size=250,
        ),
        robustness_thresholds=RobustnessThresholds(
            min_profitable_folds_pct=0.9,
            max_degradation_ratio=0.4,
        ),
    )


def merge_configs(base: MetaOptimizerConfig, override: Dict[str, Any]) -> MetaOptimizerConfig:
    """Merge override dictionary into base configuration."""
    base_dict = base.to_dict()
    _deep_merge(base_dict, override)
    return MetaOptimizerConfig.from_dict(base_dict)


def _deep_merge(base: Dict, override: Dict) -> None:
    """Recursively merge override into base dictionary."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
