"""
Report generation utilities.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class ReportGenerator:
    """
    Generate reports from meta-optimization results.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

    def generate_summary_report(
        self,
        ranked_configs: List[Dict[str, Any]],
        meta_config: Dict[str, Any],
        elapsed_seconds: float,
    ) -> Path:
        """
        Generate a summary report in JSON format.

        Args:
            ranked_configs: List of ranked configuration dictionaries
            meta_config: Meta-optimizer configuration used
            elapsed_seconds: Total elapsed time

        Returns:
            Path to generated report
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "total_configs_evaluated": len(ranked_configs),
            "configs_passing_thresholds": sum(
                1 for c in ranked_configs if c.get("passes_thresholds", False)
            ),
            "meta_config": meta_config,
            "summary_statistics": self._compute_summary_stats(ranked_configs),
            "top_10_configs": ranked_configs[:10],
        }

        report_path = self.reports_dir / "summary.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report_path

    def generate_text_summary(
        self,
        ranked_configs: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a human-readable text summary.

        Args:
            ranked_configs: List of ranked configuration dictionaries

        Returns:
            Text summary string
        """
        lines = [
            "=" * 60,
            "META-OPTIMIZATION RESULTS SUMMARY",
            "=" * 60,
            "",
            f"Total configurations evaluated: {len(ranked_configs)}",
            f"Configurations passing thresholds: {sum(1 for c in ranked_configs if c.get('passes_thresholds', False))}",
            "",
            "TOP 10 ROBUST CONFIGURATIONS",
            "-" * 40,
        ]

        for i, config in enumerate(ranked_configs[:10], start=1):
            lines.append(f"\nRank {i}:")
            lines.append(f"  Hash: {config.get('config_hash', 'N/A')[:16]}...")
            lines.append(f"  Overall Score: {config.get('overall_score', 0):.4f}")
            lines.append(f"  Mean ADG: {config.get('mean_adg', 0):.6f}")
            lines.append(f"  Mean Sharpe: {config.get('mean_sharpe', 0):.2f}")
            lines.append(f"  Passes Thresholds: {config.get('passes_thresholds', False)}")

            if config.get("failure_reasons"):
                lines.append(f"  Failure Reasons: {', '.join(config['failure_reasons'])}")

        lines.extend([
            "",
            "=" * 60,
            "END OF SUMMARY",
            "=" * 60,
        ])

        return "\n".join(lines)

    def _compute_summary_stats(
        self,
        ranked_configs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute summary statistics across all configs."""
        if not ranked_configs:
            return {}

        scores = [c.get("overall_score", 0) for c in ranked_configs]
        adgs = [c.get("mean_adg", 0) for c in ranked_configs]
        sharpes = [c.get("mean_sharpe", 0) for c in ranked_configs]

        import numpy as np

        return {
            "score_distribution": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores)),
            },
            "adg_distribution": {
                "mean": float(np.mean(adgs)),
                "std": float(np.std(adgs)),
                "min": float(np.min(adgs)),
                "max": float(np.max(adgs)),
            },
            "sharpe_distribution": {
                "mean": float(np.mean(sharpes)),
                "std": float(np.std(sharpes)),
                "min": float(np.min(sharpes)),
                "max": float(np.max(sharpes)),
            },
        }

    def save_text_report(self, content: str, filename: str = "summary.txt") -> Path:
        """Save text content to a report file."""
        report_path = self.reports_dir / filename
        with open(report_path, "w") as f:
            f.write(content)
        return report_path
