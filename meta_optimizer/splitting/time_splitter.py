"""
Time-based data splitting for walk-forward validation.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple
from dateutil.relativedelta import relativedelta


@dataclass
class TimeFold:
    """Represents a single time-based fold."""
    fold_id: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str

    def __repr__(self) -> str:
        return (
            f"TimeFold(id={self.fold_id}, "
            f"train={self.train_start} to {self.train_end}, "
            f"val={self.val_start} to {self.val_end})"
        )

    def to_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "val_start": self.val_start,
            "val_end": self.val_end,
        }


class TimeSplitter:
    """
    Generate time-based folds for walk-forward validation.

    Supports two modes:
    - rolling: Fixed-size training window that slides forward
    - expanding: Training window grows with each fold

    Example (rolling, 3 folds, 18mo train, 6mo val):
        Fold 1: Train 2021-01 to 2022-06, Val 2022-07 to 2022-12
        Fold 2: Train 2021-07 to 2023-00, Val 2023-01 to 2023-06
        Fold 3: Train 2022-01 to 2023-06, Val 2023-07 to 2023-12

    Example (expanding, 3 folds, 12mo initial, 6mo val):
        Fold 1: Train 2021-01 to 2021-12, Val 2022-01 to 2022-06
        Fold 2: Train 2021-01 to 2022-06, Val 2022-07 to 2022-12
        Fold 3: Train 2021-01 to 2022-12, Val 2023-01 to 2023-06
    """

    def __init__(
        self,
        n_folds: int = 3,
        train_months: int = 18,
        val_months: int = 6,
        mode: str = "rolling",
        gap_months: int = 0,
    ):
        self.n_folds = n_folds
        self.train_months = train_months
        self.val_months = val_months
        self.mode = mode
        self.gap_months = gap_months

        if mode not in ("rolling", "expanding"):
            raise ValueError(f"mode must be 'rolling' or 'expanding', got '{mode}'")

    def generate_folds(
        self,
        start_date: str,
        end_date: str,
    ) -> List[TimeFold]:
        """
        Generate time folds from date range.

        Args:
            start_date: Start date in format "YYYY-MM-DD"
            end_date: End date in format "YYYY-MM-DD" or "now"

        Returns:
            List of TimeFold objects
        """
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)

        total_months = self._months_between(start, end)

        if self.mode == "rolling":
            return self._generate_rolling_folds(start, end, total_months)
        else:
            return self._generate_expanding_folds(start, end, total_months)

    def _generate_rolling_folds(
        self,
        start: datetime,
        end: datetime,
        total_months: int,
    ) -> List[TimeFold]:
        """Generate rolling window folds."""
        folds = []

        # Calculate step size between folds
        fold_size = self.train_months + self.val_months + self.gap_months
        available_months = total_months - fold_size
        step_months = max(1, available_months // max(1, self.n_folds - 1)) if self.n_folds > 1 else 0

        for i in range(self.n_folds):
            offset_months = i * step_months

            train_start = start + relativedelta(months=offset_months)
            train_end = train_start + relativedelta(months=self.train_months) - timedelta(days=1)

            val_start = train_end + timedelta(days=1) + relativedelta(months=self.gap_months)
            val_end = val_start + relativedelta(months=self.val_months) - timedelta(days=1)

            # Ensure we don't exceed the end date
            if val_end > end:
                val_end = end
                if val_start > val_end:
                    break

            folds.append(TimeFold(
                fold_id=i,
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                val_start=val_start.strftime("%Y-%m-%d"),
                val_end=val_end.strftime("%Y-%m-%d"),
            ))

        return folds

    def _generate_expanding_folds(
        self,
        start: datetime,
        end: datetime,
        total_months: int,
    ) -> List[TimeFold]:
        """Generate expanding window folds."""
        folds = []

        # Calculate how much the training window grows each fold
        remaining_months = total_months - self.train_months - self.val_months - self.gap_months
        growth_per_fold = remaining_months // max(1, self.n_folds) if self.n_folds > 0 else 0

        train_start = start

        for i in range(self.n_folds):
            # Training window expands each fold
            current_train_months = self.train_months + (i * growth_per_fold)
            train_end = train_start + relativedelta(months=current_train_months) - timedelta(days=1)

            val_start = train_end + timedelta(days=1) + relativedelta(months=self.gap_months)
            val_end = val_start + relativedelta(months=self.val_months) - timedelta(days=1)

            # Ensure we don't exceed the end date
            if val_end > end:
                val_end = end
                if val_start > val_end:
                    break

            folds.append(TimeFold(
                fold_id=i,
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                val_start=val_start.strftime("%Y-%m-%d"),
                val_end=val_end.strftime("%Y-%m-%d"),
            ))

        return folds

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime."""
        if date_str.lower() == "now":
            return datetime.now()
        return datetime.strptime(date_str, "%Y-%m-%d")

    def _months_between(self, start: datetime, end: datetime) -> int:
        """Calculate months between two dates."""
        return (end.year - start.year) * 12 + (end.month - start.month)

    def validate_data_coverage(
        self,
        folds: List[TimeFold],
        available_start: str,
        available_end: str,
    ) -> Tuple[bool, List[str]]:
        """
        Validate that folds are covered by available data.

        Returns:
            Tuple of (is_valid, list of warnings)
        """
        available_start_dt = self._parse_date(available_start)
        available_end_dt = self._parse_date(available_end)
        warnings = []

        for fold in folds:
            train_start = self._parse_date(fold.train_start)
            val_end = self._parse_date(fold.val_end)

            if train_start < available_start_dt:
                warnings.append(
                    f"Fold {fold.fold_id}: train_start ({fold.train_start}) "
                    f"is before available data ({available_start})"
                )

            if val_end > available_end_dt:
                warnings.append(
                    f"Fold {fold.fold_id}: val_end ({fold.val_end}) "
                    f"is after available data ({available_end})"
                )

        return len(warnings) == 0, warnings
