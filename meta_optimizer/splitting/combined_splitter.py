"""
Combined time + coin splitting for maximum robustness validation.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

from .time_splitter import TimeSplitter, TimeFold
from .coin_splitter import CoinSplitter, CoinFold


@dataclass
class CombinedFold:
    """Represents a combined time + coin fold."""
    fold_id: int
    time_fold_id: int
    coin_fold_id: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    train_coins: List[str]
    val_coins: List[str]

    def __repr__(self) -> str:
        return (
            f"CombinedFold(id={self.fold_id}, "
            f"time={self.time_fold_id}, coin={self.coin_fold_id}, "
            f"train={self.train_start} to {self.train_end} on {len(self.train_coins)} coins, "
            f"val={self.val_start} to {self.val_end} on {len(self.val_coins)} coins)"
        )

    def to_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "time_fold_id": self.time_fold_id,
            "coin_fold_id": self.coin_fold_id,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "val_start": self.val_start,
            "val_end": self.val_end,
            "train_coins": self.train_coins,
            "val_coins": self.val_coins,
        }


class CombinedSplitter:
    """
    Generate combined time + coin folds.

    This creates the most rigorous cross-validation by holding out
    both time periods AND coins simultaneously.

    Modes:
    - "cartesian": Every combination of time fold x coin fold
    - "diagonal": Pair time fold i with coin fold i (requires equal counts)
    - "time_priority": Each time fold uses different coin validation set

    Example (cartesian, 2 time folds x 2 coin folds = 4 combined folds):
        Fold 0: Time fold 0 + Coin fold 0
        Fold 1: Time fold 0 + Coin fold 1
        Fold 2: Time fold 1 + Coin fold 0
        Fold 3: Time fold 1 + Coin fold 1
    """

    def __init__(
        self,
        time_splitter: TimeSplitter,
        coin_splitter: CoinSplitter,
        mode: str = "cartesian",
    ):
        self.time_splitter = time_splitter
        self.coin_splitter = coin_splitter
        self.mode = mode

        if mode not in ("cartesian", "diagonal", "time_priority"):
            raise ValueError(f"mode must be 'cartesian', 'diagonal', or 'time_priority', got '{mode}'")

    def generate_folds(
        self,
        start_date: str,
        end_date: str,
        coins: List[str],
        market_caps: Optional[Dict[str, float]] = None,
    ) -> List[CombinedFold]:
        """
        Generate combined folds.

        Args:
            start_date: Start date for time splits
            end_date: End date for time splits
            coins: List of coin symbols
            market_caps: Optional market cap data for stratification

        Returns:
            List of CombinedFold objects
        """
        # Generate component folds
        time_folds = self.time_splitter.generate_folds(start_date, end_date)
        coin_folds = self.coin_splitter.generate_folds(coins, market_caps)

        if self.mode == "cartesian":
            return self._generate_cartesian_folds(time_folds, coin_folds)
        elif self.mode == "diagonal":
            return self._generate_diagonal_folds(time_folds, coin_folds)
        else:
            return self._generate_time_priority_folds(time_folds, coin_folds)

    def _generate_cartesian_folds(
        self,
        time_folds: List[TimeFold],
        coin_folds: List[CoinFold],
    ) -> List[CombinedFold]:
        """Generate Cartesian product of time and coin folds."""
        combined = []
        fold_id = 0

        for tf in time_folds:
            for cf in coin_folds:
                combined.append(CombinedFold(
                    fold_id=fold_id,
                    time_fold_id=tf.fold_id,
                    coin_fold_id=cf.fold_id,
                    train_start=tf.train_start,
                    train_end=tf.train_end,
                    val_start=tf.val_start,
                    val_end=tf.val_end,
                    train_coins=cf.train_coins,
                    val_coins=cf.val_coins,
                ))
                fold_id += 1

        return combined

    def _generate_diagonal_folds(
        self,
        time_folds: List[TimeFold],
        coin_folds: List[CoinFold],
    ) -> List[CombinedFold]:
        """Generate diagonal pairing (requires equal fold counts)."""
        n_folds = min(len(time_folds), len(coin_folds))
        combined = []

        for i in range(n_folds):
            tf = time_folds[i]
            cf = coin_folds[i]

            combined.append(CombinedFold(
                fold_id=i,
                time_fold_id=tf.fold_id,
                coin_fold_id=cf.fold_id,
                train_start=tf.train_start,
                train_end=tf.train_end,
                val_start=tf.val_start,
                val_end=tf.val_end,
                train_coins=cf.train_coins,
                val_coins=cf.val_coins,
            ))

        return combined

    def _generate_time_priority_folds(
        self,
        time_folds: List[TimeFold],
        coin_folds: List[CoinFold],
    ) -> List[CombinedFold]:
        """
        Generate folds prioritizing time diversity.
        Each time fold is paired with a cycling coin fold.
        """
        combined = []
        n_coin_folds = len(coin_folds)

        for i, tf in enumerate(time_folds):
            cf = coin_folds[i % n_coin_folds]

            combined.append(CombinedFold(
                fold_id=i,
                time_fold_id=tf.fold_id,
                coin_fold_id=cf.fold_id,
                train_start=tf.train_start,
                train_end=tf.train_end,
                val_start=tf.val_start,
                val_end=tf.val_end,
                train_coins=cf.train_coins,
                val_coins=cf.val_coins,
            ))

        return combined

    @classmethod
    def from_config(
        cls,
        validation_scheme,
        mode: str = "cartesian",
    ) -> "CombinedSplitter":
        """
        Create CombinedSplitter from validation scheme config.

        Args:
            validation_scheme: ValidationScheme config object
            mode: Combination mode

        Returns:
            CombinedSplitter instance
        """
        time_cfg = validation_scheme.time_folds
        coin_cfg = validation_scheme.coin_folds

        time_splitter = TimeSplitter(
            n_folds=time_cfg.n_folds,
            train_months=time_cfg.train_months,
            val_months=time_cfg.val_months,
            mode=time_cfg.mode,
            gap_months=time_cfg.gap_months,
        )

        coin_splitter = CoinSplitter(
            n_folds=coin_cfg.n_folds,
            stratify_by=coin_cfg.stratify_by,
            min_coins_per_fold=coin_cfg.min_coins_per_fold,
        )

        return cls(time_splitter, coin_splitter, mode)


def create_splitter_from_config(validation_scheme):
    """
    Factory function to create appropriate splitter from config.

    Args:
        validation_scheme: ValidationScheme config object

    Returns:
        Splitter instance (TimeSplitter, CoinSplitter, or CombinedSplitter)
    """
    scheme_type = validation_scheme.type

    if scheme_type == "time_only":
        time_cfg = validation_scheme.time_folds
        return TimeSplitter(
            n_folds=time_cfg.n_folds,
            train_months=time_cfg.train_months,
            val_months=time_cfg.val_months,
            mode=time_cfg.mode,
            gap_months=time_cfg.gap_months,
        )

    elif scheme_type == "coin_only":
        coin_cfg = validation_scheme.coin_folds
        return CoinSplitter(
            n_folds=coin_cfg.n_folds,
            stratify_by=coin_cfg.stratify_by,
            min_coins_per_fold=coin_cfg.min_coins_per_fold,
        )

    elif scheme_type == "combined":
        return CombinedSplitter.from_config(validation_scheme)

    else:
        raise ValueError(f"Unknown validation scheme type: {scheme_type}")
