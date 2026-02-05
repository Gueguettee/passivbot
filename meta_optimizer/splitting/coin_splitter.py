"""
Coin-based data splitting for cross-coin validation.
"""

import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class CoinFold:
    """Represents a single coin-based fold."""
    fold_id: int
    train_coins: List[str]
    val_coins: List[str]

    def __repr__(self) -> str:
        return (
            f"CoinFold(id={self.fold_id}, "
            f"train={len(self.train_coins)} coins, "
            f"val={len(self.val_coins)} coins)"
        )

    def to_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "train_coins": self.train_coins,
            "val_coins": self.val_coins,
        }


class CoinSplitter:
    """
    Generate coin-based folds for cross-coin validation.

    Supports stratification by:
    - market_cap: Group coins by market cap tiers, ensure each fold has diverse representation
    - random: Random shuffling
    - alphabetical: Deterministic split by alphabetical order

    Example (2 folds, 10 coins):
        Fold 1: Train [BTC, ETH, SOL, ADA, AVAX], Val [XRP, DOGE, SHIB, MATIC, OP]
        Fold 2: Train [XRP, DOGE, SHIB, MATIC, OP], Val [BTC, ETH, SOL, ADA, AVAX]
    """

    def __init__(
        self,
        n_folds: int = 2,
        stratify_by: str = "market_cap",
        min_coins_per_fold: int = 5,
        seed: Optional[int] = None,
    ):
        self.n_folds = n_folds
        self.stratify_by = stratify_by
        self.min_coins_per_fold = min_coins_per_fold
        self.seed = seed

        if stratify_by not in ("market_cap", "random", "alphabetical"):
            raise ValueError(
                f"stratify_by must be 'market_cap', 'random', or 'alphabetical', "
                f"got '{stratify_by}'"
            )

    def generate_folds(
        self,
        coins: List[str],
        market_caps: Optional[Dict[str, float]] = None,
    ) -> List[CoinFold]:
        """
        Generate coin folds.

        Args:
            coins: List of coin symbols
            market_caps: Optional dict mapping coin symbol to market cap
                         (required if stratify_by='market_cap')

        Returns:
            List of CoinFold objects
        """
        if len(coins) < self.n_folds * self.min_coins_per_fold:
            raise ValueError(
                f"Not enough coins ({len(coins)}) for {self.n_folds} folds "
                f"with minimum {self.min_coins_per_fold} coins per fold"
            )

        if self.stratify_by == "market_cap":
            return self._generate_stratified_folds(coins, market_caps)
        elif self.stratify_by == "random":
            return self._generate_random_folds(coins)
        else:
            return self._generate_alphabetical_folds(coins)

    def _generate_stratified_folds(
        self,
        coins: List[str],
        market_caps: Optional[Dict[str, float]] = None,
    ) -> List[CoinFold]:
        """Generate folds with market cap stratification."""
        if market_caps is None:
            # Fall back to alphabetical if no market caps provided
            return self._generate_alphabetical_folds(coins)

        # Sort coins by market cap
        sorted_coins = sorted(
            coins,
            key=lambda c: market_caps.get(c, 0),
            reverse=True
        )

        # Divide into tiers (large, medium, small)
        n_coins = len(sorted_coins)
        tier_size = n_coins // 3
        large_cap = sorted_coins[:tier_size]
        mid_cap = sorted_coins[tier_size:2*tier_size]
        small_cap = sorted_coins[2*tier_size:]

        # Generate folds ensuring each has representation from all tiers
        folds = []
        fold_assignments = {i: [] for i in range(self.n_folds)}

        # Distribute each tier across folds
        for tier in [large_cap, mid_cap, small_cap]:
            # Shuffle tier for randomness within tier
            tier_shuffled = tier.copy()
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(tier_shuffled)

            # Round-robin assignment to folds
            for i, coin in enumerate(tier_shuffled):
                fold_assignments[i % self.n_folds].append(coin)

        # Create CoinFold objects
        for fold_id in range(self.n_folds):
            val_coins = fold_assignments[fold_id]
            train_coins = [c for c in coins if c not in val_coins]

            folds.append(CoinFold(
                fold_id=fold_id,
                train_coins=train_coins,
                val_coins=val_coins,
            ))

        return folds

    def _generate_random_folds(self, coins: List[str]) -> List[CoinFold]:
        """Generate random folds."""
        shuffled = coins.copy()
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(shuffled)

        # Calculate fold size
        fold_size = len(shuffled) // self.n_folds

        folds = []
        for fold_id in range(self.n_folds):
            start_idx = fold_id * fold_size
            end_idx = start_idx + fold_size if fold_id < self.n_folds - 1 else len(shuffled)
            val_coins = shuffled[start_idx:end_idx]
            train_coins = [c for c in shuffled if c not in val_coins]

            folds.append(CoinFold(
                fold_id=fold_id,
                train_coins=train_coins,
                val_coins=val_coins,
            ))

        return folds

    def _generate_alphabetical_folds(self, coins: List[str]) -> List[CoinFold]:
        """Generate deterministic folds by alphabetical order."""
        sorted_coins = sorted(coins)
        fold_size = len(sorted_coins) // self.n_folds

        folds = []
        for fold_id in range(self.n_folds):
            start_idx = fold_id * fold_size
            end_idx = start_idx + fold_size if fold_id < self.n_folds - 1 else len(sorted_coins)
            val_coins = sorted_coins[start_idx:end_idx]
            train_coins = [c for c in sorted_coins if c not in val_coins]

            folds.append(CoinFold(
                fold_id=fold_id,
                train_coins=train_coins,
                val_coins=val_coins,
            ))

        return folds

    def get_leave_one_out_folds(self, coins: List[str]) -> List[CoinFold]:
        """
        Generate leave-one-out folds (one coin held out per fold).
        Useful for testing single-coin generalization.
        """
        folds = []
        for i, coin in enumerate(coins):
            folds.append(CoinFold(
                fold_id=i,
                train_coins=[c for c in coins if c != coin],
                val_coins=[coin],
            ))
        return folds

    def validate_coins(
        self,
        folds: List[CoinFold],
        available_coins: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Validate that all coins in folds are available.

        Returns:
            Tuple of (is_valid, list of warnings)
        """
        available_set = set(available_coins)
        warnings = []

        for fold in folds:
            for coin in fold.train_coins + fold.val_coins:
                if coin not in available_set:
                    warnings.append(f"Fold {fold.fold_id}: coin '{coin}' not in available coins")

        return len(warnings) == 0, warnings
