"""Dataset splitting utilities."""

from __future__ import annotations

import logging

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SplitBuilder
# ---------------------------------------------------------------------------


class SplitBuilder:
    """Static methods that create train/val/test index splits.

    All methods return (train_indices, val_indices, test_indices) as
    lists of ints.
    """

    @staticmethod
    def temporal_split(
        dataset: Dataset,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ) -> tuple[list[int], list[int], list[int]]:
        """Split by time: earliest windows to train, middle to val, latest to test.

        No shuffling. Strict temporal ordering. Indices are consecutive slices.
        """
        n = len(dataset)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_indices = list(range(0, train_end))
        val_indices = list(range(train_end, val_end))
        test_indices = list(range(val_end, n))

        logger.info(
            "temporal_split: train=%d, val=%d, test=%d (total=%d)",
            len(train_indices),
            len(val_indices),
            len(test_indices),
            n,
        )
        return train_indices, val_indices, test_indices

    @staticmethod
    def spatio_temporal_holdout(
        dataset: Dataset,
        test_locations: list[tuple[str, str]],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ) -> tuple[list[int], list[int], list[int]]:
        """Index-level temporal split to combine with spatial subgraph holdout.

        This helper returns the same ``(train_idx, val_idx, test_idx)`` as
        ``temporal_split``; the spatial holdout is implemented by wrapping
        the dataset with a ``_SpatialSubgraphDataset`` (see
        ``src.evaluation.ood_evaluator``). The method exists so callers
        document the spatio-temporal intent explicitly.

        Args:
            dataset: An EpidemicDataset (or any Dataset with ``len()``).
            test_locations: List of (state, county) tuples to hold out. Used
                only for logging / documentation here — the actual location
                masking is done by the caller.
            train_ratio: Fraction of temporal windows for training.
            val_ratio: Fraction of temporal windows for validation.

        Returns:
            ``(train_idx, val_idx, test_idx)`` temporal splits.
        """
        train_idx, val_idx, test_idx = SplitBuilder.temporal_split(
            dataset, train_ratio=train_ratio, val_ratio=val_ratio,
        )
        logger.info(
            "spatio_temporal_holdout: %d test locations held out; "
            "temporal split sizes train=%d val=%d test=%d",
            len(test_locations),
            len(train_idx), len(val_idx), len(test_idx),
        )
        return train_idx, val_idx, test_idx

    @staticmethod
    def spatial_split(
        dataset: Dataset,
        test_locations: list[tuple[str, str]],
    ) -> tuple[list[int], list[int], list[int]]:
        """Split by location: hold out specified (state, county) tuples.

        For simplicity, returns all indices in each split. The caller is
        responsible for applying location-level masking to isolate the
        test_locations during evaluation.
        """
        n = len(dataset)
        all_indices = list(range(n))

        logger.info(
            "spatial_split: all %d indices returned for each split. "
            "Masking for %d test locations should be applied downstream.",
            n,
            len(test_locations),
        )
        return all_indices, all_indices, all_indices

    @staticmethod
    def random_split(
        dataset: Dataset,
        seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> tuple[list[int], list[int], list[int]]:
        """Standard random split using torch.Generator for reproducibility."""
        n = len(dataset)
        gen = torch.Generator()
        gen.manual_seed(seed)
        perm = torch.randperm(n, generator=gen)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_indices = perm[:train_end].tolist()
        val_indices = perm[train_end:val_end].tolist()
        test_indices = perm[val_end:].tolist()

        logger.info(
            "random_split (seed=%d): train=%d, val=%d, test=%d (total=%d)",
            seed,
            len(train_indices),
            len(val_indices),
            len(test_indices),
            n,
        )
        return train_indices, val_indices, test_indices

    @staticmethod
    def sliding_window_cv(
        dataset: Dataset,
        train_size: int,
        val_size: int,
        test_size: int,
        step_size: int,
        gap_size: int = 7,
    ) -> list[tuple[list[int], list[int], list[int]]]:
        """Generate sliding-window cross-validation folds.

        See previous docstring — unchanged behaviour.
        """
        n = len(dataset)
        block_size = train_size + 2 * gap_size + val_size + test_size

        if block_size > n:
            raise ValueError(
                f"Single fold requires {block_size} windows "
                f"(train={train_size} + 2*gap={2 * gap_size} + "
                f"val={val_size} + test={test_size}), "
                f"but dataset has only {n} windows."
            )

        folds: list[tuple[list[int], list[int], list[int]]] = []

        for offset in range(0, n - block_size + 1, step_size):
            train_idx = list(range(offset, offset + train_size))

            val_start = offset + train_size + gap_size
            val_idx = list(range(val_start, val_start + val_size))

            test_start = val_start + val_size + gap_size
            test_idx = list(range(test_start, test_start + test_size))

            folds.append((train_idx, val_idx, test_idx))

        logger.info(
            "sliding_window_cv: %d folds (train=%d, gap=%d, val=%d, "
            "gap=%d, test=%d, step=%d, total_windows=%d)",
            len(folds),
            train_size,
            gap_size,
            val_size,
            gap_size,
            test_size,
            step_size,
            n,
        )
        return folds
