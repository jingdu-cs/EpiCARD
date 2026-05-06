"""Shared helpers for evaluation modules."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def move_batch_to_device(
    batch: Any, device: torch.device,
) -> Any:
    """Recursively move all tensors in a (possibly nested) batch to *device*.

    Supports tensors, dicts, lists, and tuples. Non-tensor leaves are
    passed through unchanged (shallow-copied via container reconstruction
    only).
    """
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_batch_to_device(v, device) for v in batch)
    return batch


# Backwards-compatible private alias for call sites migrated from
# ``src.training.trainer._move_batch_to_device`` /
# ``src.evaluation.robustness_evaluator._move_batch_to_device``.
_move_batch_to_device = move_batch_to_device
