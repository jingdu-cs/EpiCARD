"""Point-only training losses for the graph-free forecaster."""

from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """Point Huber log1p loss + light shared/private decomposition regularizer."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        head_cfg = cfg["model"].get("fusion_horizon", {}).get("head", {})
        self.decomp_loss_weight: float = float(
            head_cfg.get("decomp_loss_weight", 1.0e-4)
        )
        logger.info(
            "CombinedLoss initialized: point-only, decomp_weight=%.6f",
            self.decomp_loss_weight,
        )

    def forward(
        self,
        model_output: dict,
        targets: Tensor,
        mask: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute point loss + optional decomposition regularizer."""
        loss_pred = self._point_huber_log1p(
            model_output["pred_log1p"], targets, mask,
        )

        loss_decomp = torch.tensor(0.0, device=loss_pred.device)
        if self.decomp_loss_weight > 0.0 and (
            "fusion_shared_case" in model_output
            and "fusion_private_case" in model_output
            and "fusion_shared_loc" in model_output
            and "fusion_private_loc" in model_output
        ):
            loss_decomp = self._shared_private_decomp_loss(
                model_output["fusion_shared_case"],
                model_output["fusion_private_case"],
                model_output["fusion_shared_loc"],
                model_output["fusion_private_loc"],
            )

        loss = loss_pred + self.decomp_loss_weight * loss_decomp
        return {
            "loss": loss,
            "loss_pred": loss_pred,
            "loss_decomp": loss_decomp,
        }

    @staticmethod
    def _shared_private_decomp_loss(
        case_shared: Tensor,
        case_private: Tensor,
        loc_shared: Tensor,
        loc_private: Tensor,
    ) -> Tensor:
        """Light cosine^2 regularizer for shared/private decomposition."""
        case_cos = torch.nn.functional.cosine_similarity(
            case_shared, case_private, dim=-1, eps=1e-8
        )
        loc_cos = torch.nn.functional.cosine_similarity(
            loc_shared, loc_private, dim=-1, eps=1e-8
        )
        return case_cos.square().mean() + loc_cos.square().mean()

    @staticmethod
    def _point_huber_log1p(
        pred_log1p: Tensor,
        y: Tensor,
        mask: Tensor,
    ) -> Tensor:
        target_log1p = torch.log1p(y.clamp(min=0.0))
        loss = torch.nn.functional.smooth_l1_loss(
            pred_log1p, target_log1p, reduction="none"
        )
        masked_loss = loss * mask.unsqueeze(1)
        H = y.shape[1]
        denom = mask.sum() * H
        return masked_loss.sum() / denom.clamp(min=1.0)
