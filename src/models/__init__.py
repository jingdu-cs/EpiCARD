# Graph-Free Dual-Branch Forecaster

from src.models.graph_free_model import (
    CaseEventBranch,
    GraphFreeDualBranchForecaster,
    LocationTemporalBranch,
    SharedPrivateFusionHead,
)


def build_model(
    cfg: dict, d_loc: int, num_locations: int,
) -> GraphFreeDualBranchForecaster:
    """Factory function to create GraphFreeDualBranchForecaster from config."""
    return GraphFreeDualBranchForecaster(
        cfg=cfg, d_loc=d_loc, num_locations=num_locations,
    )


__all__ = [
    "CaseEventBranch",
    "GraphFreeDualBranchForecaster",
    "LocationTemporalBranch",
    "SharedPrivateFusionHead",
    "build_model",
]
