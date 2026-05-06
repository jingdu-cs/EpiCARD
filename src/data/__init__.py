"""HierEpiGNN data module — dataset loaders, graph builders, transforms."""

from src.data.aiv_dataset import AIVDataset
from src.data.covid_dataset import COVIDDataset
from src.data.graph_builder import GraphBuilder
from src.data.japan_dataset import JapanDataset
from src.data.transforms import SplitBuilder


def build_dataset(cfg: dict, dataset_name: str):
    """Factory function to create a dataset by name.

    Args:
        cfg: Full config dict (from configs/default.yaml).
        dataset_name: "aiv", "covid", or "japan".

    Returns:
        EpidemicDataset instance.
    """
    if dataset_name == "aiv":
        return AIVDataset(cfg)
    elif dataset_name == "covid":
        return COVIDDataset(cfg)
    elif dataset_name == "japan":
        return JapanDataset(cfg)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'aiv', 'covid', or 'japan'.")


__all__ = [
    "AIVDataset",
    "COVIDDataset",
    "GraphBuilder",
    "JapanDataset",
    "SplitBuilder",
    "build_dataset",
]
