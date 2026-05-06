"""Global seed setting for reproducibility."""

import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Must be called before any data loading or model initialization.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
