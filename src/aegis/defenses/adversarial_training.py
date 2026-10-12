"""PGD adversarial training (defense).

Madry et al., arXiv:1706.06083. The core training loop lives in
``aegis.models.train_model`` with ``adversarial=True``; this module is the named
entry point and makes the intent explicit at call sites.
"""

from __future__ import annotations

import torch

from ..config import Config
from ..models import SmallCNN, train_model


def train_adversarial(
    cfg: Config,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int = 10,
) -> SmallCNN:
    """Train a SmallCNN with PGD adversarial training (robust optimization)."""
    result = train_model(cfg, x_train, y_train, num_classes=num_classes, adversarial=True)
    return result.model
