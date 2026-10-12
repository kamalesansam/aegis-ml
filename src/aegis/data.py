"""Dataset loading and splitting.

Testbed: scikit-learn's bundled ``load_digits`` (8x8 grayscale, 10 classes,
1797 samples). No network download. Pixels are scaled to [0,1] by dividing by
the configured maximum (16). Splits are seeded and stratified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .config import Config


@dataclass
class DataBundle:
    """Train/val/test tensors plus a stable fingerprint of the raw data."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    fingerprint: str

    @property
    def num_classes(self) -> int:
        return int(torch.unique(self.y_test).numel())


def load_digits_arrays(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return images shaped (N,1,8,8) scaled to [0,1] and integer labels."""
    digits = load_digits()
    x = digits.images.astype(np.float32) / float(cfg.data.pixel_max)
    x = x.reshape(-1, 1, 8, 8)
    y = digits.target.astype(np.int64)
    return x, y


def fingerprint_arrays(x: np.ndarray, y: np.ndarray) -> str:
    """SHA-256 over the raw bytes of features and labels (data provenance)."""
    import hashlib

    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()


def make_bundle(cfg: Config) -> DataBundle:
    """Seeded, stratified train/val/test split of the digits dataset."""
    x, y = load_digits_arrays(cfg)
    seed = int(cfg.seed)

    x_tmp, x_test, y_tmp, y_test = train_test_split(
        x, y, test_size=cfg.data.test_size, random_state=seed, stratify=y
    )
    val_rel = cfg.data.val_size / (1.0 - cfg.data.test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp, y_tmp, test_size=val_rel, random_state=seed, stratify=y_tmp
    )
    return DataBundle(
        x_train=torch.from_numpy(x_train),
        y_train=torch.from_numpy(y_train),
        x_val=torch.from_numpy(x_val),
        y_val=torch.from_numpy(y_val),
        x_test=torch.from_numpy(x_test),
        y_test=torch.from_numpy(y_test),
        fingerprint=fingerprint_arrays(x, y),
    )


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
