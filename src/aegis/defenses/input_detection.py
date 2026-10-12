"""Adversarial input detection via feature squeezing.

Xu et al., "Feature Squeezing: Detecting Adversarial Examples in Deep Neural
Networks" (NDSS 2018, arXiv:1704.01155).

Two squeezers are applied: bit-depth reduction and a 2x2 median filter. For each
squeezer we take the L1 distance between the softmax of the original input and
the softmax of the squeezed input. Adversarial inputs sit near a decision
boundary, so squeezing shifts their prediction more than it shifts a clean
input's.

On this coarse 8x8 data the two squeezers operate on very different scales, so we
normalize each squeezer's distance by its own clean-data quantile before taking
the max. A single final threshold on that normalized score is calibrated to a
target false-positive rate on clean validation data. The normalization keeps
both squeezers contributing instead of the noisier one dominating the max.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from ..config import Config
from ..models import SmallCNN


def bit_depth_reduce(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Reduce each pixel to ``bits`` bits of precision in [0,1]."""
    levels = float(2**bits - 1)
    return torch.round(x * levels) / levels


def median_filter_2d(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """Spatial median filter with replicate padding (kernel x kernel)."""
    if kernel <= 1:
        return x.clone()
    pad = kernel // 2
    xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    patches = xp.unfold(2, kernel, 1).unfold(3, kernel, 1)  # (N,C,H',W',k,k)
    patches = patches.contiguous().view(*patches.shape[:4], -1)
    med = patches.median(dim=-1).values
    return med[..., : x.size(2), : x.size(3)]


def _squeezers(cfg: Config):
    fs = cfg.defenses.input_detection.feature_squeezing
    bits = int(fs.bit_depth)
    ker = int(fs.median_kernel)
    return [
        lambda z: bit_depth_reduce(z, bits),
        lambda z: median_filter_2d(z, ker),
    ]


@torch.no_grad()
def per_squeezer_distances(model: SmallCNN, x: torch.Tensor, cfg: Config) -> list[np.ndarray]:
    """L1 softmax distance for each squeezer separately."""
    model.eval()
    base = F.softmax(model(x), dim=1)
    out = []
    for sq in _squeezers(cfg):
        p = F.softmax(model(sq(x)), dim=1)
        out.append((base - p).abs().sum(dim=1).cpu().numpy())
    return out


@dataclass
class SqueezeDetector:
    threshold: float
    scales: list[float] = field(default_factory=list)

    def scores(self, model: SmallCNN, x: torch.Tensor, cfg: Config) -> np.ndarray:
        dists = per_squeezer_distances(model, x, cfg)
        normed = [d / s for d, s in zip(dists, self.scales, strict=False)]
        return np.max(np.stack(normed, axis=0), axis=0)

    def flag(self, model: SmallCNN, x: torch.Tensor, cfg: Config) -> np.ndarray:
        return self.scores(model, x, cfg) > self.threshold

    def flag_rate(self, model: SmallCNN, x: torch.Tensor, cfg: Config) -> float:
        return float(self.flag(model, x, cfg).mean())


def calibrate_threshold(
    model: SmallCNN,
    x_clean_val: torch.Tensor,
    cfg: Config,
) -> SqueezeDetector:
    """Calibrate per-squeezer scales and a final threshold to a target FPR."""
    target_fpr = float(cfg.defenses.input_detection.feature_squeezing.target_fpr)
    dists = per_squeezer_distances(model, x_clean_val, cfg)
    scales = [max(float(np.quantile(d, 1.0 - target_fpr)), 1e-6) for d in dists]
    normed = [d / s for d, s in zip(dists, scales, strict=False)]
    score = np.max(np.stack(normed, axis=0), axis=0)
    thr = float(np.quantile(score, 1.0 - target_fpr))
    return SqueezeDetector(threshold=thr, scales=scales)
