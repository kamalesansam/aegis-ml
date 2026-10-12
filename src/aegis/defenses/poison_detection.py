"""Poison detection.

(1) Label-noise detection via cross-validated out-of-fold predicted
probabilities (confident-learning style, implemented here with no cleanlab
dependency). A sample is flagged when the model's out-of-fold probability for
its *given* label falls below a per-class confidence threshold.

(2) Backdoor detection via spectral signatures (Tran et al., "Spectral
Signatures in Backdoor Attacks", NeurIPS 2018, arXiv:1811.00636). Per class,
penultimate activations are centered, projected onto their top singular vector,
and scored by the squared projection. The top-k% by score are flagged, where
k is set from the expected poison rate times an overcollect factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from ..config import Config
from ..models import SmallCNN, build_model, features_matrix, train_on_arrays


@dataclass
class DetectionScore:
    precision: float
    recall: float
    f1: float
    n_flagged: int
    n_true: int


def detection_scores(flagged: np.ndarray, truth: np.ndarray, n_total: int) -> DetectionScore:
    """Precision/recall of a flagged index set against ground-truth poison idx."""
    flagged_set = {int(i) for i in flagged}
    truth_set = {int(i) for i in truth}
    tp = len(flagged_set & truth_set)
    fp = len(flagged_set - truth_set)
    fn = len(truth_set - flagged_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return DetectionScore(precision, recall, f1, len(flagged_set), len(truth_set))


def oof_probabilities(
    cfg: Config,
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = 10,
) -> np.ndarray:
    """Cross-validated out-of-fold class probabilities for every sample."""
    import torch.nn.functional as F  # noqa: N812

    n = x.size(0)
    probs = np.zeros((n, num_classes), dtype=np.float32)
    folds = int(cfg.defenses.poison_detection.label_noise.cv_folds)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(cfg.seed))
    y_np = y.numpy()
    for train_idx, val_idx in skf.split(np.zeros(n), y_np):
        model = build_model(cfg, num_classes)
        train_on_arrays(cfg, model, x[train_idx], y[train_idx])
        model.eval()
        with torch.no_grad():
            p = F.softmax(model(x[val_idx]), dim=1).numpy()
        probs[val_idx] = p
    return probs


def detect_label_noise(
    cfg: Config,
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = 10,
    probs: np.ndarray | None = None,
) -> np.ndarray:
    """Flag samples whose given-label OOF confidence is below a per-class gate.

    The per-class threshold is the class mean self-confidence scaled by a margin
    (confident-learning intuition: mislabeled points sit well below the typical
    confidence of their claimed class).
    """
    if probs is None:
        probs = oof_probabilities(cfg, x, y, num_classes)
    y_np = y.numpy()
    self_conf = probs[np.arange(len(y_np)), y_np]
    margin = float(cfg.defenses.poison_detection.label_noise.margin)

    flagged = []
    for c in range(num_classes):
        cls_idx = np.where(y_np == c)[0]
        if len(cls_idx) == 0:
            continue
        thr = self_conf[cls_idx].mean() * margin
        for i in cls_idx:
            if self_conf[i] < thr:
                flagged.append(int(i))
    return np.sort(np.array(flagged, dtype=int))


def spectral_signature_scores(
    cfg: Config,
    model: SmallCNN,
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = 10,
) -> np.ndarray:
    """Per-sample spectral score = squared projection onto the class's top
    singular subspace of centered penultimate activations (Tran et al.).

    The original method uses the single top right singular vector. On this small
    model the trigger signal spans a few dimensions, so we project onto the top
    ``n_singular`` vectors (configurable) and sum the squared projections.
    """
    n_sv = int(cfg.defenses.poison_detection.spectral.n_singular)
    feats = features_matrix(model, x)  # (N, d)
    y_np = y.numpy()
    scores = np.zeros(len(y_np), dtype=np.float32)
    for c in range(num_classes):
        cls_idx = np.where(y_np == c)[0]
        if len(cls_idx) < 2:
            continue
        f = feats[cls_idx]
        centered = f - f.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        top = vt[: min(n_sv, vt.shape[0])]  # (k, d)
        proj = centered @ top.T  # (n_c, k)
        scores[cls_idx] = (proj**2).sum(axis=1).astype(np.float32)
    return scores


def detect_backdoor_spectral(
    cfg: Config,
    model: SmallCNN,
    x: torch.Tensor,
    y: torch.Tensor,
    expected_poison_rate: float | None = None,
    num_classes: int = 10,
) -> np.ndarray:
    """Spectral-signature backdoor detector with target-class gating.

    A label-remapping backdoor (BadNets) inflates the target class, because
    poisoned samples from other classes are relabeled into it. We therefore only
    clean classes whose size is anomalously large (> ``size_mult`` x median
    class size), estimate the number of injected samples as the size excess over
    the median, and quarantine the top ``overcollect`` x excess samples by
    spectral score within each such class. Balanced (clean) data triggers no
    suspect class, so nothing is quarantined.
    """
    sp = cfg.defenses.poison_detection.spectral
    size_mult = float(sp.size_mult)
    overcollect = float(sp.overcollect)
    scores = spectral_signature_scores(cfg, model, x, y, num_classes)
    y_np = y.numpy()
    sizes = np.array([(y_np == c).sum() for c in range(num_classes)])
    median_size = float(np.median(sizes))

    flagged: list[int] = []
    for c in range(num_classes):
        cls_idx = np.where(y_np == c)[0]
        if len(cls_idx) == 0 or sizes[c] <= size_mult * median_size:
            continue
        excess = max(1, int(round(sizes[c] - median_size)))
        k = min(len(cls_idx), int(round(overcollect * excess)))
        order = cls_idx[np.argsort(-scores[cls_idx])]
        flagged.extend(int(i) for i in order[:k])
    return np.sort(np.array(flagged, dtype=int))
