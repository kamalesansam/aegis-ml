"""Data poisoning attacks.

Label flipping: random at rate ``p`` and targeted source->target.
BadNets backdoor (Gu et al., arXiv:1708.06733): a small bright patch is stamped
on a fraction of training images, which are relabeled to a target class. At test
time the same trigger flips predictions to the target.

Each function returns the poisoned dataset and the ground-truth poisoned indices
so detectors can be scored against known labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PoisonResult:
    x: torch.Tensor
    y: torch.Tensor
    poisoned_idx: np.ndarray  # indices that were tampered with


def random_label_flip(
    x: torch.Tensor,
    y: torch.Tensor,
    rate: float,
    num_classes: int,
    seed: int,
) -> PoisonResult:
    """Flip ``rate`` of labels uniformly to a different random class."""
    rng = np.random.default_rng(seed)
    y_new = y.clone()
    n = y.size(0)
    n_flip = int(round(rate * n))
    idx = rng.choice(n, size=n_flip, replace=False)
    for i in idx:
        orig = int(y_new[i].item())
        choices = [c for c in range(num_classes) if c != orig]
        y_new[i] = int(rng.choice(choices))
    return PoisonResult(x=x.clone(), y=y_new, poisoned_idx=np.sort(idx))


def targeted_label_flip(
    x: torch.Tensor,
    y: torch.Tensor,
    source_class: int,
    target_class: int,
    rate: float,
    seed: int,
) -> PoisonResult:
    """Flip ``rate`` of ``source_class`` samples to ``target_class``."""
    rng = np.random.default_rng(seed)
    y_new = y.clone()
    src_idx = np.where(y.numpy() == source_class)[0]
    n_flip = int(round(rate * len(src_idx)))
    idx = rng.choice(src_idx, size=n_flip, replace=False) if n_flip > 0 else np.array([], int)
    for i in idx:
        y_new[i] = target_class
    return PoisonResult(x=x.clone(), y=y_new, poisoned_idx=np.sort(idx))


def _stamp_trigger(img: torch.Tensor, patch_size: int, patch_value: float) -> torch.Tensor:
    """Stamp a bright square patch in the bottom-right corner (in place-safe)."""
    out = img.clone()
    out[..., -patch_size:, -patch_size:] = patch_value
    return out


def apply_trigger(x: torch.Tensor, patch_size: int, patch_value: float) -> torch.Tensor:
    """Apply the backdoor trigger to a batch. Used to measure attack success."""
    out = x.clone()
    out[..., -patch_size:, -patch_size:] = patch_value
    return out


def badnets_backdoor(
    x: torch.Tensor,
    y: torch.Tensor,
    poison_rate: float,
    target_class: int,
    patch_size: int,
    patch_value: float,
    seed: int,
) -> PoisonResult:
    """Stamp the trigger on ``poison_rate`` of non-target images and relabel them
    to ``target_class``. Returns poisoned data and the poisoned indices.
    """
    rng = np.random.default_rng(seed)
    x_new = x.clone()
    y_new = y.clone()

    # Only poison images whose true label is not already the target, so the
    # trigger -> target association is learnable and measurable.
    eligible = np.where(y.numpy() != target_class)[0]
    n_poison = int(round(poison_rate * x.size(0)))
    n_poison = min(n_poison, len(eligible))
    idx = rng.choice(eligible, size=n_poison, replace=False)
    for i in idx:
        x_new[i] = _stamp_trigger(x_new[i], patch_size, patch_value)
        y_new[i] = target_class
    return PoisonResult(x=x_new, y=y_new, poisoned_idx=np.sort(idx))


def attack_success_rate(
    model,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    target_class: int,
    patch_size: int,
    patch_value: float,
) -> float:
    """Fraction of non-target test images that the trigger pushes to the target.

    Only images whose true label != target are counted (otherwise a correct
    prediction would inflate the success rate).
    """
    from ..models import predict

    mask = y_test != target_class
    x_eval = x_test[mask]
    if x_eval.size(0) == 0:
        return 0.0
    x_trig = apply_trigger(x_eval, patch_size, patch_value)
    preds = predict(model, x_trig)
    return float((preds == target_class).float().mean().item())
