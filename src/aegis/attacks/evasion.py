"""Evasion attacks: FGSM and PGD (L-infinity).

FGSM: Goodfellow et al., "Explaining and Harnessing Adversarial Examples"
(arXiv:1412.6572).
PGD: Madry et al., "Towards Deep Learning Models Resistant to Adversarial
Attacks" (arXiv:1706.06083).

Both produce inputs clamped to [0,1] with a guaranteed perturbation bound
||delta||_inf <= eps.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


def fgsm(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """One-step L-inf attack. Returns adversarial examples in [0,1]."""
    model.eval()
    x = x.clone().detach()
    x.requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    (grad,) = torch.autograd.grad(loss, x)
    with torch.no_grad():
        x_adv = x + eps * grad.sign()
        # enforce the eps ball and the valid pixel range
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
    return x_adv.detach()


def pgd(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    step_size: float,
    steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    """Iterative L-inf attack with projection onto the eps ball.

    The perturbation is projected to [-eps, eps] every step and the output is
    kept in [0,1], so ||x_adv - x||_inf <= eps always holds.
    """
    model.eval()
    x = x.clone().detach()
    x_adv = x.clone().detach()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        (grad,) = torch.autograd.grad(loss, x_adv)
        with torch.no_grad():
            x_adv = x_adv + step_size * grad.sign()
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()
    return x_adv.detach()


def linf_norm(x_adv: torch.Tensor, x: torch.Tensor) -> float:
    """Maximum absolute perturbation. Used to verify the eps bound in tests."""
    return float((x_adv - x).abs().max().item())
