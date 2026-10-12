"""Model extraction / stealing (Knockoff Nets style).

Orekondy et al., "Knockoff Nets: Stealing Functionality of Black-box Models"
(arXiv:1812.02766).

The attacker only has query access to a victim: a local callable or an HTTP
client that returns a predicted label for an image. It sends synthetic queries
(random noise plus augmented in-distribution images), labels them with the
victim's answers, trains a surrogate, and reports:

* fidelity: agreement between surrogate and victim on the test set,
* accuracy: surrogate accuracy against the true labels.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from ..config import Config
from ..models import SmallCNN, build_model, evaluate, predict

# A query function maps a batch of images (N,1,8,8) to predicted labels (N,).
QueryFn = Callable[[torch.Tensor], torch.Tensor]


def victim_query_fn(victim: SmallCNN) -> QueryFn:
    """Local black-box oracle: returns only hard labels, like a hardened API."""

    @torch.no_grad()
    def _fn(x: torch.Tensor) -> torch.Tensor:
        return predict(victim, x)

    return _fn


def http_query_fn(url: str, api_key: str, timeout: float = 10.0) -> QueryFn:
    """Remote oracle hitting the Aegis-ML ``/predict`` endpoint (hard labels only).

    Lets the extraction attack run against a live server instead of a local
    model. The server returns only the top-1 label, which is exactly the
    hard-label signal a surrogate is trained on.
    """
    import httpx

    client = httpx.Client(base_url=url, timeout=timeout)

    def _fn(x: torch.Tensor) -> torch.Tensor:
        labels = []
        for i in range(x.size(0)):
            payload = {"pixels": x[i].reshape(-1).tolist()}
            r = client.post("/predict", json=payload, headers={"x-api-key": api_key})
            r.raise_for_status()
            labels.append(int(r.json()["label"]))
        return torch.tensor(labels, dtype=torch.long)

    return _fn


def synthesize_queries(
    cfg: Config,
    n: int,
    seed: int,
    pool_x: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build ``n`` synthetic query images in [0,1].

    A configurable fraction is pure uniform noise; the rest are augmented samples
    drawn from a surrogate data pool (jitter + noise). This mirrors an attacker
    with no access to the exact training distribution.
    """
    rng = np.random.default_rng(seed)
    noise_fraction = cfg.attacks.extraction.noise_fraction
    n_noise = int(round(noise_fraction * n))
    n_pool = n - n_noise

    parts = []
    if n_noise > 0:
        parts.append(torch.rand(n_noise, 1, 8, 8))
    if n_pool > 0 and pool_x is not None and pool_x.size(0) > 0:
        idx = rng.integers(0, pool_x.size(0), size=n_pool)
        base = pool_x[idx].clone()
        base = base + torch.from_numpy(rng.normal(0, 0.15, size=base.shape).astype(np.float32))
        parts.append(torch.clamp(base, 0.0, 1.0))
    elif n_pool > 0:
        parts.append(torch.rand(n_pool, 1, 8, 8))

    queries = torch.cat(parts, dim=0)
    perm = rng.permutation(queries.size(0))
    return queries[perm]


@dataclass
class ExtractionResult:
    budget: int
    fidelity: float
    accuracy: float
    surrogate: SmallCNN


def steal_model(
    cfg: Config,
    query_fn: QueryFn,
    budget: int,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    pool_x: torch.Tensor | None = None,
    seed: int = 0,
    victim_for_fidelity: SmallCNN | None = None,
) -> ExtractionResult:
    """Run one extraction attack at a given query budget."""
    queries = synthesize_queries(cfg, budget, seed=seed, pool_x=pool_x)
    stolen_labels = query_fn(queries).long()

    surrogate = build_model(cfg, num_classes=10)
    opt = torch.optim.Adam(surrogate.parameters(), lr=cfg.model.lr)
    import torch.nn.functional as F  # noqa: N812

    surrogate.train()
    bs = cfg.model.batch_size
    g = torch.Generator().manual_seed(seed)
    for _ in range(int(cfg.attacks.extraction.surrogate_epochs)):
        perm = torch.randperm(queries.size(0), generator=g)
        for i in range(0, queries.size(0), bs):
            sel = perm[i : i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(surrogate(queries[sel]), stolen_labels[sel])
            loss.backward()
            opt.step()
    surrogate.eval()

    accuracy = evaluate(surrogate, x_test, y_test)
    if victim_for_fidelity is not None:
        victim_preds = predict(victim_for_fidelity, x_test)
    else:
        victim_preds = query_fn(x_test).long()
    sur_preds = predict(surrogate, x_test)
    fidelity = float((sur_preds == victim_preds).float().mean().item())

    return ExtractionResult(
        budget=budget, fidelity=fidelity, accuracy=accuracy, surrogate=surrogate
    )
