"""Extraction detection via PRADA.

Juuti et al., "PRADA: Protecting Against DNN Model Stealing Attacks"
(EuroS&P 2019, arXiv:1805.02628).

For each client, PRADA tracks the minimum L2 distance of every new query to that
client's previous queries. Benign traffic (natural images) produces a roughly
normal distribution of these growing minimum distances; extraction traffic,
which deliberately explores the input space, produces a distribution that fails
a normality test. After at least ``min_queries``, a Shapiro-Wilk test is run on
the recent distance window and the client is flagged when W drops below a
threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.stats import shapiro


@dataclass
class ClientMonitor:
    """Per-client PRADA state."""

    min_queries: int = 50
    shapiro_threshold: float = 0.90
    growth_window: int = 100
    _queries: list[np.ndarray] = field(default_factory=list)
    _min_dists: list[float] = field(default_factory=list)
    flagged: bool = False
    last_w: float = 1.0

    def _flatten(self, x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy().reshape(x.shape[0], -1)

    def observe(self, x: torch.Tensor) -> bool:
        """Record a batch of queries; return True if the client is now flagged."""
        batch = self._flatten(x)
        for row in batch:
            if self._queries:
                prev = np.stack(self._queries)
                d = np.linalg.norm(prev - row, axis=1).min()
            else:
                d = 0.0
            self._queries.append(row)
            if len(self._queries) > 1:
                self._min_dists.append(float(d))

        if self.flagged:
            return True
        if len(self._min_dists) >= self.min_queries:
            window = np.array(self._min_dists[-self.growth_window :])
            # Shapiro needs some spread; a degenerate window is treated as benign.
            if np.std(window) < 1e-9:
                return self.flagged
            w, _ = shapiro(window)
            self.last_w = float(w)
            if w < self.shapiro_threshold:
                self.flagged = True
        return self.flagged

    def w_history(self) -> list[float]:
        """Recompute W after each query past ``min_queries`` (for plotting)."""
        ws = []
        for i in range(self.min_queries, len(self._min_dists) + 1):
            window = np.array(self._min_dists[max(0, i - self.growth_window) : i])
            if np.std(window) < 1e-9:
                ws.append(1.0)
                continue
            w, _ = shapiro(window)
            ws.append(float(w))
        return ws


@dataclass
class PradaMonitor:
    """Registry of per-client monitors keyed by client id (e.g. API key)."""

    min_queries: int = 50
    shapiro_threshold: float = 0.90
    growth_window: int = 100
    _clients: dict[str, ClientMonitor] = field(default_factory=dict)

    def _get(self, client_id: str) -> ClientMonitor:
        if client_id not in self._clients:
            self._clients[client_id] = ClientMonitor(
                min_queries=self.min_queries,
                shapiro_threshold=self.shapiro_threshold,
                growth_window=self.growth_window,
            )
        return self._clients[client_id]

    def observe(self, client_id: str, x: torch.Tensor) -> bool:
        return self._get(client_id).observe(x)

    def is_flagged(self, client_id: str) -> bool:
        return client_id in self._clients and self._clients[client_id].flagged


def from_config(cfg) -> PradaMonitor:
    p = cfg.defenses.extraction_detection.prada
    return PradaMonitor(
        min_queries=int(p.min_queries),
        shapiro_threshold=float(p.shapiro_threshold),
        growth_window=int(p.growth_window),
    )
