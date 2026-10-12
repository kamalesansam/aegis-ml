"""Configuration loading.

All thresholds, epsilons, rates and seeds live in ``params.yaml``. This module
loads them into a lightweight attribute-accessible structure and seeds every
random number generator for reproducibility.
"""

from __future__ import annotations

import contextlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _find_params(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Locate params.yaml by walking up from the package or cwd."""
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("AEGIS_PARAMS")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in [Path.cwd(), *here.parents]:
        candidate = parent / "params.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("params.yaml not found; set AEGIS_PARAMS or run from project root")


class Config:
    """Dict-backed config with dotted attribute access and project paths."""

    def __init__(self, data: dict[str, Any], root: Path):
        self._data = data
        self.root = root

    def __getattr__(self, name: str) -> Any:
        # Never intercept private/dunder lookups; otherwise copy/pickle machinery
        # probing for __deepcopy__/__setstate__ triggers infinite recursion.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            data = object.__getattribute__(self, "_data")
            value = data[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        if isinstance(value, dict):
            return Config(value, object.__getattribute__(self, "root"))
        return value

    def __getitem__(self, name: str) -> Any:
        return self._data[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)

    def as_dict(self) -> dict[str, Any]:
        return self._data

    def path(self, key: str) -> Path:
        """Resolve a configured relative path against the project root."""
        return (self.root / self._data["paths"][key]).resolve()


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    params_path = _find_params(path)
    with open(params_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data, params_path.parent.resolve())


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch. Optionally force deterministic algorithms."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        # These keep CPU runs reproducible across machines without breaking
        # the small conv/pool ops this project uses.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        with contextlib.suppress(Exception):  # pragma: no cover - older torch
            torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
