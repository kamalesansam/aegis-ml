"""Model serialization with integrity verification.

Weights are stored with **safetensors**, not pickle. ``torch.load`` /
``pickle.load`` deserialize arbitrary Python objects, so a tampered checkpoint
can run code on load (CWE-502: Deserialization of Untrusted Data). safetensors
is a pure-tensor format with no code execution path, which removes that class of
supply-chain risk for model files.

Every saved model gets a JSON manifest holding the SHA-256 of the weights file,
the params used, training/eval metrics and a data fingerprint. ``load_verified``
recomputes the hash and refuses to load on mismatch, raising ``IntegrityError``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .models import SmallCNN


class IntegrityError(Exception):
    """Raised when a model file's SHA-256 does not match its manifest."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Manifest:
    model_name: str
    weights_file: str
    sha256: str
    arch: dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    data_fingerprint: str = ""
    created_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "weights_file": self.weights_file,
            "sha256": self.sha256,
            "arch": self.arch,
            "params": self.params,
            "metrics": self.metrics,
            "data_fingerprint": self.data_fingerprint,
            "created_utc": self.created_utc,
        }


def _arch_of(model: SmallCNN) -> dict[str, Any]:
    return {
        "class": "SmallCNN",
        "conv1_out": model.conv1.out_channels,
        "conv2_out": model.conv2.out_channels,
        "fc_hidden": model.fc1.out_features,
        "num_classes": model.fc2.out_features,
    }


def save_model(
    model: SmallCNN,
    out_dir: str | Path,
    model_name: str,
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    data_fingerprint: str = "",
) -> tuple[Path, Path]:
    """Save weights (safetensors) + manifest (json). Returns (weights, manifest)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / f"{model_name}.safetensors"
    manifest_path = out_dir / f"{model_name}.manifest.json"

    state = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state, str(weights_path))

    manifest = Manifest(
        model_name=model_name,
        weights_file=weights_path.name,
        sha256=sha256_file(weights_path),
        arch=_arch_of(model),
        params=params or {},
        metrics=metrics or {},
        data_fingerprint=data_fingerprint,
        created_utc=datetime.now(UTC).isoformat(),
    )
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return weights_path, manifest_path


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def load_verified(manifest_path: str | Path) -> SmallCNN:
    """Load a model only if the weights file hash matches the manifest.

    Raises ``IntegrityError`` on any mismatch or missing file.
    """
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    weights_path = manifest_path.parent / manifest["weights_file"]
    if not weights_path.exists():
        raise IntegrityError(f"weights file missing: {weights_path}")

    actual = sha256_file(weights_path)
    expected = manifest["sha256"]
    if actual != expected:
        raise IntegrityError(
            f"SHA-256 mismatch for {weights_path.name}: "
            f"manifest={expected[:16]}... actual={actual[:16]}..."
        )

    arch = manifest["arch"]
    model = SmallCNN(
        conv1_out=arch["conv1_out"],
        conv2_out=arch["conv2_out"],
        fc_hidden=arch["fc_hidden"],
        num_classes=arch["num_classes"],
    )
    state = load_file(str(weights_path))
    model.load_state_dict(state)
    model.eval()
    return model


def load_verified_pair(weights_path: str | Path, manifest_path: str | Path) -> SmallCNN:
    """Verify an explicit weights path against a manifest's recorded hash."""
    manifest = load_manifest(manifest_path)
    actual = sha256_file(weights_path)
    if actual != manifest["sha256"]:
        raise IntegrityError("SHA-256 mismatch between weights and manifest")
    state = load_file(str(weights_path))
    arch = manifest["arch"]
    model = SmallCNN(
        conv1_out=arch["conv1_out"],
        conv2_out=arch["conv2_out"],
        fc_hidden=arch["fc_hidden"],
        num_classes=arch["num_classes"],
    )
    model.load_state_dict(state)
    model.eval()
    return model


def torch_state_roundtrip(model: SmallCNN) -> dict[str, torch.Tensor]:
    """Helper for tests: state dict snapshot."""
    return {k: v.clone() for k, v in model.state_dict().items()}
