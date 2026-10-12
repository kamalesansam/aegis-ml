"""Artifact integrity tests: safetensors roundtrip and hash verification."""

from __future__ import annotations

import json

import pytest
import torch

from aegis.artifacts import IntegrityError, load_verified, save_model, sha256_file


def test_save_and_load_roundtrip(cfg, bundle, trained_model, tmp_path):
    weights, manifest = save_model(
        trained_model,
        tmp_path,
        "m",
        params={"a": 1},
        metrics={"acc": 0.9},
        data_fingerprint=bundle.fingerprint,
    )
    assert weights.exists() and manifest.exists()
    loaded = load_verified(manifest)
    # identical predictions after a safetensors roundtrip
    x = bundle.x_test[:16]
    with torch.no_grad():
        a = trained_model(x)
        b = loaded(x)
    assert torch.allclose(a, b, atol=1e-6)


def test_manifest_records_hash(cfg, trained_model, tmp_path):
    weights, manifest = save_model(trained_model, tmp_path, "m")
    data = json.loads(manifest.read_text())
    assert data["sha256"] == sha256_file(weights)
    assert data["arch"]["class"] == "SmallCNN"


def test_tampered_weights_raise_integrity_error(cfg, trained_model, tmp_path):
    weights, manifest = save_model(trained_model, tmp_path, "m")
    # flip some bytes in the weights file
    raw = bytearray(weights.read_bytes())
    raw[-1] ^= 0xFF
    weights.write_bytes(bytes(raw))
    with pytest.raises(IntegrityError):
        load_verified(manifest)


def test_missing_weights_raise_integrity_error(cfg, trained_model, tmp_path):
    _weights, manifest = save_model(trained_model, tmp_path, "m")
    _weights.unlink()
    with pytest.raises(IntegrityError):
        load_verified(manifest)
