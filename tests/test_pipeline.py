"""Pipeline tests: schema validation and security gates per scenario."""

from __future__ import annotations

import pytest
import torch

from aegis.pipeline.secure_retrain import (
    GateFailure,
    SchemaError,
    build_batch,
    run_pipeline,
    validate_schema,
)


def test_schema_accepts_valid_batch():
    x = torch.rand(16, 1, 8, 8)
    y = torch.randint(0, 10, (16,))
    frame = validate_schema(x, y)
    assert frame.shape == (16, 65)  # 64 features + label


def test_schema_rejects_out_of_range():
    x = torch.rand(8, 1, 8, 8)
    x[0, 0, 0, 0] = 2.0
    with pytest.raises(SchemaError):
        validate_schema(x, torch.randint(0, 10, (8,)))


def test_schema_rejects_bad_label():
    x = torch.rand(8, 1, 8, 8)
    y = torch.randint(0, 10, (8,))
    y[0] = 42
    with pytest.raises(SchemaError):
        validate_schema(x, y)


def test_schema_rejects_duplicates():
    x = torch.rand(8, 1, 8, 8)
    x[1] = x[0]
    y = torch.randint(0, 10, (8,))
    y[1] = y[0]
    with pytest.raises(SchemaError):
        validate_schema(x, y)


def test_clean_scenario_passes_gates(cfg, tmp_path):
    # standard training is enough for clean + ASR gates; it fails only the PGD
    # robustness gate, so for this fast test we assert the pipeline runs and the
    # ASR/clean gates behave by using adversarial training.
    res = run_pipeline(
        cfg, "clean", scan=True, adversarial=True, log_mlflow=False, out_dir=tmp_path
    )
    assert res.passed
    assert res.metrics.backdoor_asr <= cfg.gates.max_backdoor_asr
    assert (tmp_path / "model_card_clean.md").exists()
    assert res.model_path.exists()


def test_backdoor_without_scan_fails_asr_gate(cfg, tmp_path):
    with pytest.raises(GateFailure) as exc:
        run_pipeline(
            cfg, "backdoor", scan=False, adversarial=True, log_mlflow=False, out_dir=tmp_path
        )
    assert any("backdoor_asr" in f for f in exc.value.failures)


def test_backdoor_with_scan_quarantines_poison(cfg, tmp_path):
    res = run_pipeline(
        cfg, "backdoor", scan=True, adversarial=True, log_mlflow=False, out_dir=tmp_path
    )
    assert res.passed
    assert res.quarantined > 0
    assert res.scan_report["detector_recall"] >= 0.7
    assert res.metrics.backdoor_asr <= cfg.gates.max_backdoor_asr


def test_build_batch_poisons_correctly(cfg, bundle):
    batch = build_batch(cfg, bundle, "backdoor")
    assert batch.poisoned_idx.size > 0
    clean = build_batch(cfg, bundle, "clean")
    assert clean.poisoned_idx.size == 0
