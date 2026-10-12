"""Secure continuous-training pipeline with security gates.

Flow for an incoming data batch (which may be poisoned):

    ingest batch (scenario: clean | label_flip | backdoor)
      -> schema validation (pandera: 64 features in [0,1], label 0..9,
         no NaNs, no duplicate rows)
      -> poison scan (confident-learning label noise + spectral signatures)
      -> quarantine flagged samples to quarantine/ with a JSON report
      -> train (adversarial by default)
      -> evaluate security gates:
             clean accuracy      >= gates.min_clean_accuracy
             PGD robust accuracy >= gates.min_pgd_robust_accuracy
             backdoor ASR        <= gates.max_backdoor_asr
      -> on pass: write safetensors weights + manifest + model card,
         log params/metrics/artifacts to MLflow (file store ./mlruns)
      -> on fail: report the failing gate(s) and exit non-zero.

Running the backdoor scenario with the scan disabled shows the ASR gate catching
a poisoned model; running it with the scan enabled shows the poison quarantined
and the gate passing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
import torch

from ..artifacts import save_model
from ..attacks.evasion import pgd
from ..attacks.poisoning import (
    attack_success_rate,
    badnets_backdoor,
    random_label_flip,
)
from ..config import Config, set_seed
from ..data import DataBundle, make_bundle
from ..defenses.poison_detection import detect_backdoor_spectral, detect_label_noise
from ..models import build_model, evaluate, train_model, train_on_arrays

SCENARIOS = ("clean", "label_flip", "backdoor")


class SchemaError(Exception):
    """Raised when an incoming batch fails schema validation."""


class GateFailure(Exception):
    """Raised when a trained model fails one or more security gates."""

    def __init__(self, failures: list[str]):
        self.failures = failures
        super().__init__("; ".join(failures))


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------
@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    poisoned_idx: np.ndarray
    scenario: str


def build_batch(cfg: Config, bundle: DataBundle, scenario: str) -> Batch:
    """Build the incoming training batch for a scenario from the train split."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    x, y = bundle.x_train, bundle.y_train
    if scenario == "clean":
        return Batch(x.clone(), y.clone(), np.array([], dtype=int), scenario)
    if scenario == "label_flip":
        lf = cfg.attacks.poisoning.label_flip
        res = random_label_flip(x, y, lf.rate, bundle.num_classes, int(cfg.seed))
        return Batch(res.x, res.y, res.poisoned_idx, scenario)
    bd = cfg.attacks.poisoning.backdoor
    res = badnets_backdoor(
        x, y, bd.poison_rate, bd.target_class, bd.patch_size, bd.patch_value, int(cfg.seed)
    )
    return Batch(res.x, res.y, res.poisoned_idx, scenario)


# ---------------------------------------------------------------------------
# Schema validation (pandera)
# ---------------------------------------------------------------------------
def _batch_to_frame(x: torch.Tensor, y: torch.Tensor) -> pd.DataFrame:
    flat = x.reshape(x.size(0), -1).cpu().numpy().astype(np.float64)
    cols = {f"f{i}": flat[:, i] for i in range(flat.shape[1])}
    cols["label"] = y.cpu().numpy().astype(np.int64)
    return pd.DataFrame(cols)


def _build_schema(n_features: int, num_classes: int) -> pa.DataFrameSchema:
    feature_cols = {
        f"f{i}": pa.Column(
            float,
            checks=[pa.Check.ge(0.0), pa.Check.le(1.0)],
            nullable=False,
        )
        for i in range(n_features)
    }
    feature_cols["label"] = pa.Column(
        int,
        checks=[pa.Check.ge(0), pa.Check.le(num_classes - 1)],
        nullable=False,
    )
    return pa.DataFrameSchema(feature_cols, strict=True, unique=None)


def validate_schema(x: torch.Tensor, y: torch.Tensor, num_classes: int = 10) -> pd.DataFrame:
    """Validate the batch. Raises ``SchemaError`` on any violation.

    Checks: exactly 64 features all in [0,1], integer labels in 0..9, no NaNs,
    and no duplicate rows (identical feature+label vectors).
    """
    frame = _batch_to_frame(x, y)
    n_features = x.reshape(x.size(0), -1).size(1)
    schema = _build_schema(n_features, num_classes)
    try:
        schema.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:  # pragma: no cover - exercised in tests
        raise SchemaError(str(exc.failure_cases.head(10))) from exc

    dup = frame.duplicated().sum()
    if dup > 0:
        raise SchemaError(f"{dup} duplicate rows in batch")
    return frame


# ---------------------------------------------------------------------------
# Poison scan + quarantine
# ---------------------------------------------------------------------------
@dataclass
class ScanResult:
    flagged: np.ndarray
    label_noise_idx: np.ndarray
    spectral_idx: np.ndarray


def scan_poison(cfg: Config, x: torch.Tensor, y: torch.Tensor, num_classes: int = 10) -> ScanResult:
    """Run both detectors and return the union of flagged indices."""
    label_idx = detect_label_noise(cfg, x, y, num_classes)
    scan_model = build_model(cfg, num_classes)
    train_on_arrays(cfg, scan_model, x, y)
    spectral_idx = detect_backdoor_spectral(cfg, scan_model, x, y, num_classes=num_classes)
    flagged = np.union1d(label_idx, spectral_idx).astype(int)
    return ScanResult(flagged=flagged, label_noise_idx=label_idx, spectral_idx=spectral_idx)


def quarantine(
    cfg: Config,
    batch: Batch,
    scan: ScanResult,
) -> dict[str, Any]:
    """Write flagged samples and a JSON report to the quarantine directory."""
    qdir = cfg.path("quarantine_dir")
    qdir.mkdir(parents=True, exist_ok=True)
    flagged = scan.flagged
    truth = {int(i) for i in batch.poisoned_idx}
    caught = [int(i) for i in flagged if int(i) in truth]

    if flagged.size:
        np.savez(
            qdir / f"quarantine_{batch.scenario}.npz",
            indices=flagged,
            x=batch.x[flagged].cpu().numpy(),
            y=batch.y[flagged].cpu().numpy(),
        )
    report = {
        "scenario": batch.scenario,
        "created_utc": datetime.now(UTC).isoformat(),
        "batch_size": int(batch.x.size(0)),
        "flagged_count": int(flagged.size),
        "label_noise_flags": int(scan.label_noise_idx.size),
        "spectral_flags": int(scan.spectral_idx.size),
        "ground_truth_poison": int(len(truth)),
        "poison_caught": int(len(caught)),
        "detector_recall": round(len(caught) / len(truth), 4) if truth else None,
        "detector_precision": round(len(caught) / flagged.size, 4) if flagged.size else None,
        "flagged_indices": [int(i) for i in flagged[:200]],
    }
    (qdir / f"report_{batch.scenario}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
@dataclass
class GateMetrics:
    clean_accuracy: float
    pgd_robust_accuracy: float
    backdoor_asr: float
    passed: bool = False
    failures: list[str] = field(default_factory=list)


def evaluate_gates(cfg: Config, model, bundle: DataBundle) -> GateMetrics:
    g = cfg.gates
    bd = cfg.attacks.poisoning.backdoor
    p = cfg.attacks.evasion.pgd
    clean = evaluate(model, bundle.x_test, bundle.y_test)
    x_adv = pgd(model, bundle.x_test, bundle.y_test, p.eps, p.step_size, p.steps, p.random_start)
    robust = evaluate(model, x_adv, bundle.y_test)
    asr = attack_success_rate(
        model, bundle.x_test, bundle.y_test, bd.target_class, bd.patch_size, bd.patch_value
    )
    failures = []
    if clean < g.min_clean_accuracy:
        failures.append(f"clean_accuracy {clean:.3f} < {g.min_clean_accuracy}")
    if robust < g.min_pgd_robust_accuracy:
        failures.append(f"pgd_robust_accuracy {robust:.3f} < {g.min_pgd_robust_accuracy}")
    if asr > g.max_backdoor_asr:
        failures.append(f"backdoor_asr {asr:.3f} > {g.max_backdoor_asr}")
    return GateMetrics(
        clean_accuracy=clean,
        pgd_robust_accuracy=robust,
        backdoor_asr=asr,
        passed=not failures,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------
def write_model_card(
    cfg: Config,
    out_dir: Path,
    scenario: str,
    scan_enabled: bool,
    adversarial: bool,
    metrics: GateMetrics,
    scan_report: dict[str, Any] | None,
    data_fingerprint: str,
) -> Path:
    g = cfg.gates
    card = out_dir / f"model_card_{scenario}.md"
    lines = [
        f"# Model Card: aegis_cnn ({scenario})",
        "",
        f"Generated {datetime.now(UTC).isoformat()} by the Aegis-ML secure "
        "continuous-training pipeline.",
        "",
        "## Model",
        "- Architecture: SmallCNN (conv16-conv32-pool-fc64-fc10)",
        "- Task: 8x8 handwritten digit classification (scikit-learn load_digits)",
        f"- Training: {'PGD adversarial training' if adversarial else 'standard'}",
        "- Weights format: safetensors (no pickle; avoids CWE-502)",
        "",
        "## Provenance",
        f"- Scenario / incoming batch: {scenario}",
        f"- Poison scan: {'enabled' if scan_enabled else 'DISABLED'}",
        f"- Data fingerprint (SHA-256): `{data_fingerprint[:32]}...`",
    ]
    if scan_report:
        lines += [
            f"- Quarantined samples: {scan_report['flagged_count']} "
            f"(label-noise {scan_report['label_noise_flags']}, "
            f"spectral {scan_report['spectral_flags']})",
            f"- Known poison caught: {scan_report['poison_caught']} / "
            f"{scan_report['ground_truth_poison']}"
            + (
                f" (recall {scan_report['detector_recall']})"
                if scan_report["detector_recall"] is not None
                else ""
            ),
        ]
    lines += [
        "",
        "## Security gates",
        "| Gate | Threshold | Measured | Result |",
        "| --- | --- | --- | --- |",
        f"| Clean accuracy | >= {g.min_clean_accuracy} | {metrics.clean_accuracy:.3f} | "
        f"{'PASS' if metrics.clean_accuracy >= g.min_clean_accuracy else 'FAIL'} |",
        f"| PGD robust accuracy | >= {g.min_pgd_robust_accuracy} | "
        f"{metrics.pgd_robust_accuracy:.3f} | "
        f"{'PASS' if metrics.pgd_robust_accuracy >= g.min_pgd_robust_accuracy else 'FAIL'} |",
        f"| Backdoor ASR | <= {g.max_backdoor_asr} | {metrics.backdoor_asr:.3f} | "
        f"{'PASS' if metrics.backdoor_asr <= g.max_backdoor_asr else 'FAIL'} |",
        "",
        "## Intended use and limitations",
        "Research and education (DevSecMLOps coursework). Not a production model.",
        "Robustness is measured only against L-infinity PGD at the configured epsilon "
        "and a single BadNets trigger; it does not certify robustness to other attacks.",
        "",
    ]
    card.write_text("\n".join(lines), encoding="utf-8")
    return card


# ---------------------------------------------------------------------------
# MLflow logging (file store)
# ---------------------------------------------------------------------------
def _log_mlflow(cfg: Config, scenario: str, params: dict, metrics: dict, artifacts: list[Path]):
    """Log to a local file-based MLflow store. Never fails the pipeline."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    # Recent MLflow gates the file store behind an opt-in flag; local file
    # tracking is all this project needs.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import mlflow

        tracking = cfg.path("mlruns_dir")
        tracking.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking.as_uri())
        mlflow.set_experiment("aegis-secure-retrain")
        with mlflow.start_run(run_name=f"{scenario}-{datetime.now(UTC):%Y%m%d%H%M%S}"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            for art in artifacts:
                if art and Path(art).exists():
                    mlflow.log_artifact(str(art))
    except Exception as exc:  # pragma: no cover - logging must not break training
        print(f"[pipeline] mlflow logging skipped: {exc}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    scenario: str
    scan_enabled: bool
    adversarial: bool
    passed: bool
    metrics: GateMetrics
    scan_report: dict[str, Any] | None
    kept: int
    quarantined: int
    model_path: Path | None
    model_card: Path | None


def run_pipeline(
    cfg: Config,
    scenario: str,
    scan: bool = True,
    adversarial: bool = True,
    log_mlflow: bool = True,
    out_dir: str | Path | None = None,
) -> PipelineResult:
    """Run the secure retraining pipeline. Raises ``GateFailure`` on gate fail."""
    set_seed(int(cfg.seed))
    bundle = make_bundle(cfg)
    num_classes = bundle.num_classes
    batch = build_batch(cfg, bundle, scenario)

    print(
        f"[pipeline] scenario={scenario} scan={'on' if scan else 'OFF'} "
        f"adversarial={'on' if adversarial else 'off'}"
    )
    validate_schema(batch.x, batch.y, num_classes)
    print(f"[pipeline] schema OK: {batch.x.size(0)} rows, 64 features in [0,1]")

    scan_report = None
    keep_idx = np.arange(batch.x.size(0))
    if scan:
        scan_result = scan_poison(cfg, batch.x, batch.y, num_classes)
        scan_report = quarantine(cfg, batch, scan_result)
        keep_idx = np.setdiff1d(keep_idx, scan_result.flagged)
        print(
            f"[pipeline] scan flagged {scan_result.flagged.size} "
            f"(label-noise {scan_result.label_noise_idx.size}, "
            f"spectral {scan_result.spectral_idx.size}); "
            f"caught {scan_report['poison_caught']}/{scan_report['ground_truth_poison']} "
            f"known poison"
        )
    else:
        print("[pipeline] scan DISABLED (no quarantine)")

    x_keep = batch.x[keep_idx]
    y_keep = batch.y[keep_idx]
    result = train_model(cfg, x_keep, y_keep, num_classes=num_classes, adversarial=adversarial)
    model = result.model

    metrics = evaluate_gates(cfg, model, bundle)
    print(
        f"[pipeline] gates: clean={metrics.clean_accuracy:.3f} "
        f"pgd_robust={metrics.pgd_robust_accuracy:.3f} asr={metrics.backdoor_asr:.3f}"
    )

    out_dir = Path(out_dir) if out_dir else cfg.path("artifacts_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics.passed:
        print(f"[pipeline] GATE FAILURE: {metrics.failures}")
        # still write a (rejected) model card for the record
        write_model_card(
            cfg, out_dir, scenario, scan, adversarial, metrics, scan_report, bundle.fingerprint
        )
        raise GateFailure(metrics.failures)

    model_name = cfg.paths.model_name
    weights_path, manifest_path = save_model(
        model,
        out_dir,
        model_name,
        params={
            "scenario": scenario,
            "scan": scan,
            "adversarial": adversarial,
            "epochs": cfg.model.epochs,
            "seed": cfg.seed,
        },
        metrics={
            "clean_accuracy": metrics.clean_accuracy,
            "pgd_robust_accuracy": metrics.pgd_robust_accuracy,
            "backdoor_asr": metrics.backdoor_asr,
        },
        data_fingerprint=bundle.fingerprint,
    )
    card = write_model_card(
        cfg, out_dir, scenario, scan, adversarial, metrics, scan_report, bundle.fingerprint
    )
    print(f"[pipeline] PASSED. wrote {weights_path.name}, {manifest_path.name}, {card.name}")

    if log_mlflow:
        _log_mlflow(
            cfg,
            scenario,
            params={"scenario": scenario, "scan": scan, "adversarial": adversarial},
            metrics={
                "clean_accuracy": metrics.clean_accuracy,
                "pgd_robust_accuracy": metrics.pgd_robust_accuracy,
                "backdoor_asr": metrics.backdoor_asr,
                "quarantined": int(batch.x.size(0) - keep_idx.size),
            },
            artifacts=[manifest_path, card],
        )

    return PipelineResult(
        scenario=scenario,
        scan_enabled=scan,
        adversarial=adversarial,
        passed=True,
        metrics=metrics,
        scan_report=scan_report,
        kept=int(keep_idx.size),
        quarantined=int(batch.x.size(0) - keep_idx.size),
        model_path=weights_path,
        model_card=card,
    )
