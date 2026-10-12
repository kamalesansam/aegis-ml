"""Typer CLI, exposed as the ``aegis`` command.

Subcommands:
    aegis train                       train + save a verified model artifact
    aegis attack evasion              FGSM/PGD accuracy vs epsilon
    aegis attack poison               run a poisoning attack + its detector
    aegis attack extract              run a model-extraction attack
    aegis pipeline run --scenario ..  run the secure continuous-training pipeline
    aegis report                      run the full experiment matrix + figures
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from .config import load_config, set_seed

app = typer.Typer(add_completion=False, help="Aegis-ML: Adversarial ML Defense Lab")
attack_app = typer.Typer(help="Run attacks")
pipeline_app = typer.Typer(help="Secure continuous-training pipeline")
app.add_typer(attack_app, name="attack")
app.add_typer(pipeline_app, name="pipeline")


@app.command()
def train(
    adversarial: bool = typer.Option(False, help="Use PGD adversarial training"),
    out: str = typer.Option(None, help="Output directory (default: artifacts/)"),
) -> None:
    """Train a SmallCNN on clean data and save weights + manifest + model card."""
    from .artifacts import save_model
    from .data import make_bundle
    from .models import evaluate, train_model

    cfg = load_config()
    set_seed(int(cfg.seed))
    bundle = make_bundle(cfg)
    result = train_model(cfg, bundle.x_train, bundle.y_train, adversarial=adversarial)
    acc = evaluate(result.model, bundle.x_test, bundle.y_test)
    out_dir = Path(out) if out else cfg.path("artifacts_dir")
    weights, manifest = save_model(
        result.model,
        out_dir,
        cfg.paths.model_name,
        params={"adversarial": adversarial, "epochs": cfg.model.epochs, "seed": cfg.seed},
        metrics={"clean_accuracy": acc},
        data_fingerprint=bundle.fingerprint,
    )
    typer.echo(f"trained clean_accuracy={acc:.4f}")
    typer.echo(f"wrote {weights}")
    typer.echo(f"wrote {manifest}")


@attack_app.command("evasion")
def attack_evasion() -> None:
    """Report FGSM and PGD accuracy across the configured epsilon grid."""
    from .attacks.evasion import fgsm, linf_norm, pgd
    from .data import make_bundle
    from .models import evaluate, train_model

    cfg = load_config()
    set_seed(int(cfg.seed))
    bundle = make_bundle(cfg)
    model = train_model(cfg, bundle.x_train, bundle.y_train).model
    typer.echo(f"clean accuracy: {evaluate(model, bundle.x_test, bundle.y_test):.4f}")
    for eps in cfg.attacks.evasion.eps_grid:
        xa = fgsm(model, bundle.x_test, bundle.y_test, eps)
        acc = evaluate(model, xa, bundle.y_test)
        typer.echo(f"FGSM eps={eps}: acc={acc:.4f} linf={linf_norm(xa, bundle.x_test):.4f}")
    p = cfg.attacks.evasion.pgd
    xa = pgd(model, bundle.x_test, bundle.y_test, p.eps, p.step_size, p.steps, p.random_start)
    typer.echo(
        f"PGD eps={p.eps}: acc={evaluate(model, xa, bundle.y_test):.4f} "
        f"linf={linf_norm(xa, bundle.x_test):.4f}"
    )


@attack_app.command("poison")
def attack_poison(
    kind: str = typer.Option("backdoor", help="label_flip or backdoor"),
) -> None:
    """Run a poisoning attack and score its detector."""
    import numpy as np

    from .attacks.poisoning import (
        attack_success_rate,
        badnets_backdoor,
        random_label_flip,
    )
    from .data import make_bundle
    from .defenses.poison_detection import (
        detect_backdoor_spectral,
        detect_label_noise,
        detection_scores,
    )
    from .models import build_model, evaluate, train_model, train_on_arrays

    cfg = load_config()
    set_seed(int(cfg.seed))
    bundle = make_bundle(cfg)

    if kind == "label_flip":
        lf = cfg.attacks.poisoning.label_flip
        res = random_label_flip(bundle.x_train, bundle.y_train, lf.rate, 10, int(cfg.seed))
        flagged = detect_label_noise(cfg, res.x, res.y)
        ds = detection_scores(flagged, res.poisoned_idx, len(res.y))
        typer.echo(f"label flip rate={lf.rate}: poisoned={len(res.poisoned_idx)}")
        typer.echo(f"detector precision={ds.precision:.3f} recall={ds.recall:.3f}")
    elif kind == "backdoor":
        bd = cfg.attacks.poisoning.backdoor
        res = badnets_backdoor(
            bundle.x_train,
            bundle.y_train,
            bd.poison_rate,
            bd.target_class,
            bd.patch_size,
            bd.patch_value,
            int(cfg.seed),
        )
        model = train_model(cfg, res.x, res.y).model
        asr = attack_success_rate(
            model, bundle.x_test, bundle.y_test, bd.target_class, bd.patch_size, bd.patch_value
        )
        scan = build_model(cfg, 10)
        train_on_arrays(cfg, scan, res.x, res.y)
        flagged = detect_backdoor_spectral(cfg, scan, res.x, res.y)
        ds = detection_scores(flagged, res.poisoned_idx, len(res.y))
        keep = np.setdiff1d(np.arange(len(res.y)), flagged)
        m2 = train_model(cfg, res.x[keep], res.y[keep]).model
        asr2 = attack_success_rate(
            m2, bundle.x_test, bundle.y_test, bd.target_class, bd.patch_size, bd.patch_value
        )
        typer.echo(f"backdoor poison_rate={bd.poison_rate}: ASR={asr:.3f}")
        typer.echo(f"spectral detector precision={ds.precision:.3f} recall={ds.recall:.3f}")
        typer.echo(
            f"ASR after filtering={asr2:.3f} clean_acc={evaluate(m2, bundle.x_test, bundle.y_test):.3f}"
        )
    else:
        typer.echo(f"unknown kind {kind!r}; use label_flip or backdoor", err=True)
        raise typer.Exit(2)


@attack_app.command("extract")
def attack_extract(
    budget: int = typer.Option(1000, help="Query budget"),
) -> None:
    """Run a Knockoff-style model-extraction attack against a local victim."""
    from .attacks.extraction import steal_model, victim_query_fn
    from .data import make_bundle
    from .models import train_model

    cfg = load_config()
    set_seed(int(cfg.seed))
    bundle = make_bundle(cfg)
    victim = train_model(cfg, bundle.x_train, bundle.y_train).model
    qfn = victim_query_fn(victim)
    res = steal_model(
        cfg,
        qfn,
        budget,
        bundle.x_test,
        bundle.y_test,
        pool_x=bundle.x_train,
        victim_for_fidelity=victim,
    )
    typer.echo(
        f"extraction budget={budget}: surrogate_acc={res.accuracy:.3f} fidelity={res.fidelity:.3f}"
    )


@pipeline_app.command("run")
def pipeline_run(
    scenario: str = typer.Option("clean", help="clean | label_flip | backdoor"),
    scan: bool = typer.Option(True, help="Enable the poison scan + quarantine"),
    adversarial: bool = typer.Option(True, help="Use PGD adversarial training"),
) -> None:
    """Run the secure continuous-training pipeline. Non-zero exit on gate failure."""
    from .pipeline.secure_retrain import GateFailure, run_pipeline

    cfg = load_config()
    try:
        run_pipeline(cfg, scenario, scan=scan, adversarial=adversarial)
    except GateFailure as exc:
        typer.echo(f"PIPELINE REJECTED: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def report(
    out: str = typer.Option(None, help="Reports directory (default: reports/)"),
) -> None:
    """Run the full experiment matrix and write results.json, report, figures."""
    from .report import generate_report

    cfg = load_config()
    out_dir = Path(out) if out else cfg.path("reports_dir")
    generate_report(cfg, out_dir)
    typer.echo(f"wrote report artifacts under {out_dir}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
