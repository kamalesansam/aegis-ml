"""Full experiment matrix: measures every number in the README and draws figures.

Writes:
    reports/results.json          all measured numbers
    reports/robustness_report.md  human-readable summary
    reports/figures/*.png         five figures (matplotlib, Agg backend)

Everything is seeded, so the numbers are reproducible.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .attacks.evasion import fgsm, pgd  # noqa: E402
from .attacks.extraction import (  # noqa: E402
    steal_model,
    synthesize_queries,
    victim_query_fn,
)
from .attacks.poisoning import (  # noqa: E402
    apply_trigger,
    attack_success_rate,
    badnets_backdoor,
    random_label_flip,
)
from .config import Config, set_seed  # noqa: E402
from .data import make_bundle  # noqa: E402
from .defenses.adversarial_training import train_adversarial  # noqa: E402
from .defenses.extraction_detection import ClientMonitor  # noqa: E402
from .defenses.input_detection import calibrate_threshold  # noqa: E402
from .defenses.poison_detection import (  # noqa: E402
    detect_backdoor_spectral,
    detect_label_noise,
    detection_scores,
)
from .models import (  # noqa: E402
    build_model,
    evaluate,
    predict,
    train_model,
    train_on_arrays,
)


def _evasion_matrix(cfg, bundle, std_model, adv_model) -> dict:
    out = {"eps_grid": list(cfg.attacks.evasion.eps_grid), "fgsm": {}, "pgd": {}}
    p = cfg.attacks.evasion.pgd
    for name, model in [("standard", std_model), ("adversarial", adv_model)]:
        out["fgsm"][name] = []
        out["pgd"][name] = []
        for eps in cfg.attacks.evasion.eps_grid:
            xf = fgsm(model, bundle.x_test, bundle.y_test, eps)
            out["fgsm"][name].append(round(evaluate(model, xf, bundle.y_test), 4))
            xp = pgd(model, bundle.x_test, bundle.y_test, eps, p.step_size, p.steps, p.random_start)
            out["pgd"][name].append(round(evaluate(model, xp, bundle.y_test), 4))
    return out


def _feature_squeezing(cfg, bundle, std_model) -> dict:
    det = calibrate_threshold(std_model, bundle.x_val, cfg)
    clean_fpr = det.flag_rate(std_model, bundle.x_test, cfg)
    p = cfg.attacks.evasion.pgd
    xf = fgsm(std_model, bundle.x_test, bundle.y_test, p.eps)
    xp = pgd(std_model, bundle.x_test, bundle.y_test, p.eps, p.step_size, p.steps, p.random_start)
    return {
        "eps": p.eps,
        "threshold": round(det.threshold, 4),
        "clean_fpr": round(clean_fpr, 4),
        "fgsm_detection_rate": round(det.flag_rate(std_model, xf, cfg), 4),
        "pgd_detection_rate": round(det.flag_rate(std_model, xp, cfg), 4),
    }


def _label_flip(cfg, bundle) -> dict:
    lf = cfg.attacks.poisoning.label_flip
    clean_model = train_model(cfg, bundle.x_train, bundle.y_train).model
    clean_acc = evaluate(clean_model, bundle.x_test, bundle.y_test)
    res = random_label_flip(bundle.x_train, bundle.y_train, lf.rate, 10, int(cfg.seed))
    pois_model = train_model(cfg, res.x, res.y).model
    pois_acc = evaluate(pois_model, bundle.x_test, bundle.y_test)
    flagged = detect_label_noise(cfg, res.x, res.y)
    ds = detection_scores(flagged, res.poisoned_idx, len(res.y))
    keep = np.setdiff1d(np.arange(len(res.y)), flagged)
    cleaned_model = train_model(cfg, res.x[keep], res.y[keep]).model
    cleaned_acc = evaluate(cleaned_model, bundle.x_test, bundle.y_test)
    return {
        "rate": lf.rate,
        "clean_accuracy": round(clean_acc, 4),
        "poisoned_accuracy": round(pois_acc, 4),
        "detector_precision": round(ds.precision, 4),
        "detector_recall": round(ds.recall, 4),
        "accuracy_after_removal": round(cleaned_acc, 4),
        "n_poison": int(len(res.poisoned_idx)),
        "n_flagged": int(ds.n_flagged),
    }


def _backdoor(cfg, bundle) -> dict:
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
    clean_acc = evaluate(model, bundle.x_test, bundle.y_test)
    asr = attack_success_rate(
        model, bundle.x_test, bundle.y_test, bd.target_class, bd.patch_size, bd.patch_value
    )
    scan = build_model(cfg, 10)
    train_on_arrays(cfg, scan, res.x, res.y)
    flagged = detect_backdoor_spectral(cfg, scan, res.x, res.y)
    ds = detection_scores(flagged, res.poisoned_idx, len(res.y))
    keep = np.setdiff1d(np.arange(len(res.y)), flagged)
    cleaned = train_model(cfg, res.x[keep], res.y[keep]).model
    asr_after = attack_success_rate(
        cleaned, bundle.x_test, bundle.y_test, bd.target_class, bd.patch_size, bd.patch_value
    )
    cleaned_acc = evaluate(cleaned, bundle.x_test, bundle.y_test)
    return {
        "poison_rate": bd.poison_rate,
        "target_class": bd.target_class,
        "clean_accuracy": round(clean_acc, 4),
        "asr": round(asr, 4),
        "detector_precision": round(ds.precision, 4),
        "detector_recall": round(ds.recall, 4),
        "asr_after_filtering": round(asr_after, 4),
        "accuracy_after_filtering": round(cleaned_acc, 4),
        "n_poison": int(len(res.poisoned_idx)),
        "n_flagged": int(ds.n_flagged),
    }


def _extraction(cfg, bundle, victim) -> dict:
    qfn = victim_query_fn(victim)
    budgets = list(cfg.attacks.extraction.query_budgets)
    rows = []
    for budget in budgets:
        res = steal_model(
            cfg,
            qfn,
            budget,
            bundle.x_test,
            bundle.y_test,
            pool_x=bundle.x_train,
            victim_for_fidelity=victim,
            seed=int(cfg.seed),
        )
        rows.append(
            {
                "budget": budget,
                "surrogate_accuracy": round(res.accuracy, 4),
                "fidelity": round(res.fidelity, 4),
            }
        )

    # PRADA queries-until-detection (attacker) and benign false positives
    p = cfg.defenses.extraction_detection.prada
    atk_mon = ClientMonitor(int(p.min_queries), float(p.shapiro_threshold), int(p.growth_window))
    attacker_q = synthesize_queries(cfg, 500, seed=7, pool_x=bundle.x_train)
    detect_at = None
    for i in range(attacker_q.size(0)):
        if atk_mon.observe(attacker_q[i : i + 1]) and detect_at is None:
            detect_at = i + 1
            break

    ben_mon = ClientMonitor(int(p.min_queries), float(p.shapiro_threshold), int(p.growth_window))
    benign_q = bundle.x_test
    benign_flagged = False
    for i in range(benign_q.size(0)):
        if ben_mon.observe(benign_q[i : i + 1]):
            benign_flagged = True
            break

    return {
        "budgets": rows,
        "prada_queries_to_detect_attacker": detect_at,
        "prada_benign_flagged": bool(benign_flagged),
        "victim_test_accuracy": round(evaluate(victim, bundle.x_test, bundle.y_test), 4),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _fig_adversarial_digits(cfg, bundle, model, fig_dir: Path) -> None:
    n = 6
    idx = np.arange(n)
    x = bundle.x_test[idx]
    y = bundle.y_test[idx]
    p = cfg.attacks.evasion.pgd
    xf = fgsm(model, x, y, p.eps)
    xp = pgd(model, x, y, p.eps, p.step_size, p.steps, p.random_start)
    rows = [("clean", x), (f"FGSM eps={p.eps}", xf), (f"PGD eps={p.eps}", xp)]
    fig, axes = plt.subplots(3, n, figsize=(1.6 * n, 5.2))
    for r, (name, data) in enumerate(rows):
        preds = predict(model, data)
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(data[c, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            color = "green" if int(preds[c]) == int(y[c]) else "red"
            ax.set_title(f"pred {int(preds[c])}", fontsize=9, color=color)
        axes[r, 0].set_ylabel(name, fontsize=10)
    fig.suptitle("Clean vs FGSM vs PGD (green=correct, red=fooled)", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "adversarial_examples.png", dpi=120)
    plt.close(fig)


def _fig_backdoor(cfg, bundle, fig_dir: Path) -> None:
    bd = cfg.attacks.poisoning.backdoor
    n = 6
    x = bundle.x_test[:n]
    xt = apply_trigger(x, bd.patch_size, bd.patch_value)
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.6))
    for c in range(n):
        axes[0, c].imshow(x[c, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, c].imshow(xt[c, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        for r in range(2):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    axes[0, 0].set_ylabel("clean", fontsize=10)
    axes[1, 0].set_ylabel("triggered", fontsize=10)
    fig.suptitle(
        f"BadNets trigger: {bd.patch_size}x{bd.patch_size} patch -> target class {bd.target_class}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "backdoor_trigger.png", dpi=120)
    plt.close(fig)


def _fig_eps_curve(cfg, evasion: dict, fig_dir: Path) -> None:
    eps = evasion["eps_grid"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eps, evasion["pgd"]["standard"], "o-", label="standard (PGD)")
    ax.plot(eps, evasion["pgd"]["adversarial"], "s-", label="adv-trained (PGD)")
    ax.plot(eps, evasion["fgsm"]["standard"], "o--", alpha=0.6, label="standard (FGSM)")
    ax.plot(eps, evasion["fgsm"]["adversarial"], "s--", alpha=0.6, label="adv-trained (FGSM)")
    ax.set_xlabel("L-infinity epsilon")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy under attack: standard vs adversarial training")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy_vs_epsilon.png", dpi=120)
    plt.close(fig)


def _fig_poison_detection(label_flip: dict, backdoor: dict, fig_dir: Path) -> None:
    labels = ["label-flip\n(confident learning)", "backdoor\n(spectral signatures)"]
    precision = [label_flip["detector_precision"], backdoor["detector_precision"]]
    recall = [label_flip["detector_recall"], backdoor["detector_recall"]]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, precision, w, label="precision")
    ax.bar(x + w / 2, recall, w, label="recall")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Poison detector precision and recall")
    for i, (pr, rc) in enumerate(zip(precision, recall, strict=False)):
        ax.text(i - w / 2, pr + 0.02, f"{pr:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, rc + 0.02, f"{rc:.2f}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "poison_detection.png", dpi=120)
    plt.close(fig)


def _fig_prada(cfg, bundle, fig_dir: Path) -> None:
    p = cfg.defenses.extraction_detection.prada
    atk_mon = ClientMonitor(int(p.min_queries), float(p.shapiro_threshold), int(p.growth_window))
    ben_mon = ClientMonitor(int(p.min_queries), float(p.shapiro_threshold), int(p.growth_window))
    attacker_q = synthesize_queries(cfg, 400, seed=7, pool_x=bundle.x_train)
    benign_q = bundle.x_test[:400]
    for i in range(attacker_q.size(0)):
        atk_mon.observe(attacker_q[i : i + 1])
    for i in range(min(400, benign_q.size(0))):
        ben_mon.observe(benign_q[i : i + 1])
    atk_w = atk_mon.w_history()
    ben_w = ben_mon.w_history()
    start = int(p.min_queries)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(start, start + len(atk_w)), atk_w, label="extraction attacker")
    ax.plot(range(start, start + len(ben_w)), ben_w, label="benign client")
    ax.axhline(
        float(p.shapiro_threshold),
        color="red",
        ls="--",
        label=f"flag threshold W={p.shapiro_threshold}",
    )
    ax.set_xlabel("queries observed")
    ax.set_ylabel("Shapiro-Wilk W of growing min-distances")
    ax.set_title("PRADA: extraction attacker vs benign client")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "prada_w_statistic.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate_report(cfg: Config, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg.seed))
    bundle = make_bundle(cfg)

    print("[report] training standard and adversarial models ...")
    std_model = train_model(cfg, bundle.x_train, bundle.y_train).model
    adv_model = train_adversarial(cfg, bundle.x_train, bundle.y_train)
    baseline_acc = evaluate(std_model, bundle.x_test, bundle.y_test)

    print("[report] evasion matrix ...")
    evasion = _evasion_matrix(cfg, bundle, std_model, adv_model)
    print("[report] feature squeezing ...")
    squeezing = _feature_squeezing(cfg, bundle, std_model)
    print("[report] label flip ...")
    label_flip = _label_flip(cfg, bundle)
    print("[report] backdoor ...")
    backdoor = _backdoor(cfg, bundle)
    print("[report] extraction + PRADA ...")
    extraction = _extraction(cfg, bundle, std_model)

    results = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "seed": int(cfg.seed),
        "dataset": {
            "name": "sklearn load_digits",
            "train": int(bundle.x_train.size(0)),
            "val": int(bundle.x_val.size(0)),
            "test": int(bundle.x_test.size(0)),
            "classes": bundle.num_classes,
        },
        "baseline_clean_accuracy": round(baseline_acc, 4),
        "adversarial_clean_accuracy": round(evaluate(adv_model, bundle.x_test, bundle.y_test), 4),
        "evasion": evasion,
        "feature_squeezing": squeezing,
        "label_flip": label_flip,
        "backdoor": backdoor,
        "extraction": extraction,
        "gates": {
            "min_clean_accuracy": cfg.gates.min_clean_accuracy,
            "min_pgd_robust_accuracy": cfg.gates.min_pgd_robust_accuracy,
            "max_backdoor_asr": cfg.gates.max_backdoor_asr,
        },
    }

    print("[report] figures ...")
    _fig_adversarial_digits(cfg, bundle, std_model, fig_dir)
    _fig_backdoor(cfg, bundle, fig_dir)
    _fig_eps_curve(cfg, evasion, fig_dir)
    _fig_poison_detection(label_flip, backdoor, fig_dir)
    _fig_prada(cfg, bundle, fig_dir)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_markdown(results, out_dir / "robustness_report.md")
    print(f"[report] done: {out_dir}/results.json")
    return results


def _write_markdown(r: dict, path: Path) -> None:
    ev = r["evasion"]
    eps = ev["eps_grid"]
    lf = r["label_flip"]
    bd = r["backdoor"]
    fs = r["feature_squeezing"]
    lines = [
        "# Aegis-ML Robustness Report",
        "",
        f"Generated {r['generated_utc']} (seed {r['seed']}).",
        "",
        f"Dataset: {r['dataset']['name']} - "
        f"{r['dataset']['train']} train / {r['dataset']['val']} val / "
        f"{r['dataset']['test']} test, {r['dataset']['classes']} classes.",
        "",
        "## Evasion (test accuracy under attack)",
        "",
        "| eps | " + " | ".join(str(e) for e in eps) + " |",
        "| --- | " + " | ".join("---" for _ in eps) + " |",
        "| FGSM standard | " + " | ".join(f"{v:.3f}" for v in ev["fgsm"]["standard"]) + " |",
        "| FGSM adv-trained | " + " | ".join(f"{v:.3f}" for v in ev["fgsm"]["adversarial"]) + " |",
        "| PGD standard | " + " | ".join(f"{v:.3f}" for v in ev["pgd"]["standard"]) + " |",
        "| PGD adv-trained | " + " | ".join(f"{v:.3f}" for v in ev["pgd"]["adversarial"]) + " |",
        "",
        f"Baseline clean accuracy: {r['baseline_clean_accuracy']:.3f} "
        f"(standard), {r['adversarial_clean_accuracy']:.3f} (adversarial).",
        "",
        "## Feature squeezing (adversarial input detection)",
        "",
        f"At eps={fs['eps']}, threshold calibrated to clean FPR "
        f"{fs['clean_fpr']:.3f}: FGSM detection {fs['fgsm_detection_rate']:.3f}, "
        f"PGD detection {fs['pgd_detection_rate']:.3f}.",
        "",
        "## Data poisoning",
        "",
        f"- Label flip (rate {lf['rate']}): accuracy {lf['clean_accuracy']:.3f} -> "
        f"{lf['poisoned_accuracy']:.3f} poisoned; detector precision "
        f"{lf['detector_precision']:.3f}, recall {lf['detector_recall']:.3f}; "
        f"accuracy after removal {lf['accuracy_after_removal']:.3f}.",
        f"- Backdoor (poison rate {bd['poison_rate']}): clean accuracy "
        f"{bd['clean_accuracy']:.3f}, ASR {bd['asr']:.3f}; spectral detector "
        f"precision {bd['detector_precision']:.3f}, recall {bd['detector_recall']:.3f}; "
        f"ASR after filtering {bd['asr_after_filtering']:.3f}.",
        "",
        "## Model extraction",
        "",
        "| query budget | surrogate accuracy | fidelity |",
        "| --- | --- | --- |",
    ]
    for row in r["extraction"]["budgets"]:
        lines.append(
            f"| {row['budget']} | {row['surrogate_accuracy']:.3f} | {row['fidelity']:.3f} |"
        )
    lines += [
        "",
        f"PRADA flags the extraction client after "
        f"{r['extraction']['prada_queries_to_detect_attacker']} queries; "
        f"benign client flagged: {r['extraction']['prada_benign_flagged']}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
