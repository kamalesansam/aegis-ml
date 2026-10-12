"""Defense tests: squeezing, spectral signatures, label noise, PRADA."""

from __future__ import annotations

from aegis.attacks.evasion import pgd
from aegis.attacks.extraction import synthesize_queries
from aegis.attacks.poisoning import badnets_backdoor, random_label_flip
from aegis.defenses.extraction_detection import ClientMonitor
from aegis.defenses.input_detection import calibrate_threshold
from aegis.defenses.poison_detection import (
    detect_backdoor_spectral,
    detect_label_noise,
    detection_scores,
)
from aegis.models import build_model, train_on_arrays


def test_feature_squeezing_flags_adversarial_more_than_clean(cfg, bundle, trained_model):
    det = calibrate_threshold(trained_model, bundle.x_val, cfg)
    clean_rate = det.flag_rate(trained_model, bundle.x_test, cfg)
    p = cfg.attacks.evasion.pgd
    x_adv = pgd(
        trained_model, bundle.x_test, bundle.y_test, p.eps, p.step_size, p.steps, p.random_start
    )
    adv_rate = det.flag_rate(trained_model, x_adv, cfg)
    assert clean_rate <= 0.15  # calibrated near the 5% target FPR
    assert adv_rate > clean_rate + 0.15  # clearly higher on adversarial inputs


def test_spectral_backdoor_recall(cfg, bundle):
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
    scan = build_model(cfg, 10)
    train_on_arrays(cfg, scan, res.x, res.y)
    flagged = detect_backdoor_spectral(cfg, scan, res.x, res.y)
    ds = detection_scores(flagged, res.poisoned_idx, len(res.y))
    assert ds.recall >= 0.7


def test_spectral_no_false_alarm_on_clean(cfg, bundle):
    scan = build_model(cfg, 10)
    train_on_arrays(cfg, scan, bundle.x_train, bundle.y_train)
    flagged = detect_backdoor_spectral(cfg, scan, bundle.x_train, bundle.y_train)
    # balanced classes -> no suspect class -> nothing quarantined
    assert flagged.size == 0


def test_label_noise_detection(cfg, bundle):
    res = random_label_flip(bundle.x_train, bundle.y_train, 0.3, 10, int(cfg.seed))
    flagged = detect_label_noise(cfg, res.x, res.y)
    ds = detection_scores(flagged, res.poisoned_idx, len(res.y))
    assert ds.recall >= 0.8
    assert ds.precision >= 0.6


def test_prada_flags_extraction_not_benign(cfg, bundle):
    p = cfg.defenses.extraction_detection.prada
    atk = ClientMonitor(int(p.min_queries), float(p.shapiro_threshold), int(p.growth_window))
    q = synthesize_queries(cfg, 400, seed=7, pool_x=bundle.x_train)
    flagged = False
    for i in range(q.size(0)):
        if atk.observe(q[i : i + 1]):
            flagged = True
            break
    assert flagged, "PRADA should flag synthetic extraction queries"

    ben = ClientMonitor(int(p.min_queries), float(p.shapiro_threshold), int(p.growth_window))
    benign_flagged = False
    for i in range(bundle.x_test.size(0)):
        if ben.observe(bundle.x_test[i : i + 1]):
            benign_flagged = True
            break
    assert not benign_flagged, "PRADA must not flag benign real images"
