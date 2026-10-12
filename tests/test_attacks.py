"""Attack tests: evasion bounds/efficacy and poisoning bookkeeping."""

from __future__ import annotations

import torch

from aegis.attacks.evasion import fgsm, linf_norm, pgd
from aegis.attacks.extraction import steal_model, synthesize_queries, victim_query_fn
from aegis.attacks.poisoning import (
    badnets_backdoor,
    random_label_flip,
    targeted_label_flip,
)
from aegis.models import evaluate


def test_fgsm_reduces_accuracy_and_respects_bounds(cfg, bundle, trained_model):
    x, y = bundle.x_test, bundle.y_test
    clean = evaluate(trained_model, x, y)
    eps = 0.2
    x_adv = fgsm(trained_model, x, y, eps)
    adv_acc = evaluate(trained_model, x_adv, y)
    assert adv_acc < clean
    # L-inf bound and valid pixel range
    assert linf_norm(x_adv, x) <= eps + 1e-5
    assert float(x_adv.min()) >= 0.0 and float(x_adv.max()) <= 1.0


def test_pgd_reduces_accuracy_more_than_fgsm(cfg, bundle, trained_model):
    x, y = bundle.x_test, bundle.y_test
    p = cfg.attacks.evasion.pgd
    x_pgd = pgd(trained_model, x, y, p.eps, p.step_size, p.steps, p.random_start)
    x_fgsm = fgsm(trained_model, x, y, p.eps)
    assert evaluate(trained_model, x_pgd, y) <= evaluate(trained_model, x_fgsm, y)
    assert linf_norm(x_pgd, x) <= p.eps + 1e-5
    assert float(x_pgd.min()) >= 0.0 and float(x_pgd.max()) <= 1.0


def test_random_label_flip_counts(cfg, bundle):
    x, y = bundle.x_train, bundle.y_train
    rate = 0.2
    res = random_label_flip(x, y, rate, 10, int(cfg.seed))
    expected = int(round(rate * y.size(0)))
    assert len(res.poisoned_idx) == expected
    # every flagged index actually changed label, others unchanged
    changed = (res.y != y).nonzero().flatten().numpy()
    assert set(changed.tolist()) == set(res.poisoned_idx.tolist())


def test_targeted_label_flip(cfg, bundle):
    x, y = bundle.x_train, bundle.y_train
    res = targeted_label_flip(x, y, source_class=1, target_class=7, rate=0.5, seed=int(cfg.seed))
    for i in res.poisoned_idx:
        assert int(y[i]) == 1
        assert int(res.y[i]) == 7


def test_badnets_counts_and_trigger(cfg, bundle):
    x, y = bundle.x_train, bundle.y_train
    bd = cfg.attacks.poisoning.backdoor
    res = badnets_backdoor(
        x, y, bd.poison_rate, bd.target_class, bd.patch_size, bd.patch_value, int(cfg.seed)
    )
    n_expected = int(round(bd.poison_rate * x.size(0)))
    assert len(res.poisoned_idx) == n_expected
    # all poisoned samples relabeled to the target and carry the bright patch
    for i in res.poisoned_idx[:20]:
        assert int(res.y[i]) == bd.target_class
        patch = res.x[i, 0, -bd.patch_size :, -bd.patch_size :]
        assert torch.allclose(patch, torch.full_like(patch, bd.patch_value))


def test_extraction_improves_with_budget(cfg, bundle, trained_model):
    qfn = victim_query_fn(trained_model)
    small = steal_model(
        cfg,
        qfn,
        200,
        bundle.x_test,
        bundle.y_test,
        pool_x=bundle.x_train,
        victim_for_fidelity=trained_model,
    )
    large = steal_model(
        cfg,
        qfn,
        800,
        bundle.x_test,
        bundle.y_test,
        pool_x=bundle.x_train,
        victim_for_fidelity=trained_model,
    )
    assert large.fidelity >= small.fidelity - 0.05
    assert large.accuracy > 0.5


def test_synthesize_queries_in_range(cfg, bundle):
    q = synthesize_queries(cfg, 64, seed=0, pool_x=bundle.x_train)
    assert q.shape == (64, 1, 8, 8)
    assert float(q.min()) >= 0.0 and float(q.max()) <= 1.0
