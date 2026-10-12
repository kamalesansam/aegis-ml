"""Shared fixtures. A tiny model is trained once per session to keep tests fast."""

from __future__ import annotations

import pytest

from aegis.config import load_config, set_seed
from aegis.data import make_bundle
from aegis.models import train_model


@pytest.fixture(scope="session")
def cfg():
    """Config with reduced epochs / folds so the whole suite runs in seconds."""
    c = load_config()
    c._data["model"]["epochs"] = 6
    c._data["defenses"]["poison_detection"]["label_noise"]["cv_folds"] = 3
    c._data["adv_training"]["steps"] = 3
    c._data["attacks"]["extraction"]["surrogate_epochs"] = 6
    c._data["attacks"]["evasion"]["pgd"]["steps"] = 10
    # The 6-epoch model under-trains for the production 0.30 PGD gate; these
    # tests check pipeline mechanics (gate enforcement, quarantine), so use a
    # threshold the fast model clears. The real gate lives in params.yaml.
    c._data["gates"]["min_pgd_robust_accuracy"] = 0.15
    return c


@pytest.fixture(scope="session")
def bundle(cfg):
    set_seed(int(cfg.seed))
    return make_bundle(cfg)


@pytest.fixture(scope="session")
def trained_model(cfg, bundle):
    set_seed(int(cfg.seed))
    return train_model(cfg, bundle.x_train, bundle.y_train).model
