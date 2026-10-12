"""API tests: auth, validation, hardened output, health, extraction blocking."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aegis.artifacts import save_model
from aegis.attacks.extraction import synthesize_queries
from aegis.config import load_config
from aegis.data import make_bundle

API_KEY = "test-key-abc123"


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    from aegis.models import train_model

    out = tmp_path_factory.mktemp("artifacts")
    cfg = load_config()
    cfg._data["model"]["epochs"] = 6
    cfg._data["api"]["rate_limit_per_minute"] = 100000  # don't let rate limit mask PRADA
    bundle = make_bundle(cfg)
    model = train_model(cfg, bundle.x_train, bundle.y_train).model
    _w, manifest = save_model(model, out, cfg.paths.model_name, data_fingerprint=bundle.fingerprint)

    import os

    os.environ["AEGIS_MODEL_MANIFEST"] = str(manifest)
    os.environ["AEGIS_API_KEYS"] = f"{API_KEY},second-key"

    from aegis.api.app import build_app

    app = build_app(cfg)
    return TestClient(app), bundle, cfg


def _valid_payload(bundle):
    img = bundle.x_test[0].reshape(-1).tolist()
    return {"pixels": img}


def test_health_ok(api):
    client, _bundle, _cfg = api
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_requires_api_key(api):
    client, bundle, _cfg = api
    r = client.post("/predict", json=_valid_payload(bundle))
    assert r.status_code == 401


def test_predict_rejects_wrong_key(api):
    client, bundle, _cfg = api
    r = client.post("/predict", json=_valid_payload(bundle), headers={"x-api-key": "nope"})
    assert r.status_code == 401


def test_predict_happy_path_hardened_output(api):
    client, bundle, _cfg = api
    r = client.post("/predict", json=_valid_payload(bundle), headers={"x-api-key": API_KEY})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"label", "confidence", "adversarial", "extraction_alert"}
    assert 0 <= body["label"] <= 9
    # confidence rounded to 2 decimals, no probability vector leaked
    assert round(body["confidence"], 2) == body["confidence"]
    assert "probabilities" not in body


def test_predict_accepts_nested_8x8(api):
    client, bundle, _cfg = api
    nested = bundle.x_test[1, 0].tolist()
    r = client.post("/predict", json={"pixels": nested}, headers={"x-api-key": API_KEY})
    assert r.status_code == 200


def test_predict_rejects_wrong_shape(api):
    client, _bundle, _cfg = api
    r = client.post("/predict", json={"pixels": [0.1, 0.2, 0.3]}, headers={"x-api-key": API_KEY})
    assert r.status_code == 422


def test_predict_rejects_out_of_range(api):
    client, bundle, _cfg = api
    bad = bundle.x_test[0].reshape(-1).tolist()
    bad[0] = 5.0
    r = client.post("/predict", json={"pixels": bad}, headers={"x-api-key": API_KEY})
    assert r.status_code == 422


def test_metrics_endpoint(api):
    client, _bundle, _cfg = api
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "aegis_requests_total" in r.text


def test_extraction_client_eventually_blocked(api):
    client, bundle, cfg = api
    queries = synthesize_queries(cfg, 400, seed=7, pool_x=bundle.x_train)
    blocked = False
    for i in range(queries.size(0)):
        payload = {"pixels": queries[i].reshape(-1).tolist()}
        r = client.post("/predict", json=payload, headers={"x-api-key": "second-key"})
        if r.status_code == 429 and "extraction" in r.json().get("detail", "").lower():
            blocked = True
            break
    assert blocked, "extraction query stream should eventually be blocked by PRADA"
