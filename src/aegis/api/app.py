"""Hardened FastAPI model server.

Security controls:

* API-key auth via the ``x-api-key`` header, compared with
  ``secrets.compare_digest``. Keys come from the ``AEGIS_API_KEYS`` environment
  variable (never hardcoded).
* Strict request validation (pydantic): exactly 64 pixels (flat or 8x8 nested),
  every value in [0,1].
* Feature-squeezing adversarial flag on each prediction.
* Per-client PRADA monitor: once a client's query stream looks like model
  extraction, further requests get HTTP 429.
* Per-key rate limit.
* Output hardening: only the top-1 label and a confidence rounded to 2 decimals.
  The full probability vector is never returned (limits extraction signal).
* The model is loaded with ``load_verified`` (SHA-256 manifest check) at startup.
* Prometheus metrics at ``/metrics``.
"""

from __future__ import annotations

import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from pydantic import BaseModel, Field, field_validator

from ..artifacts import load_verified
from ..config import load_config
from ..data import make_bundle
from ..defenses.extraction_detection import from_config as prada_from_config
from ..defenses.input_detection import SqueezeDetector, calibrate_threshold
from ..models import SmallCNN, predict_proba

# Prometheus counters are process-global and registered once.
REQ = Counter("aegis_requests_total", "Total prediction requests")
ADV = Counter("aegis_adversarial_flags_total", "Requests flagged as adversarial")
EXT = Counter("aegis_extraction_alerts_total", "Clients flagged for extraction")
AUTH = Counter("aegis_auth_failures_total", "Authentication failures")


class PredictRequest(BaseModel):
    pixels: list = Field(
        ...,
        description="64 floats, or an 8x8 nested list. Every value in [0,1].",
    )

    @field_validator("pixels")
    @classmethod
    def _normalize(cls, v: list) -> list[float]:
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("pixels must be a non-empty list")
        if all(isinstance(row, list) for row in v):
            if len(v) != 8 or any(len(row) != 8 for row in v):
                raise ValueError("nested pixels must be an 8x8 grid")
            flat = [p for row in v for p in row]
        elif all(isinstance(p, (int, float)) and not isinstance(p, bool) for p in v):
            flat = list(v)
        else:
            raise ValueError("pixels must be 64 floats or an 8x8 nested list of floats")
        if len(flat) != 64:
            raise ValueError("expected exactly 64 pixel values")
        flat = [float(p) for p in flat]
        if any(p < 0.0 or p > 1.0 for p in flat):
            raise ValueError("pixel values must be within [0,1]")
        return flat


class PredictResponse(BaseModel):
    label: int
    confidence: float
    adversarial: bool
    extraction_alert: bool


def _load_keys() -> list[str]:
    raw = os.environ.get("AEGIS_API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def _check_key(provided: str | None, valid: list[str]) -> bool:
    if not provided or not valid:
        return False
    # constant-time comparison against every configured key
    return any(secrets.compare_digest(provided, k) for k in valid)


def _resolve_manifest(cfg) -> Path:
    env = os.environ.get("AEGIS_MODEL_MANIFEST")
    if env:
        return Path(env)
    return cfg.path("artifacts_dir") / f"{cfg.paths.model_name}.manifest.json"


def build_app(cfg=None) -> FastAPI:
    cfg = cfg or load_config()
    app = FastAPI(title="Aegis-ML Model Server", version="0.1.0")

    manifest = _resolve_manifest(cfg)
    if not manifest.exists():
        raise RuntimeError(f"model manifest not found at {manifest}. Train first: `aegis train`.")
    model: SmallCNN = load_verified(manifest)

    # calibrate the feature-squeezing detector on clean validation data
    bundle = make_bundle(cfg)
    detector: SqueezeDetector = calibrate_threshold(model, bundle.x_val, cfg)

    app.state.cfg = cfg
    app.state.model = model
    app.state.detector = detector
    app.state.prada = prada_from_config(cfg)
    app.state.keys = _load_keys()
    app.state.rate_limit = int(cfg.api.rate_limit_per_minute)
    app.state.conf_decimals = int(cfg.api.confidence_decimals)
    app.state.hits: dict[str, deque] = defaultdict(deque)

    def require_key(x_api_key: str | None = Header(default=None)) -> str:
        if not _check_key(x_api_key, app.state.keys):
            AUTH.inc()
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        return x_api_key

    def _rate_limited(key: str) -> bool:
        now = time.time()
        window = app.state.hits[key]
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= app.state.rate_limit:
            return True
        window.append(now)
        return False

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": cfg.paths.model_name}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict", response_model=PredictResponse)
    def predict_endpoint(
        req: PredictRequest,
        request: Request,
        api_key: str = Depends(require_key),
    ) -> PredictResponse:
        if _rate_limited(api_key):
            raise HTTPException(status_code=429, detail="rate limit exceeded")

        REQ.inc()
        x = torch.tensor(req.pixels, dtype=torch.float32).reshape(1, 1, 8, 8)

        # PRADA extraction monitor (per API key)
        flagged = app.state.prada.observe(api_key, x)
        if flagged:
            EXT.inc()
            raise HTTPException(
                status_code=429,
                detail="query pattern flagged as model extraction; client throttled",
            )

        # feature-squeezing adversarial detection
        is_adv = bool(app.state.detector.flag(model, x, cfg)[0])
        if is_adv:
            ADV.inc()

        proba = predict_proba(model, x)[0]
        label = int(torch.argmax(proba).item())
        conf = round(float(proba[label].item()), app.state.conf_decimals)
        return PredictResponse(
            label=label,
            confidence=conf,
            adversarial=is_adv,
            # A flagged client is rejected with 429 above, so this is False on a
            # 200 response; exposed for client-side awareness and completeness.
            extraction_alert=app.state.prada.is_flagged(api_key),
        )

    return app


def get_app() -> FastAPI:
    """Factory for uvicorn: `uvicorn aegis.api.app:get_app --factory`.

    Importing this module never builds the app (so it needs no trained model at
    import time, keeping test collection cheap). The app is built only when the
    factory is called.
    """
    return build_app()
