"""Model-extraction demo against a running Aegis-ML API.

Streams Knockoff-style queries (noise + augmented images) to the live server and
reports when PRADA throttles the client with HTTP 429.

Usage:
    AEGIS_API_KEYS=demo-key uv run uvicorn aegis.api.app:get_app --factory &
    AEGIS_API_URL=http://127.0.0.1:8000 API_KEY=demo-key \
        uv run python scripts/extraction_demo.py
"""

from __future__ import annotations

import os
import sys

import httpx

from aegis.attacks.extraction import synthesize_queries
from aegis.config import load_config
from aegis.data import make_bundle


def main() -> int:
    url = os.environ.get("AEGIS_API_URL", "http://127.0.0.1:8000")
    api_key = os.environ.get("API_KEY", "demo-key")
    budget = int(os.environ.get("BUDGET", "400"))

    cfg = load_config()
    bundle = make_bundle(cfg)
    queries = synthesize_queries(cfg, budget, seed=7, pool_x=bundle.x_train)

    stolen = 0
    blocked_at = None
    with httpx.Client(base_url=url, timeout=10.0) as client:
        for i in range(queries.size(0)):
            payload = {"pixels": queries[i].reshape(-1).tolist()}
            r = client.post("/predict", json=payload, headers={"x-api-key": api_key})
            if r.status_code == 200:
                stolen += 1
            elif r.status_code == 429:
                blocked_at = i + 1
                print(f"blocked after {blocked_at} queries: {r.json().get('detail')}")
                break
            else:
                print(f"unexpected {r.status_code}: {r.text}")
                return 2

    print(f"successful queries before block: {stolen}")
    if blocked_at is None:
        print("NOT blocked (PRADA did not trigger within budget)")
        return 1
    print("PRADA defense worked: extraction client throttled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
