# Inference image for the Aegis-ML API.
# Base images are pinned by digest; Dependabot proposes updates.

# ---- builder: install deps (no dev) and train the model artifact ----
FROM python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89 AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.25@sha256:1e3808aa9023d0980e7c15b1fa7c1ac16ff35925780cf5c459858b2d693f01a9 /uv /bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# Resolve the environment first for better layer caching. Linux pulls CPU-only
# torch wheels via the pytorch-cpu index declared in pyproject.toml.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Install the project itself (non-editable), then train.
COPY src ./src
COPY params.yaml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Produce aegis_cnn.safetensors + manifest at build time (no committed binary).
# safetensors writes 0600 files; make them world-readable (still root-owned,
# so the runtime user can read but never modify them).
RUN /app/.venv/bin/aegis train && chmod -R a+rX,go-w /app/artifacts

# ---- runtime: slim, non-root, no build tooling ----
FROM python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89 AS runtime

LABEL org.opencontainers.image.title="aegis-ml" \
      org.opencontainers.image.source="https://github.com/kamalesansam/aegis-ml" \
      org.opencontainers.image.licenses="MIT"

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# No package managers at runtime: drop the base image's pip/setuptools/wheel
# (nothing to install malware with, fewer CVEs), then add a non-root user.
RUN python -m pip uninstall --yes --no-cache-dir pip setuptools wheel \
    && useradd --create-home --uid 10001 aegis

# Environment, config and the trained artifact stay root-owned and read-only
# for the app user: a compromised process cannot swap the model it serves.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/artifacts /app/artifacts
COPY params.yaml ./

USER 10001

EXPOSE 8000

# API keys are provided at runtime via AEGIS_API_KEYS (never baked into the image).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status != 200)"]

ENTRYPOINT ["uvicorn", "aegis.api.app:get_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
