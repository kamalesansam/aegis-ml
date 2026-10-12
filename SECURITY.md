# Security Policy

Aegis-ML is an educational adversarial-ML defense lab. It is not production
software, but it is built to demonstrate secure engineering practices, so
security reports are welcome.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Instead:

1. Use GitHub's private vulnerability reporting (Security tab -> "Report a
   vulnerability") on this repository, or
2. Email the maintainer with a clear description and reproduction steps.

Include: affected file or endpoint, the impact, and a minimal reproduction.
You can expect an acknowledgement within a few days.

## Scope

In scope:

- The model server (`src/aegis/api/`): authentication, input validation, output
  hardening, rate limiting, extraction monitoring.
- The secure retraining pipeline (`src/aegis/pipeline/`): schema validation,
  poison scanning, model-integrity verification.
- Artifact integrity (`src/aegis/artifacts.py`): safetensors loading and the
  SHA-256 manifest check.
- Supply-chain configuration: pinned GitHub Actions, the locked dependency set,
  the container image.

Out of scope:

- The accuracy or robustness of the underlying classifier (it is a teaching
  model trained on `sklearn.load_digits`).
- Denial of service from unbounded local resource use.

## Handling of secrets

No secrets are committed. API keys are read from the `AEGIS_API_KEYS`
environment variable at runtime and are never baked into the image or logged.
Gitleaks runs in CI to catch accidental secret commits.
