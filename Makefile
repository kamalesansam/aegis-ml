.PHONY: install lint fmt test security train pipeline report serve clean help
.DEFAULT_GOAL := help

export MLFLOW_DISABLE_AGENT_HINT := 1

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Create the environment from uv.lock
	uv sync

lint: ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Apply ruff formatting + autofixes
	uv run ruff format .
	uv run ruff check --fix .

test: ## Run the test suite with coverage
	uv run pytest -q --cov=aegis --cov-report=term-missing

security: ## Static security scans (Bandit SAST + pip-audit SCA)
	uv run bandit -r src -ll
	uv export --no-dev --no-emit-project --format requirements-txt > requirements.lock.txt
	uv run pip-audit -r requirements.lock.txt

train: ## Train and save a verified model artifact
	uv run aegis train

pipeline: ## Run the secure CT pipeline on the backdoor scenario (scan on)
	uv run aegis pipeline run --scenario backdoor

report: ## Run the full experiment matrix and write reports/
	uv run aegis report

serve: ## Serve the hardened API (set AEGIS_API_KEYS first)
	uv run uvicorn aegis.api.app:get_app --factory --host 0.0.0.0 --port 8000

clean: ## Remove generated ML outputs (keeps reports/)
	rm -rf mlruns artifacts quarantine requirements.lock.txt .pytest_cache .ruff_cache
