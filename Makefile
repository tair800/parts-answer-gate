# Everything a reviewer needs, in the order they would run it.
#
#   make setup      dependencies
#   make db         a local PostgreSQL with pgvector
#   make migrate    bring the schema up to head
#   make corpus     the synthetic trilingual corpus, from a committed seed
#   make index      embed and load it
#   make artifacts  the evidence the kill test grades
#   make test       the whole suite
#   make console    the Answer Gate Lab on http://127.0.0.1:8071
#
# `make evidence` is the whole chain and is what CI runs.

.PHONY: help setup db db-down migrate corpus determinism index artifacts artifacts-check test \
        fast lint types breaches console evidence screenshots clean

PY := .venv/Scripts/python.exe
ifeq ($(OS),)
PY := .venv/bin/python
endif

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## install dependencies into .venv
	uv sync

db: ## start PostgreSQL with pgvector on 127.0.0.1:15440
	docker compose up -d postgres

db-down: ## stop it
	docker compose down

migrate: ## bring the schema up to head
	$(PY) -m alembic upgrade head

corpus: ## generate the synthetic trilingual corpus from its committed seed
	$(PY) scripts/generate_corpus.py

determinism: ## build the corpus twice and diff every byte
	$(PY) scripts/generate_corpus.py --verify-determinism

index: ## embed the corpus and load it into pgvector
	$(PY) scripts/seed_index.py

artifacts: ## run the evaluation and write the evidence the kill test grades
	$(PY) scripts/build_artifacts.py

artifacts-check: ## fail if the committed evidence no longer matches a fresh build
	$(PY) scripts/check_artifacts_current.py

lint: ## ruff
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

types: ## mypy --strict
	$(PY) -m mypy src

fast: lint types ## everything that needs no infrastructure
	$(PY) -m pytest tests -q --ignore=tests/test_kill_criteria.py

test: ## the whole suite, kill criteria included
	$(PY) -m pytest tests -q

breaches: ## plant defects into every guarantee and check each is caught
	$(PY) scripts/plant_breaches.py

console: ## serve the Answer Gate Lab
	$(PY) -m uvicorn parts_answer_gate.api.app:app --port 8071 --reload

screenshots: ## capture the console's screens into docs/screenshots/
	$(PY) scripts/screenshots.py

evidence: corpus determinism migrate index artifacts test ## the full chain CI runs
	@echo
	@echo "evidence rebuilt and graded against the predeclared thresholds"

clean: ## remove generated fixtures and caches
	rm -rf data/generated .pytest_cache .mypy_cache .ruff_cache
