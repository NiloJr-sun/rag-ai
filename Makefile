# Shortcuts for the commands this project uses repeatedly.
#
# Recipes call .venv/bin/ directly rather than relying on an activated
# virtualenv, so every target works from a cold shell.

VENV    := .venv
PY      := $(VENV)/bin/python
PYTEST  := $(VENV)/bin/pytest
RUFF    := $(VENV)/bin/ruff
MYPY    := $(VENV)/bin/mypy
UVICORN := $(VENV)/bin/uvicorn

# Throwaway Postgres for the integration tests. Port 55432 so it cannot
# collide with a real Postgres on 5432.
EMBED_MODEL := nomic-embed-text
CHAT_MODEL  := qwen2
OLLAMA_URL  := http://localhost:11434

PG_CONTAINER := ragpg
PG_IMAGE     := pgvector/pgvector:pg16
TEST_DSN     := postgresql://postgres:test@127.0.0.1:55432/postgres

.DEFAULT_GOAL := help
.PHONY: help setup doctor venv models env dev test test-all check fmt lint types \
        ingest ask db-up db-down clean

help:  ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk -F':.*?## ' '{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: doctor venv models env db-up ingest  ## One command for a fresh clone
	@echo
	@echo "Setup complete. Next:"
	@echo "  make ask Q=\"how do I cook pasta?\"   ask a question"
	@echo "  make dev                             API on http://127.0.0.1:8000"

doctor:  ## Check that the tools this project needs are installed
	@missing=0; \
	for tool in uv docker ollama psql; do \
	  if command -v $$tool >/dev/null 2>&1; then \
	    echo "  ok       $$tool"; \
	  else \
	    echo "  MISSING  $$tool"; missing=1; \
	  fi; \
	done; \
	if curl -s -m 2 $(OLLAMA_URL)/api/tags >/dev/null 2>&1; then \
	  echo "  ok       ollama serving on $(OLLAMA_URL)"; \
	else \
	  echo "  MISSING  ollama is installed but not serving -- run: ollama serve"; \
	  missing=1; \
	fi; \
	if [ $$missing -ne 0 ]; then \
	  echo; \
	  echo "Install what is missing, then re-run. On macOS:"; \
	  echo "  brew install uv ollama libpq && brew install --cask docker"; \
	  exit 1; \
	fi

models:  ## Pull the Ollama models (skips any already present)
	@ollama list | grep -q "^$(EMBED_MODEL)" \
	  || ollama pull $(EMBED_MODEL)
	@ollama list | grep -q "^$(CHAT_MODEL)" \
	  || ollama pull $(CHAT_MODEL)
	@echo "models ready: $(EMBED_MODEL), $(CHAT_MODEL)"

env:  ## Create .env from the template, pointed at the local container
	@if [ -f .env ]; then \
	  echo ".env already exists, leaving it alone"; \
	else \
	  sed 's|^DATABASE_URL=$$|DATABASE_URL=$(TEST_DSN)|' .env.example > .env; \
	  echo "wrote .env pointing at the local container"; \
	  echo "  edit DATABASE_URL to use Supabase instead"; \
	fi

venv:  ## Create the virtualenv and install the backend with dev extras
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(PY) -e "backend[dev]"

dev:  ## Run the API with auto-reload on http://127.0.0.1:8000
	$(UVICORN) app.api.main:app --reload

test:  ## Unit tests; no database or Ollama needed
	$(PYTEST) backend/tests

test-all: db-up  ## Unit tests plus the storage integration tests
	TEST_DATABASE_URL=$(TEST_DSN) $(PYTEST) backend/tests

check: lint types test  ## Everything CI runs
	@echo "all checks passed"

lint:  ## ruff check and format --check
	$(RUFF) check backend
	$(RUFF) format --check backend

fmt:  ## Apply ruff formatting and autofixes
	$(RUFF) check --fix backend scripts
	$(RUFF) format backend scripts

types:  ## mypy
	$(MYPY) backend

ingest:  ## Ingest data/samples (override: make ingest ARGS="--reset")
	$(PY) scripts/ingest.py data/samples --max-size 500 $(ARGS)

ask:  ## Ask interactively (or: make ask Q="how do I cook pasta?")
	$(PY) scripts/ask.py $(Q)

db-up:  ## Start the throwaway pgvector container and apply the migration
	@docker start $(PG_CONTAINER) >/dev/null 2>&1 || \
	  docker run -d --rm --name $(PG_CONTAINER) \
	    -e POSTGRES_PASSWORD=test -p 55432:5432 $(PG_IMAGE) >/dev/null
	@for i in $$(seq 1 60); do \
	  docker exec $(PG_CONTAINER) psql -U postgres -h 127.0.0.1 -c 'select 1' \
	    >/dev/null 2>&1 && break; \
	  sleep 1; \
	done
	@docker exec -i $(PG_CONTAINER) psql -q -U postgres -h 127.0.0.1 \
	  -v ON_ERROR_STOP=1 -c 'set client_min_messages = warning' \
	  -f - < backend/migrations/0001_init.sql >/dev/null
	@echo "pgvector ready on $(TEST_DSN)"

db-down:  ## Stop the throwaway container
	-docker stop $(PG_CONTAINER) >/dev/null 2>&1
	@echo "stopped"

clean:  ## Remove caches
	find . -name __pycache__ -type d -not -path './.venv/*' -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
