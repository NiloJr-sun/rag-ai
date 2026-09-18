"""FastAPI application (T2.1).

Structure: this module owns the app and its wiring only. Endpoints that do
real work call into app.pipeline, so the HTTP layer stays a thin adapter over
the same functions the CLI scripts use -- ingestion at T2.2, asking after it.

Run it with:
    uvicorn app.api.main:app --reload
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import FastAPI

from app import config
from app.storage import supabase

TITLE = "rag-ai"
VERSION = "0.1.0"

DESCRIPTION = """\
Question answering over documents, using local Ollama embeddings and a
Supabase/pgvector store.

Interactive docs are at `/docs`; the OpenAPI schema is at `/openapi.json`.
"""

app = FastAPI(title=TITLE, version=VERSION, description=DESCRIPTION)


@app.get("/health", tags=["health"], summary="Liveness")
def health() -> dict[str, str]:
    """Whether the process is up. Deliberately cheap: no dependency calls.

    Use this for a restart check. /health/ready is the one that tells you
    whether the service can actually answer a question.
    """
    return {"status": "ok", "version": VERSION}


@app.get("/health/ready", tags=["health"], summary="Readiness")
def ready() -> dict[str, Any]:
    """Whether the dependencies this service needs are reachable.

    Reports rather than raises: a 200 with database.ok false is more useful
    to a human than a 503 with no detail. Callers should read the payload.
    """
    checks: dict[str, Any] = {}

    dsn = config.database_url()
    if dsn is None:
        checks["database"] = {"ok": False, "detail": "DATABASE_URL is not set"}
    else:
        try:
            with supabase.connect(dsn) as conn:
                checks["database"] = {"ok": True, "chunks": supabase.count_chunks(conn)}
        except supabase.StorageError as exc:
            checks["database"] = {"ok": False, "detail": str(exc)}

    try:
        response = httpx.get(f"{config.ollama_base_url()}/api/tags", timeout=5.0)
        response.raise_for_status()
        names = [model["name"] for model in response.json().get("models", [])]
        checks["ollama"] = {
            "ok": True,
            "embedding_model_present": any(
                name.startswith(config.embedding_model()) for name in names
            ),
            "chat_model_present": any(
                name.startswith(config.chat_model()) for name in names
            ),
        }
    except httpx.HTTPError as exc:
        checks["ollama"] = {"ok": False, "detail": str(exc)}

    ready = all(check["ok"] for check in checks.values())
    return {"ready": ready, "checks": checks}


@app.get("/config", tags=["health"], summary="Effective settings")
def settings() -> dict[str, str]:
    """The settings actually in force, for diagnosing environment problems.

    Never includes DATABASE_URL: it carries a password. Only whether it is
    configured is reported.
    """
    return config.describe()
