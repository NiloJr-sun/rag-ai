"""Tests for the FastAPI layer (T2.1).

TestClient drives the app in-process, so these need no running server. The
readiness check reaches for real dependencies, so those calls are stubbed.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app import config
from app.api.main import VERSION, app
from app.storage import supabase


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _flatten(body: dict[str, str]) -> str:
    return " ".join(f"{key}={value}" for key, value in body.items())


def test_health_is_cheap_and_does_not_touch_dependencies(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("liveness must not call out to anything")

    monkeypatch.setattr(httpx, "get", explode)
    monkeypatch.setattr(supabase, "connect", explode)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": VERSION}


def test_openapi_schema_is_served(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "rag-ai"
    assert "/health" in schema["paths"]


def test_docs_are_served(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200


def test_config_endpoint_never_leaks_the_connection_string(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:sup3rs3cret@host/db")
    body = client.get("/config").json()

    assert body["database_configured"] == "true"
    assert "sup3rs3cret" not in _flatten(body)
    assert body["chat_model"] == config.DEFAULT_CHAT_MODEL


def test_readiness_reports_a_missing_database_rather_than_failing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    def no_ollama(*args: Any, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", no_ollama)

    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is False
    assert body["checks"]["database"]["ok"] is False
    assert "DATABASE_URL" in body["checks"]["database"]["detail"]
    assert body["checks"]["ollama"]["ok"] is False


def test_readiness_reports_which_models_are_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    def tags(*args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={"models": [{"name": f"{config.DEFAULT_EMBEDDING_MODEL}:latest"}]},
            request=httpx.Request("GET", "http://localhost:11434/api/tags"),
        )

    monkeypatch.setattr(httpx, "get", tags)

    body = client.get("/health/ready").json()
    assert body["checks"]["ollama"]["ok"] is True
    assert body["checks"]["ollama"]["embedding_model_present"] is True
    assert body["checks"]["ollama"]["chat_model_present"] is False
