"""Unit tests for the embedding client.

These use a mock transport rather than a live Ollama server, so they run in
CI where no model is installed.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from app import config
from app.rag.embeddings import EmbeddingError, embed_text

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_returns_the_vector() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embedding": [0.1, -0.2, 0.3]})

    with _client(handler) as client:
        assert embed_text("hello", client=client) == [0.1, -0.2, 0.3]


def test_sends_model_and_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "some-other-model")
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = httpx.Response(200, content=request.content).json()
        return httpx.Response(200, json={"embedding": [1.0]})

    with _client(handler) as client:
        embed_text("some text", client=client)

    assert seen["url"] == f"{config.DEFAULT_OLLAMA_BASE_URL}/api/embeddings"
    assert seen["body"] == {"model": "some-other-model", "prompt": "some text"}


def test_trailing_slash_in_base_url_does_not_double_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.test:11434/")
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"embedding": [1.0]})

    with _client(handler) as client:
        embed_text("some text", client=client)

    assert seen["url"] == "http://example.test:11434/api/embeddings"


def test_unknown_model_surfaces_ollamas_message() -> None:
    """A missing model is a 404 whose body says how to fix it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404, json={"error": 'model "nope" not found, try pulling it first'}
        )

    with _client(handler) as client:
        with pytest.raises(EmbeddingError, match="try pulling it first"):
            embed_text("hello", client=client)


def test_non_json_error_body_falls_back_to_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with _client(handler) as client, pytest.raises(EmbeddingError, match="boom"):
        embed_text("hello", client=client)


def test_connection_failure_is_wrapped() -> None:
    """Ollama not running at all -- no response to read a status off."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with _client(handler) as client, pytest.raises(EmbeddingError, match="failed"):
        embed_text("hello", client=client)


def test_empty_text_is_rejected() -> None:
    with pytest.raises(EmbeddingError, match="empty"):
        embed_text("   ")
