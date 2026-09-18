"""Tests for the ingestion and ask endpoints (T2.2).

The database and Ollama are replaced by dependency overrides and stubs, so
these run in CI. What they check is the HTTP contract: validation, status
codes, and how upstream failures are reported.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import main
from app.api.deps import get_connection, get_http_client
from app.api.main import MAX_UPLOAD_BYTES, app
from app.pipeline import IngestResult
from app.rag.generation import Answer, GenerationError
from app.storage import supabase
from app.storage.supabase import ChunkMatch


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_connection] = lambda: None
    app.dependency_overrides[get_http_client] = lambda: None
    yield TestClient(app)
    app.dependency_overrides.clear()


def _match(index: int, content: str, similarity: float = 0.7) -> ChunkMatch:
    return ChunkMatch(
        chunk_id=f"doc-{index:04d}",
        document_id="doc",
        chunk_index=index,
        content=content,
        source="sample.txt",
        similarity=similarity,
    )


# --- uploads ----------------------------------------------------------------


def test_upload_stores_the_document(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_ingest(text: str, *, source: str, **kwargs: Any) -> IngestResult:
        seen["text"] = text
        seen["source"] = source
        return IngestResult(
            source=source,
            document_id="abc123",
            chunks_stored=3,
            chunks_failed=0,
            seconds=0.42,
        )

    monkeypatch.setattr(main, "ingest_text", fake_ingest)

    response = client.post(
        "/documents",
        files={"file": ("notes.md", b"Some prose. More prose.", "text/markdown")},
    )

    assert response.status_code == 201
    assert response.json() == {
        "source": "notes.md",
        "document_id": "abc123",
        "chunks_stored": 3,
        "chunks_failed": 0,
        "seconds": 0.42,
    }
    assert seen["text"] == "Some prose. More prose."
    # The filename is the identity, so re-uploading it replaces the document.
    assert seen["source"] == "notes.md"


def test_unsupported_extension_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/documents", files={"file": ("report.pdf", b"%PDF-1.4", "application/pdf")}
    )
    assert response.status_code == 415
    assert "T2.5" in response.json()["detail"]


def test_a_binary_file_wearing_a_txt_extension_is_rejected(
    client: TestClient,
) -> None:
    """Extensions are a claim: data/samples holds a .PNG that is really a JPEG."""
    jpeg_header = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01"
    response = client.post(
        "/documents", files={"file": ("photo.txt", jpeg_header, "text/plain")}
    )
    assert response.status_code == 415
    assert "not UTF-8" in response.json()["detail"]


def test_an_oversized_upload_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/documents",
        files={"file": ("big.txt", b"x" * (MAX_UPLOAD_BYTES + 1), "text/plain")},
    )
    assert response.status_code == 413


def test_an_empty_file_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/documents", files={"file": ("blank.txt", b"   \n  ", "text/plain")}
    )
    assert response.status_code == 422


def test_a_database_failure_is_reported_as_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise supabase.StorageError("could not connect")

    monkeypatch.setattr(main, "ingest_text", unavailable)

    response = client.post(
        "/documents", files={"file": ("notes.txt", b"prose", "text/plain")}
    )
    assert response.status_code == 503
    assert "could not connect" in response.json()["detail"]


# --- ask --------------------------------------------------------------------


def test_ask_returns_the_answer_and_its_sources(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_answer(question: str, **kwargs: Any) -> Answer:
        return Answer(
            text="Boil water first [1].",
            sources=[_match(4, "First, bring a large pot of salted water to a boil.")],
            seconds=2.0,
            prompt_tokens=559,
            response_tokens=61,
        )

    monkeypatch.setattr(main, "answer_question", fake_answer)

    response = client.post("/ask", json={"question": "How do I cook pasta?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Boil water first [1]."
    assert body["is_refusal"] is False
    assert body["prompt_tokens"] == 559
    assert len(body["sources"]) == 1
    assert body["sources"][0]["source"] == "sample.txt"
    assert body["sources"][0]["chunk_index"] == 4


def test_a_refusal_is_a_200_not_an_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declining for lack of context is the correct outcome, not a failure."""

    def refuse(question: str, **kwargs: Any) -> Answer:
        return Answer(
            text="Apologies, but I don't have that information.",
            sources=[],
            seconds=1.0,
            prompt_tokens=100,
            response_tokens=10,
        )

    monkeypatch.setattr(main, "answer_question", refuse)

    response = client.post("/ask", json={"question": "What is the capital of France?"})
    assert response.status_code == 200
    assert response.json()["is_refusal"] is True


def test_top_k_is_passed_through(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def capture(question: str, **kwargs: Any) -> Answer:
        seen.update(kwargs)
        return Answer(
            text="ok", sources=[], seconds=0.1, prompt_tokens=1, response_tokens=1
        )

    monkeypatch.setattr(main, "answer_question", capture)

    client.post("/ask", json={"question": "q", "top_k": 3})
    assert seen["top_k"] == 3


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "q", "top_k": 0},
        {"question": "q", "top_k": 500},
        {},
    ],
)
def test_invalid_ask_payloads_are_rejected(
    client: TestClient, payload: dict[str, Any]
) -> None:
    assert client.post("/ask", json=payload).status_code == 422


def test_an_ollama_failure_is_reported_as_bad_gateway(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise GenerationError("Request to Ollama failed: connection refused")

    monkeypatch.setattr(main, "answer_question", unavailable)

    response = client.post("/ask", json={"question": "q"})
    assert response.status_code == 502
    assert "Ollama" in response.json()["detail"]
