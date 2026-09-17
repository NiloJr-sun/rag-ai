"""Unit tests for answer generation (T1.10). Mock transport, no Ollama."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from app.rag.generation import (
    DEFAULT_NUM_CTX,
    GenerationError,
    build_context,
    generate_answer,
)
from app.storage.supabase import ChunkMatch

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _match(index: int, content: str, similarity: float = 0.9) -> ChunkMatch:
    return ChunkMatch(
        chunk_id=f"doc-{index:04d}",
        document_id="doc",
        chunk_index=index,
        content=content,
        source="data/samples/sample.txt",
        similarity=similarity,
    )


def _reply(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {"role": "assistant", "content": content},
            "prompt_eval_count": 85,
            "eval_count": 226,
        },
    )


def test_context_is_numbered_for_citation() -> None:
    context = build_context([_match(0, "first"), _match(1, "second")])
    assert context == "[1] first\n\n[2] second"


def test_answer_carries_text_sources_and_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _reply("  Cosine measures the angle [1].  ")

    matches = [_match(0, "cosine measures the angle")]
    with _client(handler) as client:
        answer = generate_answer("How does it work?", matches, client=client)

    assert answer.text == "Cosine measures the angle [1]."
    assert answer.sources == matches
    assert answer.prompt_tokens == 85
    assert answer.response_tokens == 226
    assert answer.seconds >= 0.0


def test_streaming_is_disabled() -> None:
    """Ollama streams NDJSON by default, which would break .json()."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = httpx.Response(200, content=request.content).json()
        seen["url"] = str(request.url)
        return _reply("ok")

    with _client(handler) as client:
        generate_answer("q", [_match(0, "c")], client=client)

    body = seen["body"]
    assert isinstance(body, dict)
    assert body["stream"] is False
    assert str(seen["url"]).endswith("/api/chat")
    assert [message["role"] for message in body["messages"]] == ["system", "user"]


def test_the_retrieved_text_reaches_the_prompt() -> None:
    """The model has no access to the corpus; the chunk must be in the prompt."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = httpx.Response(200, content=request.content).json()
        return _reply("ok")

    with _client(handler) as client:
        generate_answer("q", [_match(0, "a distinctive sentence")], client=client)

    body = seen["body"]
    assert isinstance(body, dict)
    assert "a distinctive sentence" in body["messages"][0]["content"]


def test_refusal_is_detected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _reply("Apologies, but I don't have that information.")

    with _client(handler) as client:
        answer = generate_answer("q", [], client=client)
    assert answer.is_refusal


def test_a_normal_answer_is_not_a_refusal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _reply("Cosine measures the angle [1].")

    with _client(handler) as client:
        answer = generate_answer("q", [_match(0, "c")], client=client)
    assert not answer.is_refusal


def test_no_context_still_produces_a_valid_prompt() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = httpx.Response(200, content=request.content).json()
        return _reply("Apologies, but I don't have that information.")

    with _client(handler) as client:
        generate_answer("q", [], client=client)

    body = seen["body"]
    assert isinstance(body, dict)
    assert "(no context found)" in body["messages"][0]["content"]


def test_missing_model_surfaces_ollamas_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404, json={"error": 'model "nope" not found, try pulling it first'}
        )

    with _client(handler) as client:
        with pytest.raises(GenerationError, match="try pulling it first"):
            generate_answer("q", [_match(0, "c")], client=client)


def test_connection_failure_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with _client(handler) as client, pytest.raises(GenerationError, match="failed"):
        generate_answer("q", [_match(0, "c")], client=client)


def test_empty_answer_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": ""}})

    with _client(handler) as client, pytest.raises(GenerationError, match="No answer"):
        generate_answer("q", [_match(0, "c")], client=client)


def test_empty_question_is_rejected() -> None:
    with pytest.raises(GenerationError, match="empty question"):
        generate_answer("   ", [_match(0, "c")])


def test_context_window_is_set_explicitly() -> None:
    """Ollama truncates silently, so the ceiling must not be left to chance."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = httpx.Response(200, content=request.content).json()
        return _reply("ok")

    with _client(handler) as client:
        generate_answer("q", [_match(0, "c")], client=client)

    body = seen["body"]
    assert isinstance(body, dict)
    assert body["options"]["num_ctx"] == DEFAULT_NUM_CTX


def test_a_null_error_field_does_not_render_as_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": None})

    with _client(handler) as client:
        with pytest.raises(GenerationError) as excinfo:
            generate_answer("q", [_match(0, "c")], client=client)
    assert "None" not in str(excinfo.value)


def test_a_non_dict_error_body_is_survivable() -> None:
    """A bare JSON string body must not raise AttributeError from the handler."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json="service unavailable")

    with _client(handler) as client:
        with pytest.raises(GenerationError, match="503"):
            generate_answer("q", [_match(0, "c")], client=client)
