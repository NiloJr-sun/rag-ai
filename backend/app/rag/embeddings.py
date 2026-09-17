"""Text -> embedding vector, via a local Ollama model (T1.1).

Talks to Ollama's HTTP API directly instead of going through a client
library, so the request and response stay visible while learning.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import httpx

from app.rag.chunking import Chunk

# Defaults live here, not in a config file. Set the matching environment
# variable to override -- which is how Docker points at another host (T6.1)
# and how T3.7 swaps embedding models.
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "nomic-embed-text"

# The first call after a cold start includes loading the model into memory,
# which is far slower than the embedding itself.
DEFAULT_TIMEOUT_SECONDS = 60.0


class EmbeddingError(RuntimeError):
    """Ollama did not return a usable embedding."""


def _base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _model() -> str:
    return os.environ.get("OLLAMA_EMBEDDING_MODEL", DEFAULT_MODEL)


def ollama_error_detail(exc: httpx.HTTPStatusError) -> str:
    """Pull Ollama's own message out of an error response.

    A missing model answers 404 with {"error": "model ... not found, try
    pulling it first"}, which is far more useful than the status code alone.
    Shared with app.rag.generation, which hits the same API.
    """
    try:
        body = exc.response.json()
    except ValueError:
        body = None
    message = body.get("error") if isinstance(body, dict) else None
    return str(message or exc.response.text or exc)


def embed_text(text: str, *, client: httpx.Client | None = None) -> list[float]:
    """Return the embedding vector for ``text``.

    Pass ``client`` to reuse a connection across many calls, or to supply a
    mock transport in tests so they run without a live Ollama server.
    """
    if not text.strip():
        raise EmbeddingError("Cannot embed empty text")

    owned_client = client is None
    client = client or httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS)
    try:
        response = client.post(
            f"{_base_url()}/api/embeddings",
            json={"model": _model(), "prompt": text},
        )
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPStatusError as exc:
        raise EmbeddingError(
            f"Ollama returned {exc.response.status_code}: {ollama_error_detail(exc)}"
        ) from exc
    except httpx.HTTPError as exc:
        raise EmbeddingError(f"Request to Ollama failed: {exc}") from exc
    finally:
        if owned_client:
            client.close()

    # Defensive: a 200 with no vector in it is still unusable downstream.
    embedding = body.get("embedding")
    if not embedding:
        detail = body.get("error", body)
        raise EmbeddingError(f"No embedding returned for model {_model()!r}: {detail}")

    return [float(value) for value in embedding]


@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk paired with its vector, ready to store."""

    chunk: Chunk
    embedding: list[float]


@dataclass
class EmbeddingRun:
    """Result of embedding a batch, including what failed and how long it took."""

    embedded: list[EmbeddedChunk] = field(default_factory=list)
    failed: list[tuple[Chunk, str]] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def seconds_per_chunk(self) -> float:
        total = len(self.embedded) + len(self.failed)
        return self.seconds / total if total else 0.0


def embed_chunks(
    chunks: Sequence[Chunk],
    *,
    client: httpx.Client | None = None,
    skip_failures: bool = False,
) -> EmbeddingRun:
    """Embed every chunk, reusing one HTTP connection for the whole batch.

    With ``skip_failures`` a chunk that cannot be embedded is recorded in
    ``failed`` and the rest continue; otherwise the first failure propagates.
    Partial ingestion is usually better than none, but it is opt-in so a
    caller cannot silently store an incomplete document.
    """
    owned_client = client is None
    client = client or httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS)
    run = EmbeddingRun()
    started = time.perf_counter()
    try:
        for chunk in chunks:
            try:
                vector = embed_text(chunk.text, client=client)
            except EmbeddingError as exc:
                if not skip_failures:
                    raise
                run.failed.append((chunk, str(exc)))
                continue
            run.embedded.append(EmbeddedChunk(chunk=chunk, embedding=vector))
    finally:
        run.seconds = time.perf_counter() - started
        if owned_client:
            client.close()
    return run
