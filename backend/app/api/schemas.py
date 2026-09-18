"""Request and response bodies for the API (T2.2).

Kept apart from main.py so the HTTP contract is readable in one place, and
apart from the dataclasses in app.rag so an internal refactor does not
silently change what clients receive.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.pipeline import IngestResult
from app.rag.generation import Answer
from app.storage.supabase import ChunkMatch


class IngestResponse(BaseModel):
    source: str = Field(description="Identity the document is stored under")
    document_id: str
    chunks_stored: int
    chunks_failed: int
    seconds: float = Field(description="Time spent generating embeddings")

    @classmethod
    def from_result(cls, result: IngestResult) -> IngestResponse:
        return cls(
            source=result.source,
            document_id=result.document_id,
            chunks_stored=result.chunks_stored,
            chunks_failed=result.chunks_failed,
            seconds=round(result.seconds, 3),
        )


class Source(BaseModel):
    """One retrieved chunk, as cited in an answer."""

    source: str
    chunk_index: int
    chunk_id: str
    similarity: float = Field(
        description=(
            "1.0 is identical. There is no absolute relevance floor -- unrelated "
            "text still scores around 0.4 -- so compare these only against each "
            "other, within one answer."
        )
    )
    content: str

    @classmethod
    def from_match(cls, match: ChunkMatch) -> Source:
        return cls(
            source=match.source,
            chunk_index=match.chunk_index,
            chunk_id=match.chunk_id,
            similarity=round(match.similarity, 4),
            content=match.content,
        )


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(
        default=None,
        gt=0,
        le=100,
        description="Chunks to retrieve. Defaults to RETRIEVAL_TOP_K.",
    )


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    is_refusal: bool = Field(
        description=(
            "True when the model declined for lack of supporting context. A "
            "correct outcome, not an error."
        )
    )
    seconds: float
    prompt_tokens: int
    response_tokens: int

    @classmethod
    def from_answer(cls, answer: Answer) -> AskResponse:
        return cls(
            answer=answer.text,
            sources=[Source.from_match(match) for match in answer.sources],
            is_refusal=answer.is_refusal,
            seconds=round(answer.seconds, 3),
            prompt_tokens=answer.prompt_tokens,
            response_tokens=answer.response_tokens,
        )
