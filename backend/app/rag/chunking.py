"""Splitting documents into embeddable pieces (T1.3).

Two reasons chunking exists, both measured rather than assumed:

1. A hard limit. Ollama rejects input above roughly 7,000 characters with
   "the input length exceeds the context length" -- see docs/rag/embeddings.md.
   A document larger than that cannot be embedded at all.

2. Retrieval precision. One vector per document forces the whole document to
   have a single "meaning". Splitting lets retrieval point at the paragraph
   that answers the question instead of the file that mentions it.

Chunk size is a trade-off, not a setting with a right answer:
smaller chunks rank more precisely but carry less context for the model to
answer from, and very short text scores misleadingly well -- see
docs/rag/retrieval.md.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

# Split after . ! or ? when followed by whitespace. Deliberately naive: it
# mis-handles "Dr. Smith" and "3.14". Good enough to compare strategies,
# and the failure mode is a chunk boundary in an odd place, not lost text.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class ChunkingError(ValueError):
    """The requested chunking parameters cannot produce sensible chunks."""


def chunk_fixed(
    text: str,
    *,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split ``text`` into ``size``-character windows advancing by ``size - overlap``.

    Overlap exists so that a sentence straddling a boundary still appears
    whole in one of the two chunks.
    """
    if size <= 0:
        raise ChunkingError(f"size must be positive, got {size}")
    if not 0 <= overlap < size:
        raise ChunkingError(f"overlap must be in 0..{size - 1}, got {overlap}")

    text = text.strip()
    if not text:
        return []

    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunks.append(text[start : start + size])
        # Stop once a window reaches the end, or the overlap would emit
        # chunks that are wholly contained in the previous one.
        if start + size >= len(text):
            break
    return chunks


def split_sentences(text: str) -> list[str]:
    """Break ``text`` into sentences, discarding blank ones."""
    return [
        part.strip() for part in _SENTENCE_BOUNDARY.split(text.strip()) if part.strip()
    ]


def chunk_by_sentence(text: str, *, max_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """Pack whole sentences into chunks of at most ``max_size`` characters.

    Sentences are never split, so a single sentence longer than ``max_size``
    is emitted on its own and exceeds the limit. That is the trade-off: this
    strategy keeps ideas intact but cannot guarantee a maximum size.
    """
    if max_size <= 0:
        raise ChunkingError(f"max_size must be positive, got {max_size}")

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for sentence in split_sentences(text):
        # +1 for the space that will join it to the previous sentence.
        addition = len(sentence) + (1 if current else 0)
        if current and length + addition > max_size:
            chunks.append(" ".join(current))
            current, length = [], 0
            addition = len(sentence)
        current.append(sentence)
        length += addition

    if current:
        chunks.append(" ".join(current))
    return chunks


@dataclass(frozen=True)
class Chunk:
    """One embeddable piece of a document, with enough context to cite it.

    The fields mirror the columns in backend/migrations/0001_init.sql.
    """

    document_id: str
    chunk_id: str
    chunk_index: int
    text: str
    source: str


def document_id_for(source: str) -> str:
    """Stable id for a document, derived from where it came from.

    Deriving rather than generating means re-ingesting the same file produces
    the same id, so the row is updated instead of duplicated (T5.9).
    """
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def chunk_id_for(document_id: str, chunk_index: int) -> str:
    """Stable id for a chunk, derived from its position in the document.

    Positional rather than content-derived on purpose: when a document is
    edited, chunk 3 stays chunk 3 and its row is overwritten. Hashing the
    content instead would mint a new id and orphan the old row.
    """
    return f"{document_id}-{chunk_index:04d}"


def content_hash(text: str) -> str:
    """Hash of a document's contents, for skipping unchanged files (T5.4)."""
    return hashlib.sha256(text.encode()).hexdigest()


def chunk_document(
    text: str,
    *,
    source: str,
    max_size: int = DEFAULT_CHUNK_SIZE,
) -> list[Chunk]:
    """Split ``text`` into structured chunks tagged with their origin.

    Uses sentence packing rather than fixed windows: see docs/rag/chunking.md
    for the measurement behind that choice -- fixed windows scored higher but
    returned fragments beginning mid-word.
    """
    document_id = document_id_for(source)
    return [
        Chunk(
            document_id=document_id,
            chunk_id=chunk_id_for(document_id, index),
            chunk_index=index,
            text=piece,
            source=source,
        )
        for index, piece in enumerate(chunk_by_sentence(text, max_size=max_size))
    ]
