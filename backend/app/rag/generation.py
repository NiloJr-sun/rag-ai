"""Turning retrieved chunks into an answer (T1.10).

The model never sees the corpus. Retrieved text is pasted into the prompt,
and the system instruction tells the model to use only that -- which is the
whole mechanism behind RAG, and the whole reason it can still go wrong: a
model asked about a topic it already knows will happily answer from memory
instead. T3.5 measures that as faithfulness.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from app.config import chat_model, ollama_base_url
from app.rag.embeddings import ollama_error_detail
from app.storage.supabase import ChunkMatch

# Generation is far slower than embedding -- a few seconds warm, much longer on
# a cold model load -- so it gets its own budget rather than borrowing the
# embedding client's 60s.
DEFAULT_TIMEOUT_SECONDS = 300.0

# Ollama's own default context window is small (2048 tokens on many builds) and
# it truncates SILENTLY: raise RETRIEVAL_TOP_K enough and the retrieved chunks
# fall off the end of the prompt with no error, leaving the model to answer
# from memory. Setting it explicitly makes the ceiling visible.
DEFAULT_NUM_CTX = 8192

SYSTEM_PROMPT = """You answer questions using only the numbered context below.

Rules:
- Use only information from the context. Do not add anything you know from
  elsewhere, even if you are confident it is correct.
- Cite the context you used with its number, like [1] or [2].
- If the context does not contain the answer, reply exactly:
  Apologies, but I don't have that information.

Context:
{context}"""


class GenerationError(RuntimeError):
    """Ollama did not return a usable answer."""


@dataclass(frozen=True)
class Answer:
    """A generated answer and the chunks it was given to work from."""

    text: str
    sources: list[ChunkMatch]
    seconds: float
    prompt_tokens: int
    response_tokens: int

    @property
    def is_refusal(self) -> bool:
        """Whether the model declined for lack of context.

        Not a failure: refusing beats inventing an answer, and T3.5 checks
        that a question with no supporting context produces exactly this.
        """
        return self.text.strip().startswith(
            "Apologies, but I don't have that information."
        )


def build_context(matches: Sequence[ChunkMatch]) -> str:
    """Render chunks as numbered blocks the model can cite by number."""
    return "\n\n".join(
        f"[{number}] {match.content}" for number, match in enumerate(matches, start=1)
    )


def generate_answer(
    question: str,
    matches: Sequence[ChunkMatch],
    *,
    client: httpx.Client | None = None,
) -> Answer:
    """Ask the chat model ``question``, grounded in ``matches``."""
    if not question.strip():
        raise GenerationError("Cannot answer an empty question")

    owned_client = client is None
    client = client or httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS)
    started = time.perf_counter()
    try:
        response = client.post(
            f"{ollama_base_url()}/api/chat",
            json={
                "model": chat_model(),
                # Ollama streams newline-delimited JSON by default, which
                # would make .json() fail on the second line. T4.4 turns
                # streaming back on deliberately.
                "stream": False,
                # Ollama defaults to temperature 0.8, at which the same
                # question answered or refused at random across runs. 0.2 is
                # much steadier but still samples -- set it to 0 for the
                # reproducibility T3's evaluation suite will want.
                "options": {
                    "temperature": 0.2,
                    "num_ctx": DEFAULT_NUM_CTX,
                },
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT.format(
                            context=build_context(matches) or "(no context found)"
                        ),
                    },
                    {"role": "user", "content": question},
                ],
            },
        )
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPStatusError as exc:
        raise GenerationError(
            f"Ollama returned {exc.response.status_code}: {ollama_error_detail(exc)}"
        ) from exc
    except httpx.HTTPError as exc:
        raise GenerationError(f"Request to Ollama failed: {exc}") from exc
    finally:
        if owned_client:
            client.close()

    content = body.get("message", {}).get("content")
    if not content:
        raise GenerationError(
            f"No answer returned for model {chat_model()!r}: {body.get('error', body)}"
        )

    return Answer(
        text=content.strip(),
        sources=list(matches),
        seconds=time.perf_counter() - started,
        prompt_tokens=int(body.get("prompt_eval_count") or 0),
        response_tokens=int(body.get("eval_count") or 0),
    )
