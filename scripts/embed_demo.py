"""T1.1 -- see what an embedding actually is.

Needs Ollama serving and the model pulled:

    ollama pull nomic-embed-text
    python scripts/embed_demo.py
"""

from __future__ import annotations

import math
from pathlib import Path

import httpx
from app.rag.embeddings import embed_text

SENTENCES = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",  # same meaning, almost no shared words
    "Kubernetes schedules containers across a cluster.",  # unrelated
    "Hi.",  # much shorter, to show length does not change the dimensions
]

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "sample.txt"


def main() -> None:
    with httpx.Client(timeout=60.0) as client:
        for sentence in SENTENCES:
            vector = embed_text(sentence, client=client)
            norm = math.sqrt(math.fsum(value * value for value in vector))
            preview = ", ".join(f"{value:+.4f}" for value in vector[:4])
            print(f"{len(vector):>4} dims  |norm| {norm:6.3f}  [{preview}, ...]")
            print(f"           {sentence}\n")

        if SAMPLE.exists():
            vector = embed_text(SAMPLE.read_text(), client=client)
            print(f"{len(vector):>4} dims  from {SAMPLE.name}")


if __name__ == "__main__":
    main()
