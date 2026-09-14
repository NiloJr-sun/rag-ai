"""Placeholder suite so CI has something to collect.

Real coverage arrives per ticket, starting with T1.1 (embeddings).
"""


def test_rag_modules_import() -> None:
    from app.rag import chunking, embeddings, generation, retrieval

    assert all(m is not None for m in (chunking, embeddings, generation, retrieval))
