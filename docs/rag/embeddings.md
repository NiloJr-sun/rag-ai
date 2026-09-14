# Embeddings (T1.1)

An embedding turns a piece of text into a fixed-length list of numbers — a
point in high-dimensional space. The model positions text so that similar
meaning lands in a similar place, which is what later makes search by meaning
possible rather than search by keyword.

## Setup

- Model: `nomic-embed-text` (274 MB), served by local Ollama on `:11434`
- Endpoint: `POST /api/embeddings` with `{"model": ..., "prompt": ...}`
- Code: [`backend/app/rag/embeddings.py`](../../backend/app/rag/embeddings.py)
- Reproduce: `python scripts/embed_demo.py`

## Observations

Measured locally on 2026-09-14 against the running model.

### The dimension count never changes

768 floats come back regardless of input size:

| input | chars | dims | ‖v‖ |
|---|---|---|---|
| `cat` | 3 | 768 | 23.07 |
| one sentence | 23 | 768 | 21.42 |
| ~100 words | 480 | 768 | 22.52 |

This is the property that makes a database column possible at all — every
chunk, long or short, occupies the same `vector(768)` shape.

### Vectors are not normalised

The norms sit around 21–23, not 1. So the raw distance between two vectors
mixes "how different is the meaning" with "how long was the text", and
comparing them by magnitude is meaningless. Direction carries the meaning;
length does not.

### Same input gives the identical vector

Embedding the same string twice returned exactly equal lists. Embedding is a
pure function of the text, so re-embedding unchanged content is wasted work —
worth remembering at T5.7 when deciding whether a modified Drive file really
needs re-processing.

### There is a hard input ceiling

Text above roughly **7,000 characters (~1,750 tokens)** is rejected outright:

```
EmbeddingError: Ollama returned 500: the input length exceeds the context length
```

It fails loudly rather than silently truncating, which is the better failure —
a truncated embedding would look perfectly valid while quietly ignoring most
of the document.

**This is the concrete reason chunking exists.** A document larger than the
context window cannot be embedded at all, so it has to be split first. The
ceiling is the upper bound on chunk size; T1.3 explores what size is actually
*good*, which is a different question.

### Latency

About 19 ms for a 480-character input once the model is warm, 9 ms for a
single word. The first call after a cold start is far slower because it
includes loading the model into memory — hence the 60-second timeout default.

At ~19 ms per chunk, a 300-chunk document costs roughly 6 seconds of embedding
alone. That is what pushes ingestion into a background job at T6.3.

## Open question → T1.2

Nothing here can yet answer the obvious question: *are two pieces of text
similar?* The demo shows "The cat sat on the mat." and "A feline rested on the
rug." have near-identical norms (21.42 vs 21.32) — and so does an unrelated
sentence about Kubernetes (21.25). Magnitude cannot distinguish them.

Comparing direction instead of magnitude is cosine similarity, which is T1.2.

## My notes

<!-- Anything you noticed that isn't captured above. -->
