# Chunking (T1.3)

Splitting a document into smaller pieces before embedding them.

- Code: [`backend/app/rag/chunking.py`](../../backend/app/rag/chunking.py)
- Reproduce: `python scripts/chunking_demo.py`

## Why it is necessary

Two separate reasons, and only the first is a hard limit.

**The context ceiling.** Ollama rejects input above roughly 7,000 characters
(see [`embeddings.md`](embeddings.md)). Anything larger cannot be embedded at
all.

**Retrieval precision.** This one bites long before the ceiling does. Our
sample corpus is 2,540 characters — comfortably embeddable whole — and
embedding it as a single chunk is still useless:

| query | score | retrieved |
|---|---|---|
| How does cosine similarity work? | 0.5534 | the whole file |
| Why split a document into smaller pieces? | 0.5507 | the whole file |
| How do I cook pasta? | 0.4565 | the whole file |

With one vector there is only one possible answer, so retrieval cannot
discriminate at all. A question about pasta scores 0.4565 against a document
that is mostly about vector databases. **Chunking is what gives retrieval
something to choose between.**

## Measured comparison

Measured locally on 2026-09-14 against `data/samples/sample.txt` (2,540 chars).
Score is the best-matching chunk for each query.

| strategy | chunks | avg size | cosine query | chunking query | pasta query |
|---|---|---|---|---|---|
| whole document | 1 | 2540 | 0.5534 | 0.5507 | 0.4565 |
| fixed 200 / 0 | 13 | 195 | **0.8280** | 0.7666 | 0.7384 |
| fixed 200 / 50 | 17 | 196 | 0.7937 | **0.7731** | **0.7828** |
| fixed 500 / 50 | 6 | 465 | 0.7264 | 0.7398 | 0.6650 |
| fixed 1000 / 100 | 3 | 913 | 0.6669 | 0.6725 | 0.6068 |
| sentences ≤ 500 | 6 | 421 | 0.7136 | 0.7367 | 0.6747 |

## Trade-offs

### Smaller chunks score higher — but the score is misleading

Scores fall steadily as chunks grow: 0.83 at 200 characters, 0.67 at 1000.
That looks like an argument for small chunks, and it is a trap. It is the
same effect recorded in [`retrieval.md`](retrieval.md), where the bare word
`cat` outranked a full sentence: less text means less competing meaning, so
the vector points more sharply at one topic.

What the score does not measure is whether the chunk is *usable as an answer*.

### Fixed-size chunking cuts mid-word

The top hit for "How does cosine similarity work?" under `fixed 200 / 0`,
which scored the highest of any run at 0.8280:

> `are based on the angle between them rather than their abso...`

It begins mid-sentence and ends mid-word. Other winners across the runs start
`umber of characters...` (from "number"), `vant to a question...` (from
"relevant"), `ing to the instructions...` (from "according"). The retrieval is
*correct* — that really is the right region of the document — but at T1.10
this fragment gets pasted into a prompt, and the model has to answer from a
sentence missing its beginning.

### Sentence chunking scores lower and reads better

Same query under `sentences ≤ 500`, scoring 0.7136:

> `This makes cosine similarity useful for semantic search an...`

About 0.11 lower, and a complete thought. That is the central trade-off of
this ticket: **fixed-size wins on score, sentence-based wins on usefulness.**
Since the score is only used to rank candidates against each other, and the
text is what the model actually reads, readable chunks are worth more than
the number attached to them.

### Overlap recovers boundary losses

`fixed 200 / 0` and `fixed 200 / 50` retrieve different chunks for the cosine
query. With no overlap the boundary fell mid-explanation; with 50 characters
of overlap one window captured the topic sentence:

> `bedding space. A common similarity metric is cosine simila...`

Overlap costs storage — 17 chunks instead of 13, roughly 30% more embeddings
to generate and store — to stop an answer being lost at a boundary.

## Current choice

`sentences ≤ 500` with the plan to add overlap at T2.4. Good retrieval and
readable context beat a higher score that arrives mid-word.

Open question for T3.6: all of this is judged by eye on three queries. The
evaluation suite is what turns "reads better" into a measurement.

## Implementation notes

- `chunk_fixed(text, size, overlap)` steps by `size - overlap` and stops once
  a window reaches the end, so no chunk is wholly contained in its
  predecessor.
- `chunk_by_sentence(text, max_size)` packs whole sentences and **never
  splits one**, so a single sentence longer than `max_size` is emitted alone
  and exceeds it. Whole ideas were worth more than a guaranteed ceiling.
- Sentence detection is a naive regex on `.!?` followed by whitespace. It
  mis-handles `Dr. Smith` and `3.14`. The failure mode is a boundary in an odd
  place, never lost text.
- Chunks are plain strings here. T1.6 adds stable IDs and source metadata.

## My notes

<!-- Anything you noticed that isn't captured above. -->
