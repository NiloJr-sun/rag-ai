# Cosine similarity (T1.2)

[`embeddings.md`](embeddings.md) ends on a problem: three sentences with
completely different meanings all had norms around 21, so magnitude tells you
nothing. Cosine similarity solves it by comparing **direction** instead.

```
cos(a, b) = dot(a, b) / (||a|| * ||b||)
```

Dividing by both magnitudes cancels length out entirely, leaving only the
angle between the two vectors. 1.0 is the same direction, 0.0 unrelated,
-1.0 opposite.

- Code: [`backend/app/rag/retrieval.py`](../../backend/app/rag/retrieval.py)
- Reproduce: `python scripts/similarity_demo.py`

## Observations

Measured locally on 2026-09-14 with `nomic-embed-text`.

Query: **"Where was the cat resting?"**

| score | document |
|---|---|
| 0.7722 | A feline rested on the rug. |
| 0.7562 | cat |
| 0.7426 | The cat sat on the mat. |
| 0.6313 | The dog slept in its basket. |
| 0.3744 | Kubernetes schedules containers across a cluster. |

### Meaning beats keywords

The top result shares **no significant words with the query** — no "cat", no
"resting", no "where". "A feline rested on the rug." still outranks "The cat
sat on the mat.", which contains the literal word "cat".

That is the entire justification for the vector approach. Keyword search
scores the feline sentence at zero; embeddings put it first.

Directly comparing the two cat sentences gives **0.7237**, despite them
sharing only "the".

### The scale does not start at zero

The unrelated Kubernetes sentence scores **0.3744**, not 0.0. Nothing lands
near zero in practice, because all English prose shares some direction in
embedding space.

So there is no universal "relevant above 0.5" threshold to be had — a score
is only meaningful *relative to the other candidates for the same query*.
This is why retrieval takes the top *k* results rather than everything above
a cut-off (T1.9), and why a fixed threshold would need tuning per corpus.

### Short text scores misleadingly well

The bare word `cat` came **second**, above the full sentence that actually
answers the question. A single word carries almost no information, but its
vector points squarely at the topic, so it scores highly.

The practical consequence for chunking: very small chunks will compete
unfairly well in ranking while being useless as context for the model to
answer from. Chunk size is not only about the context-window ceiling from
T1.1 — it also affects what ranking does. Something to watch at T1.3.

## Implementation notes

- No numpy. `dot_product` and `magnitude` are explicit loops so the formula
  stays visible.
- `math.fsum` rather than `sum()`: 768 float additions accumulate rounding
  error, and fsum keeps the running total exact.
- Mismatched dimensions raise rather than silently comparing a prefix —
  which is exactly the mistake waiting at T3.7, where different embedding
  models produce different dimension counts.
- A zero vector raises too: it has no direction, so the formula would divide
  by zero.

## Next

`rank_by_similarity` loops over candidates in memory. That is fine for five
sentences and hopeless for a corpus — T1.8 and T1.9 move both the vectors and
this comparison into pgvector.

## My notes

<!-- Anything you noticed that isn't captured above. -->
