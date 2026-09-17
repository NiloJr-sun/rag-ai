# Generation (T1.10)

Turning retrieved chunks into an answer.

- Code: [`backend/app/rag/generation.py`](../../backend/app/rag/generation.py)
- Reproduce: `python scripts/ask.py "How do I cook pasta?"`

## The mechanism

The model has no access to the database. Retrieved chunks are pasted into the
prompt as numbered blocks, and the system prompt tells the model to use only
those. That substitution is the whole of RAG — and the whole reason it can
still go wrong, because a model asked about something it already knows will
answer from memory unless told firmly not to.

## Observations

Measured locally on 2026-09-17 against `data/samples/sample.txt`, using a
six-question set: four answerable from the corpus, two not.

### Generation is not deterministic

Embedding is a pure function of its input ([`embeddings.md`](embeddings.md)).
Generation is not. At Ollama's default `temperature: 0.8`, the same question
with the same context gave:

```
run 1: answered    run 4: answered
run 2: REFUSED     run 5: answered
run 3: answered    run 6: REFUSED
```

Two refusals in six runs, on a question the corpus answers. This is why
`temperature` is now set explicitly, and why an evaluation suite that runs
each question once (T3.3) would produce noise rather than a measurement.

### A wrong answer can come from correct retrieval

At `chunk_size=1000`, "How do I cook pasta?" was refused even though the
cooking instructions were in the retrieved context, at the tail of a 927-char
chunk that opened with unrelated material about RAG. Retrieval did its job —
the right text was in the prompt — and generation still failed.

Lowering the chunk size to 500 fixed it. The lesson is that retrieval quality
and answer quality are separate measurements, which is why T3.4 and T3.5 are
separate tickets.

### The model mattered more than the prompt

Same prompts, same context, temperature 0, scored on the six-question set:

| model | size | score |
|---|---|---|
| llama3.2 | 3B | 5/6 |
| qwen2 | 7B | 6/6 |

Prompt rewording moved llama3.2's result around but never fixed it. Two
variations worth recording anyway, since they mattered for the smaller model:

- Moving the context from the system message into the user message turned a
  failure into a pass. Small models treat system content as weak guidance.
- Rewording the rules changed which questions passed, with no obvious pattern.

### Refusal is a feature

Asked "What is the capital of France?", the model refuses, despite certainly
knowing the answer. That is the correct behaviour for a grounded system and is
what `Answer.is_refusal` reports.

Detection is prose-matching on the refusal sentence, which is fragile: a
paraphrase would be counted as an answer. Worth hardening before refusal rate
becomes a metric in T3.5.

## Implementation notes

- `"stream": False` — Ollama streams newline-delimited JSON by default, one
  object per token, which breaks `.json()`. T4.4 turns it back on deliberately.
- `"num_ctx"` is set explicitly. Ollama's default window is small on many
  builds and it truncates **silently**: raise `RETRIEVAL_TOP_K` far enough and
  the retrieved chunks drop off the end of the prompt with no error at all.
- `/api/chat`, not `/api/generate` — the `messages` list is what T4.6's
  multi-turn conversation will need.
- Generation has its own 300s timeout. Reusing the embedding client's 60s made
  a cold model load look like a network failure.

## Cost

Roughly 2 seconds warm for a 5-chunk prompt, against ~19ms to embed. Generation
dominates the latency budget entirely — the reason T4.4 streams tokens to the
UI, and T6.9 measures each stage separately.

## My notes

<!-- Anything you noticed that isn't captured above. -->
