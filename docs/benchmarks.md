# Benchmarks

This file is **regenerated** by `emb-shootout sweep aggregate` from the JSON
files under `results/`. Don't hand-edit; commit the underlying result JSON
files instead and re-run the aggregator.

## Current results

The table below is real, but it's only the dep-free `hash-embedder-128d-ngram2`
baseline. The real provider rows (OpenAI / Voyage / Cohere / BGE / Nomic)
land here when the operator runs `emb-shootout sweep run --provider <P>`
against the corpus with their API key and commits the resulting
`results/<P>.json`. Per the no-fabricated-benchmarks rule, this README does
**not** carry placeholder numbers for those providers.

| embedder | dim | n_corpus | n_queries | recall@1 | recall@5 | recall@10 | NDCG@10 | corpus embed (ms) | query p50 (ms) | query p95 (ms) | $/1M tokens |
|----------|----:|---------:|----------:|---:|---:|---:|--------:|------------------:|---------------:|---------------:|------------:|
| hash-embedder-128d-ngram2 | 128 | 12010 | 50 | 0.320 | 0.520 | 0.620 | 0.449 | 429 | 0.0 | 0.0 | $0.000 |

`hash-embedder-128d-ngram2`'s recall@5 of **0.52** isn't the takeaway — it's
the *lower bound*. A SHA-256 bag-of-bigrams projection is a useless
embedder; the real embedders should score substantially higher on the same
queries. When they don't, that's the interesting finding.

## Reproducing

```bash
# Build the corpus (CPython stdlib introspection; deterministic on a fixed
# Python version + module list).
emb-shootout corpus build --out data/corpus.jsonl

# Run a provider. Each non-hash provider needs its own API key + the
# matching extra installed (e.g., pip install 'emb-shootout[openai]').
OPENAI_API_KEY=sk-... emb-shootout sweep run --provider openai \
  --corpus data/corpus.jsonl --queries 200 --output results/openai.json

# Aggregate everything in results/ into this file.
emb-shootout sweep aggregate --results-dir results --out docs/benchmarks.md
```

The query set is derived deterministically from the corpus at sweep time
(seed `42` by default, configurable). All providers run against the same
queries by construction, so cross-provider rows in this table are
apples-to-apples.
