# embedding-model-shootout
> Reproducible empirical comparison of embedding models on a technical-docs corpus: recall@k, NDCG, cost per million tokens, latency, Pareto frontier.

![CI](https://github.com/jt-mchorse/embedding-model-shootout/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## What this is

Picking an embedding model is one of the few choices in a RAG stack that
*directly* moves recall numbers — and the only way to make it
defensibly is to measure on a corpus that looks like yours. Most
published shootouts hide their corpus, vary their k, or quietly use a
held-out set the embedding provider trained on. This repo's bet is the
opposite: a single, publicly-licensed corpus, the same query set across
every model, and every number reproducible from a fresh clone.

This PR ships the **corpus layer**: the curated module list, the
deterministic loader, and the documentation that makes the corpus
choice auditable. The corpus is one chunk per documented CPython
standard-library member (function, class, method), regenerated from the
running Python interpreter via `inspect` ([D-002] — not committed as
data, reproduced from source). On CPython 3.14 the curated list yields
**12,010 chunks**, comfortably above the ≥10k acceptance bar for stable
metrics.

The actual model sweep — five providers, recall@k, NDCG, cost, latency,
Pareto plot — lands in issue [#2]. The corpus's chunk shape and provenance
fields (`chunk_id`, `qualname`, `kind`, `source`) are the contract the
benchmark code will hold to.

[#2]: https://github.com/jt-mchorse/embedding-model-shootout/issues/2

## Architecture

```
emb_shootout/
├── corpus.py     ← DEFAULT_MODULES + build_corpus(modules) + write_jsonl()
├── queries.py    ← #2: deterministic verbatim-snippet queries from the corpus
├── sweep.py      ← #2: Embedder Protocol + run_sweep + recall@k + NDCG + aggregator
├── providers/    ← #2: HashEmbedder (default) + OpenAI, Voyage, Cohere, BGE, Nomic
└── cli.py        ← emb-shootout corpus build / sweep run / sweep aggregate
```

The corpus loader walks each module via `inspect`, treats `__all__` as the
authoritative re-export list, emits one `Chunk` per documented member
(signature-then-docstring), deduplicates by qualname, and writes JSONL.
See [`docs/corpus.md`](docs/corpus.md) for the full chunk shape,
license, and provenance.

The sweep harness (#2) is one `Embedder` Protocol (`embed(texts) -> list[list[float]]`
plus `name`, `dim`, `cost_per_million_tokens`) and one `run_sweep(corpus,
queries, embedder)` function that embeds corpus + queries, runs cosine
top-k retrieval, and reports recall@1/5/10, NDCG@10, and embed-latency
p50/p95. The dep-free `HashEmbedderProvider` ships in the base install for
hermetic CI; the five real providers (OpenAI, Voyage, Cohere, BGE, Nomic)
are lazy-imported behind their respective optional extras and wire to
their SDKs via env vars.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# Build the full curated-list corpus (≥10k chunks, ~5–10s on a laptop).
emb-shootout corpus build --out data/corpus.jsonl

# Or restrict to a single module:
emb-shootout corpus build --out /tmp/just-json.jsonl --module json

# Inspect:
wc -l data/corpus.jsonl
head -1 data/corpus.jsonl | python -m json.tool
```

Tests, lint, format check:

```bash
ruff check . && ruff format --check . && pytest
```

## Sweep harness (#2 · this PR)

```bash
# Build the corpus.
emb-shootout corpus build --out data/corpus.jsonl

# Run a provider. Hash-only (dep-free, hermetic) baseline:
emb-shootout sweep run --provider hash \
  --corpus data/corpus.jsonl --queries 200 --output results/hash.json

# Real providers each need their SDK + API key:
pip install 'emb-shootout[openai]'
OPENAI_API_KEY=sk-... emb-shootout sweep run --provider openai \
  --corpus data/corpus.jsonl --queries 200 --output results/openai.json

# Aggregate JSONs into the markdown table.
emb-shootout sweep aggregate --results-dir results --out docs/benchmarks.md
```

Six providers wired (`hash`, `openai`, `voyage`, `cohere`, `bge`, `nomic`).
The query set is derived deterministically from the corpus at sweep time
(seed `42` by default), so all providers run against the same queries by
construction — cross-provider rows are apples-to-apples.

## Benchmarks / Results

See [`docs/benchmarks.md`](docs/benchmarks.md). The `hash` baseline is
real (a real run on the 12010-chunk corpus); the five real providers'
rows land when the operator runs them with their API keys + budgets and
commits the resulting `results/<provider>.json`. Per the
no-fabricated-benchmarks rule, this README does **not** carry placeholder
numbers for those providers.

## Demo

*60-second demo pending — depends on issue [#2].*

## Why these decisions

See [`MEMORY/core_decisions_human.md`](MEMORY/core_decisions_human.md).

## License

MIT.

The corpus content itself is derived from CPython stdlib docstrings,
licensed under the [Python Software Foundation License v2][psf]. See
`docs/corpus.md` for attribution details.

[psf]: https://docs.python.org/3/license.html
[D-002]: MEMORY/core_decisions_human.md
