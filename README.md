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
├── corpus.py   ← DEFAULT_MODULES + build_corpus(modules) + write_jsonl()
└── cli.py      ← emb-shootout corpus build --out data/corpus.jsonl
```

The loader walks each module via `inspect`, treats `__all__` as the
authoritative re-export list, emits one `Chunk` per documented member
(signature-then-docstring), deduplicates by qualname, and writes JSONL.
See [`docs/corpus.md`](docs/corpus.md) for the full chunk shape,
license, and provenance.

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

## Benchmarks / Results

*The model sweep (recall@k, NDCG, cost per million tokens, p95 latency,
Pareto frontier) is pending issue [#2]. This PR locks the corpus + chunk
shape so issue #2 has a stable contract to benchmark against.*

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
