# Architecture

## Shipped (this PR — issue #1)

```mermaid
flowchart LR
    classDef shipped fill:#dcffe4,stroke:#22863a,color:#000
    classDef pending fill:#fff5b4,stroke:#c69400,color:#000

    Py["CPython stdlib<br/>(running interpreter)"]:::shipped --> Insp["inspect-based loader<br/>emb_shootout.corpus.build_corpus()"]:::shipped
    Insp --> Chunks["Chunks<br/>(chunk_id, text, module,<br/>qualname, kind, source)"]:::shipped
    Chunks --> JSONL[("data/corpus.jsonl<br/>≥10k records")]:::shipped

    JSONL --> Sweep["Model sweep (#2)<br/>· OpenAI<br/>· Voyage<br/>· Cohere<br/>· BGE<br/>· Nomic"]:::pending
    QS["Held-out queries<br/>(#2)"]:::pending --> Sweep
    Sweep --> Metrics["recall@k / NDCG / cost / p95 (#2)"]:::pending
    Metrics --> Plot["Pareto frontier plot<br/>(#2)"]:::pending
```

## Components shipped

- **`emb_shootout.corpus.DEFAULT_MODULES`** — curated list of ~140
  stdlib modules. The list is the corpus's only configuration.
- **`build_corpus(modules)`** — generator over `Chunk` records. Skips
  unimportable modules silently (e.g., `readline` on Windows) so a
  Windows reproduction doesn't fail; the set of skipped modules
  surfaces in the CLI's JSON summary.
- **`write_jsonl(chunks, path)`** — deterministic JSONL writer.
- **`emb-shootout corpus build`** — argparse CLI entry point.

## Pending

- **Issue #2:** the actual model sweep. Five embedders, the same
  technical-docs query set, recall@1/5/10, NDCG@10, cost per million
  tokens, p95 latency. Results land in `results/*.json`; Pareto plot in
  `docs/benchmarks.md`.
- Query-set construction methodology will be locked when #2 ships
  (current plan: mine questions from real Python user activity against
  expected answer modules — exact methodology decided in #2).
