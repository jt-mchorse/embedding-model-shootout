"""embedding-model-shootout: reproducible technical-docs retrieval benchmark.

Issue #1 surface:

    from emb_shootout.corpus import (
        Chunk,
        DEFAULT_MODULES,
        build_corpus,
        write_jsonl,
    )

Subsequent issues:
    #2 — model sweep (recall@k, NDCG, cost, latency, Pareto frontier).
"""

from .corpus import DEFAULT_MODULES, Chunk, build_corpus, write_jsonl

__all__ = [
    "DEFAULT_MODULES",
    "Chunk",
    "build_corpus",
    "write_jsonl",
]
