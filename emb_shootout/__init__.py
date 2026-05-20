"""embedding-model-shootout: reproducible technical-docs retrieval benchmark.

Public surface:

    from emb_shootout.corpus import (
        Chunk, DEFAULT_MODULES, build_corpus, write_jsonl,
    )
    # Sweep (#2):
    from emb_shootout import (
        CorpusChunk, Query, SweepResult,
        run_sweep, build_queries, aggregate_markdown,
        HashEmbedderProvider,  # OpenAI/Voyage/Cohere/BGE/Nomic via optional extras
    )
"""

__version__ = "0.0.1"  # mirror of pyproject.toml [project] version

from .corpus import DEFAULT_MODULES, Chunk, build_corpus, write_jsonl
from .providers import HashEmbedderProvider
from .queries import build_queries
from .sweep import (
    CorpusChunk,
    Embedder,
    Query,
    SweepResult,
    aggregate_markdown,
    cosine,
    ndcg_at_k,
    run_sweep,
)

__all__ = [
    # Corpus (#1)
    "DEFAULT_MODULES",
    "Chunk",
    "build_corpus",
    "write_jsonl",
    # Sweep (#2)
    "CorpusChunk",
    "Embedder",
    "HashEmbedderProvider",
    "Query",
    "SweepResult",
    "aggregate_markdown",
    "build_queries",
    "cosine",
    "ndcg_at_k",
    "run_sweep",
]
