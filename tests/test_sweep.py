"""Tests for the sweep harness (issue #2).

Covers:
- cosine + ndcg + percentile math against textbook examples.
- Query construction is deterministic given a seed.
- HashEmbedderProvider conforms to the Embedder Protocol.
- run_sweep end-to-end with HashEmbedderProvider produces sane recall + NDCG.
- SweepResult round-trips through to_dict / from_dict.
- aggregate_markdown emits a header + one row per result.
"""

from __future__ import annotations

import json

import pytest

from emb_shootout import (
    CorpusChunk,
    HashEmbedderProvider,
    SweepResult,
    aggregate_markdown,
    build_queries,
    cosine,
    ndcg_at_k,
    run_sweep,
)
from emb_shootout.sweep import percentile, retrieve_top_k

# ----------------------------------------------------------------------
# Math
# ----------------------------------------------------------------------


def test_cosine_identical_is_one():
    assert cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector_is_zero():
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        cosine([1.0], [1.0, 2.0])


def test_ndcg_perfect_relevance_is_one():
    assert ndcg_at_k([1, 0, 0, 0], 4) == pytest.approx(1.0)


def test_ndcg_no_relevance_is_zero():
    assert ndcg_at_k([0, 0, 0, 0], 4) == 0.0


def test_ndcg_relevant_at_position_2_is_less_than_one():
    # Relevant doc at rank 2 → DCG = 1/log2(3) ≈ 0.6309; iDCG = 1.0.
    score = ndcg_at_k([0, 1, 0, 0], 4)
    assert 0.5 < score < 0.7


def test_ndcg_rejects_zero_k():
    with pytest.raises(ValueError, match="positive"):
        ndcg_at_k([1, 0], 0)


def test_percentile_handles_extremes():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 100.0) == 5.0
    assert percentile(values, 50.0) == pytest.approx(3.0)


def test_percentile_empty_returns_zero():
    assert percentile([], 95.0) == 0.0


# ----------------------------------------------------------------------
# retrieve_top_k
# ----------------------------------------------------------------------


def test_retrieve_top_k_returns_highest_similarity_first():
    corpus = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
    ids = ["a", "b", "c"]
    out = retrieve_top_k([1.0, 0.0], corpus, ids, k=2)
    assert [r[0] for r in out] == ["a", "c"]


def test_retrieve_top_k_rejects_zero_k():
    with pytest.raises(ValueError, match="positive"):
        retrieve_top_k([1.0], [[1.0]], ["a"], k=0)


def test_retrieve_top_k_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        retrieve_top_k([1.0], [[1.0]], ["a", "b"], k=1)


# ----------------------------------------------------------------------
# Query construction
# ----------------------------------------------------------------------


_CORPUS = [
    CorpusChunk(
        chunk_id="c-001",
        text="Postgres autovacuum tuning is a long process that benefits from incremental adjustments rather than wholesale changes.",
    ),
    CorpusChunk(
        chunk_id="c-002",
        text="HNSW indexes trade construction time and memory for sub-linear ANN query latency at the cost of recall.",
    ),
    CorpusChunk(
        chunk_id="c-003",
        text="Anthropic prompt caching cuts token cost on repeated context by ninety percent at read time.",
    ),
    CorpusChunk(
        chunk_id="c-004",
        text="Reciprocal rank fusion combines lexical and dense retrievals without cross-method score normalization.",
    ),
    CorpusChunk(
        chunk_id="c-005",
        text="Asyncio TaskGroup propagates cancellation to every spawned task on the first unhandled exception.",
    ),
]


def test_build_queries_is_deterministic_given_seed():
    a = build_queries(_CORPUS, n=20, seed=42)
    b = build_queries(_CORPUS, n=20, seed=42)
    assert a == b


def test_build_queries_assigns_expected_chunk_id_correctly():
    queries = build_queries(_CORPUS, n=10, seed=1)
    chunk_id_to_text = {c.chunk_id: c.text for c in _CORPUS}
    for q in queries:
        # Query text is a verbatim window of its expected chunk.
        assert q.text in chunk_id_to_text[q.expected_chunk_id]


def test_build_queries_validates_inputs():
    with pytest.raises(ValueError, match="n must be positive"):
        build_queries(_CORPUS, n=0)
    with pytest.raises(ValueError, match="corpus must be non-empty"):
        build_queries([], n=5)


# ----------------------------------------------------------------------
# HashEmbedderProvider
# ----------------------------------------------------------------------


def test_hash_provider_conforms_to_embedder_protocol():
    p = HashEmbedderProvider()
    assert hasattr(p, "name")
    assert isinstance(p.name, str)
    assert hasattr(p, "dim")
    assert p.dim > 0
    assert hasattr(p, "cost_per_million_tokens")
    vecs = p.embed(["hello world", "different text entirely"])
    assert len(vecs) == 2
    assert all(len(v) == p.dim for v in vecs)


def test_hash_provider_validates_dim():
    with pytest.raises(ValueError, match="dim"):
        HashEmbedderProvider(dim=0)


# ----------------------------------------------------------------------
# run_sweep end-to-end
# ----------------------------------------------------------------------


def test_run_sweep_with_hash_provider_runs_and_reports_real_numbers():
    provider = HashEmbedderProvider()
    queries = build_queries(_CORPUS, n=20, seed=42)
    result = run_sweep(_CORPUS, queries, embedder=provider)
    # Shape contract.
    assert isinstance(result, SweepResult)
    assert result.n_corpus == len(_CORPUS)
    assert result.n_queries == 20
    assert set(result.recall_at_k.keys()) == {1, 5, 10}
    for v in result.recall_at_k.values():
        assert 0.0 <= v <= 1.0
    assert 0.0 <= result.ndcg_at_10 <= 1.0
    # Hash embedder on a 5-doc corpus where each query is a verbatim window
    # of its source chunk → recall should be high. Don't pin an exact number.
    assert result.recall_at_k[5] >= 0.6


def test_run_sweep_validates_inputs():
    p = HashEmbedderProvider()
    qs = build_queries(_CORPUS, n=5, seed=1)
    with pytest.raises(ValueError, match="corpus must be non-empty"):
        run_sweep([], qs, embedder=p)
    with pytest.raises(ValueError, match="queries must be non-empty"):
        run_sweep(_CORPUS, [], embedder=p)
    with pytest.raises(ValueError, match="k_values"):
        run_sweep(_CORPUS, qs, embedder=p, k_values=())


# ----------------------------------------------------------------------
# Round-trip + aggregation
# ----------------------------------------------------------------------


def test_sweep_result_round_trips_through_dict():
    provider = HashEmbedderProvider()
    qs = build_queries(_CORPUS, n=10, seed=42)
    original = run_sweep(_CORPUS, qs, embedder=provider)
    serialized = json.loads(json.dumps(original.to_dict()))
    restored = SweepResult.from_dict(serialized)
    assert restored.embedder_name == original.embedder_name
    assert restored.recall_at_k == original.recall_at_k
    assert restored.ndcg_at_10 == pytest.approx(original.ndcg_at_10)


def test_aggregate_markdown_renders_one_row_per_result():
    provider1 = HashEmbedderProvider(dim=64, ngram=1)
    provider2 = HashEmbedderProvider(dim=128, ngram=2)
    qs = build_queries(_CORPUS, n=10, seed=42)
    r1 = run_sweep(_CORPUS, qs, embedder=provider1)
    r2 = run_sweep(_CORPUS, qs, embedder=provider2)
    md = aggregate_markdown([r1, r2])
    assert provider1.name in md
    assert provider2.name in md
    assert "recall@5" in md
    assert "NDCG@10" in md


def test_aggregate_markdown_handles_empty():
    md = aggregate_markdown([])
    assert "no results" in md
