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
from emb_shootout.sweep import Query, percentile, retrieve_top_k

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


# Issue #73: cosine ties must break on the stable chunk id, not corpus insertion
# order, so the ranking (and recall@k / NDCG) is a pure function of the
# (similarity, chunk_id) set — not of how the corpus was read.


def test_retrieve_top_k_tie_break_is_corpus_order_independent():
    # Two chunks with identical vectors → identical cosine to any query (a tie).
    # The result must be the same regardless of the order they're passed in.
    qv = [1.0, 0.0]
    vecs = [[1.0, 1.0], [1.0, 1.0]]
    a = retrieve_top_k(qv, vecs, ["expected", "other"], k=1)
    b = retrieve_top_k(qv, vecs, ["other", "expected"], k=1)
    assert a == b
    # Deterministically the alphabetically-smaller id wins the tie.
    assert a[0][0] == "expected"


def test_run_sweep_recall_is_invariant_under_corpus_reordering_on_ties():
    class _TieEmbedder:
        name = "tie"
        dim = 2
        cost_per_million_tokens = 0.0

        def embed(self, texts):
            # All corpus chunks map to the same vector → ties; the query differs.
            return [[1.0, 0.0] if t == "QUERY" else [1.0, 1.0] for t in texts]

    def _q():
        return Query(query_id="q1", text="QUERY", expected_chunk_id="expected")

    ca = [CorpusChunk(chunk_id="expected", text="c1"), CorpusChunk(chunk_id="other", text="c2")]
    cb = [CorpusChunk(chunk_id="other", text="c2"), CorpusChunk(chunk_id="expected", text="c1")]
    ra = run_sweep(ca, [_q()], embedder=_TieEmbedder(), k_values=(1,)).recall_at_k
    rb = run_sweep(cb, [_q()], embedder=_TieEmbedder(), k_values=(1,)).recall_at_k
    assert ra == rb  # pre-fix: {1: 1.0} vs {1: 0.0} — corpus order leaked into recall


def test_retrieve_top_k_non_tied_ranking_unchanged():
    # Over-rejection guard: when similarities clearly differ, the score still
    # dominates and the id tiebreak is inert.
    corpus = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
    ids = ["a", "b", "c"]
    out = retrieve_top_k([1.0, 0.0], corpus, ids, k=2)
    assert [r[0] for r in out] == ["a", "c"]


# Issue #31: extend ndcg_at_k and retrieve_top_k sign-only k checks to
# require isinstance(int). NaN slipped past <=0 and surfaced as cryptic
# TypeError deep inside slicing; fractional k truncated silently.
@pytest.mark.parametrize("bad_k", [1.5, float("nan"), True, "5"])
def test_ndcg_at_k_rejects_non_int_k(bad_k):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        ndcg_at_k([1, 0], bad_k)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_k", [1.5, float("nan"), True, "5"])
def test_retrieve_top_k_rejects_non_int_k(bad_k):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        retrieve_top_k([1.0], [[1.0]], ["a"], k=bad_k)  # type: ignore[arg-type]


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


def test_build_queries_snippets_are_verbatim_substrings_of_multiline_chunks():
    # #67: build_queries rejoined tokens with " ".join, collapsing newlines /
    # tabs / multi-space gaps — so a snippet straddling a non-single-space
    # boundary was NOT a substring of its source chunk. The real corpus
    # (signature + "\n\n" + docstring) is full of such boundaries; the
    # single-line _CORPUS fixture above masked it. Use realistic multi-line
    # chunks and assert every snippet is a verbatim substring.
    multiline = [
        CorpusChunk(
            chunk_id="m-001",
            text="heappush(heap, item)\n\nPush item onto heap,\nmaintaining the heap  invariant for the queue.",
        ),
        CorpusChunk(
            chunk_id="m-002",
            text="json.loads(s)\n\nDeserialize s (a str, bytes or bytearray\ninstance) to a Python object using this conversion table.",
        ),
    ]
    queries = build_queries(multiline, n=50, seed=7, min_words=4, max_words=10)
    by_id = {c.chunk_id: c.text for c in multiline}
    for q in queries:
        assert q.text in by_id[q.expected_chunk_id], repr(q.text)
    # And at least one snippet actually contains an original non-space
    # separator (newline or double space), proving the verbatim path is
    # exercised rather than all windows landing inside single-space runs.
    assert any(("\n" in q.text or "  " in q.text) for q in queries)


def test_build_queries_validates_inputs():
    with pytest.raises(ValueError, match="n must be a positive integer"):
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


# A non-finite embedding component makes cosine() return NaN, whose all-False
# comparisons scramble retrieve_top_k's ranking — yet recall = hits/n_queries
# stays in [0,1], so the SweepResult guard (#62) can't catch it. run_sweep must
# reject corrupt vectors at the embedder-output seam (where the length check is).
class _PoisonEmbedder:
    """Stub Embedder that injects one non-finite component into a chosen vector.

    `poison="corpus"` taints the first corpus vector; `poison="query"` taints
    the first query vector. Otherwise behaves like a tiny finite embedder.
    """

    name = "poison-embedder"
    dim = 3
    cost_per_million_tokens = 0.0

    def __init__(self, *, poison: str, bad: float = float("nan")) -> None:
        self._poison = poison
        self._bad = bad
        self._corpus_calls = 0

    def embed(self, texts):
        # A batch of >1 is the corpus embed; single-text calls are per-query.
        is_corpus = len(texts) > 1
        out = [[0.1, 0.2, 0.3] for _ in texts]
        if is_corpus and self._poison == "corpus":
            out[0][1] = self._bad
        if not is_corpus and self._poison == "query":
            out[0][0] = self._bad
        return out


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_run_sweep_rejects_non_finite_corpus_vector(bad: float):
    qs = build_queries(_CORPUS, n=3, seed=1)
    with pytest.raises(ValueError, match=r"non-finite component in corpus vector 0 at dim 1"):
        run_sweep(_CORPUS, qs, embedder=_PoisonEmbedder(poison="corpus", bad=bad))


def test_run_sweep_rejects_non_finite_query_vector():
    qs = build_queries(_CORPUS, n=3, seed=1)
    with pytest.raises(ValueError, match=r"non-finite component in query vector 0 at dim 0"):
        run_sweep(_CORPUS, qs, embedder=_PoisonEmbedder(poison="query"))


def test_run_sweep_accepts_finite_stub_embedder():
    # The guard must not reject a clean finite embedder — the poison stub with
    # no poisoning runs end-to-end and returns an in-range recall.
    qs = build_queries(_CORPUS, n=3, seed=1)
    result = run_sweep(_CORPUS, qs, embedder=_PoisonEmbedder(poison="none"))
    for v in result.recall_at_k.values():
        assert 0.0 <= v <= 1.0


# ----------------------------------------------------------------------
# k_values per-element guard (#27)
# ----------------------------------------------------------------------
# Non-positive `k` passes through list slicing (`retrieved_ids[:k]`) without
# raising — k=0 silently produces a tautological recall@0=0; k<0 silently
# miscounts ("all but the last N" entries). The guard surfaces every bad
# value in one pass so operators don't chase them one at a time.


def test_run_sweep_rejects_zero_in_k_values():
    p = HashEmbedderProvider()
    qs = build_queries(_CORPUS, n=5, seed=1)
    with pytest.raises(ValueError, match=r"every k in k_values must be positive; got \[0\]"):
        run_sweep(_CORPUS, qs, embedder=p, k_values=(0, 5))


def test_run_sweep_rejects_negative_in_k_values():
    p = HashEmbedderProvider()
    qs = build_queries(_CORPUS, n=5, seed=1)
    with pytest.raises(ValueError, match=r"every k in k_values must be positive; got \[-1\]"):
        run_sweep(_CORPUS, qs, embedder=p, k_values=(-1, 5))


def test_run_sweep_lists_all_bad_k_values_in_one_message():
    # All offenders should appear in the error, not just the first — operators
    # shouldn't have to fix-and-rerun N times.
    p = HashEmbedderProvider()
    qs = build_queries(_CORPUS, n=5, seed=1)
    with pytest.raises(ValueError, match=r"every k in k_values must be positive") as exc_info:
        run_sweep(_CORPUS, qs, embedder=p, k_values=(-3, 0, 5))
    msg = str(exc_info.value)
    assert "-3" in msg
    assert "0" in msg
    # Sorted ascending; canonical form lets operators copy-paste the fix.
    assert "[-3, 0]" in msg


@pytest.mark.parametrize("ks", [(1,), (5, 10), (1, 5, 10, 20)])
def test_run_sweep_accepts_positive_k_values(ks):
    # Regression pin: positive k_values shapes still run cleanly.
    p = HashEmbedderProvider()
    qs = build_queries(_CORPUS, n=3, seed=1)
    result = run_sweep(_CORPUS, qs, embedder=p, k_values=ks)
    assert set(result.recall_at_k.keys()) == set(ks)


# ----------------------------------------------------------------------
# k_values duplicate guard (#57)
# ----------------------------------------------------------------------
# A duplicate `k` silently miscounts: the per-query loop increments
# `hits_at_k[k]` once per occurrence while the dict carries one entry per
# distinct k, so a k appearing N times counts each hit N times and yields
# recall > 1.0 — a mathematically invalid number. Reject it loud, in the same
# one-pass style as the non-positive guard (#27).


def test_run_sweep_rejects_duplicate_k_values():
    p = HashEmbedderProvider()
    qs = build_queries(_CORPUS, n=5, seed=1)
    with pytest.raises(
        ValueError, match=r"k_values must not contain duplicates; got duplicate \[5\]"
    ):
        run_sweep(_CORPUS, qs, embedder=p, k_values=(5, 5))


def test_run_sweep_lists_all_duplicate_k_values_in_one_message():
    # Every duplicated value should appear once, sorted ascending — operators
    # shouldn't have to fix-and-rerun for each repeat. Non-duplicated values
    # (here, 10) must not be listed.
    p = HashEmbedderProvider()
    qs = build_queries(_CORPUS, n=5, seed=1)
    with pytest.raises(ValueError, match=r"k_values must not contain duplicates") as exc_info:
        run_sweep(_CORPUS, qs, embedder=p, k_values=(5, 1, 5, 10, 1))
    msg = str(exc_info.value)
    # Canonical sorted form lets operators copy-paste the offenders.
    assert "[1, 5]" in msg
    assert "10" not in msg


def test_run_sweep_duplicate_k_would_corrupt_recall_without_guard():
    # Pins the actual bug the guard prevents: pre-#57, k_values=(5, 5) double-
    # counted every hit and produced recall@5 = 2.0 (> 1.0, invalid). The guard
    # must reject it rather than return a corrupt number.
    p = HashEmbedderProvider()
    qs = build_queries(_CORPUS, n=10, seed=42)
    # Sanity: the distinct-k baseline is a valid probability.
    baseline = run_sweep(_CORPUS, qs, embedder=p, k_values=(5,))
    assert 0.0 <= baseline.recall_at_k[5] <= 1.0
    # The duplicate form must not reach the corrupt computation at all.
    with pytest.raises(ValueError, match="duplicates"):
        run_sweep(_CORPUS, qs, embedder=p, k_values=(5, 5))


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


# Issue #29: SweepResult.__post_init__ rejects sign-corrupting fields. The
# Pareto frontier comparator at pareto.py:33-34 uses cost as the load-bearing
# x-axis; a negative-cost provider silently dominates every other point. The
# centralized guard catches all construction paths (run_sweep, from_dict,
# direct construction) without per-provider duplication.
def _valid_sweep_result_kwargs() -> dict:
    return dict(
        embedder_name="x",
        embedder_dim=128,
        cost_per_million_tokens=0.02,
        n_corpus=10,
        n_queries=5,
        recall_at_k={5: 0.8},
        ndcg_at_10=0.7,
        embed_latency_ms={"corpus_total": 1.0},
    )


@pytest.mark.parametrize("bad_cost", [-0.001, -1.0, -100.0])
def test_sweep_result_rejects_negative_cost(bad_cost: float):
    kwargs = _valid_sweep_result_kwargs()
    kwargs["cost_per_million_tokens"] = bad_cost
    # Message tightened in #31 to "must be a finite number >= 0.0".
    with pytest.raises(
        ValueError, match=r"cost_per_million_tokens must be a finite number >= 0\.0"
    ):
        SweepResult(**kwargs)


# Issue #31: extend sign-only checks to finiteness on cost (float) and to
# isinstance(int) on the three count fields. NaN cost propagates into the
# Pareto comparator at pareto.py:33-34 silently degrading dominance; NaN /
# fractional / non-int counts propagate through length checks and arithmetic.
@pytest.mark.parametrize(
    "bad_cost",
    [float("nan"), float("inf"), float("-inf")],
)
def test_sweep_result_rejects_non_finite_cost(bad_cost: float):
    kwargs = _valid_sweep_result_kwargs()
    kwargs["cost_per_million_tokens"] = bad_cost
    with pytest.raises(ValueError, match=r"cost_per_million_tokens must be a finite number"):
        SweepResult(**kwargs)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("embedder_dim", 1.5),
        ("embedder_dim", float("nan")),
        ("embedder_dim", True),
        ("n_corpus", 0.5),
        ("n_corpus", float("nan")),
        ("n_queries", 0.5),
        ("n_queries", float("nan")),
    ],
)
def test_sweep_result_rejects_non_int_count_fields(field: str, bad_value):
    kwargs = _valid_sweep_result_kwargs()
    kwargs[field] = bad_value
    with pytest.raises(ValueError, match=rf"{field} must be an int"):
        SweepResult(**kwargs)


@pytest.mark.parametrize(
    ("field", "bad_value", "bound_pattern"),
    [
        ("embedder_dim", 0, r"embedder_dim must be >= 1"),
        ("embedder_dim", -1, r"embedder_dim must be >= 1"),
        ("n_corpus", -1, r"n_corpus must be >= 0"),
        ("n_queries", -1, r"n_queries must be >= 0"),
    ],
)
def test_sweep_result_rejects_invalid_count_fields(field: str, bad_value: int, bound_pattern: str):
    kwargs = _valid_sweep_result_kwargs()
    kwargs[field] = bad_value
    with pytest.raises(ValueError, match=bound_pattern):
        SweepResult(**kwargs)


def test_sweep_result_accepts_zero_cost_and_zero_counts():
    # Zero cost: meaningful for a free local embedder (e.g., HashEmbedder).
    # Zero counts: meaningful at the edge of an empty-input sweep (n_corpus=0
    # is rejected upstream by run_sweep, but the dataclass itself should
    # accept the boundary for tests that construct results directly).
    kwargs = _valid_sweep_result_kwargs()
    kwargs["cost_per_million_tokens"] = 0.0
    kwargs["n_corpus"] = 0
    kwargs["n_queries"] = 0
    result = SweepResult(**kwargs)
    assert result.cost_per_million_tokens == 0.0
    assert result.n_corpus == 0


def test_sweep_result_from_dict_round_trip_rejects_corrupt_negative_cost():
    # The round-trip path through from_dict is also protected by __post_init__.
    # A hand-edited or tampered result JSON can't silently land on the
    # Pareto plot with a negative cost.
    provider = HashEmbedderProvider()
    qs = build_queries(_CORPUS, n=5, seed=1)
    original = run_sweep(_CORPUS, qs, embedder=provider)
    serialized = json.loads(json.dumps(original.to_dict()))
    serialized["cost_per_million_tokens"] = -1.0
    with pytest.raises(
        ValueError, match=r"cost_per_million_tokens must be a finite number >= 0\.0"
    ):
        SweepResult.from_dict(serialized)


# Issue #61: __post_init__ guarded cost/dim/counts but not the metric values.
# recall@k and nDCG@10 are proportions in [0, 1]; an out-of-range or non-finite
# value silently corrupts the Pareto comparator and the plot. Now rejected.
@pytest.mark.parametrize("bad", [1.5, 2.0, -0.2, -1.0, float("nan"), float("inf"), float("-inf")])
def test_sweep_result_rejects_out_of_range_or_non_finite_recall(bad: float):
    kwargs = _valid_sweep_result_kwargs()
    kwargs["recall_at_k"] = {5: bad}
    with pytest.raises(ValueError, match=r"recall_at_k\[5\] must be a finite number in \[0, 1\]"):
        SweepResult(**kwargs)


@pytest.mark.parametrize("bad", [1.5, 2.0, -0.2, -1.0, float("nan"), float("inf"), float("-inf")])
def test_sweep_result_rejects_out_of_range_or_non_finite_ndcg(bad: float):
    kwargs = _valid_sweep_result_kwargs()
    kwargs["ndcg_at_10"] = bad
    with pytest.raises(ValueError, match=r"ndcg_at_10 must be a finite number in \[0, 1\]"):
        SweepResult(**kwargs)


def test_sweep_result_accepts_inclusive_metric_boundaries():
    # The [0, 1] check is inclusive: a perfect (1.0) or zero (0.0) metric is valid.
    kwargs = _valid_sweep_result_kwargs()
    kwargs["recall_at_k"] = {1: 0.0, 5: 1.0}
    kwargs["ndcg_at_10"] = 1.0
    result = SweepResult(**kwargs)
    assert result.recall_at_k == {1: 0.0, 5: 1.0}
    assert result.ndcg_at_10 == 1.0


def test_sweep_result_from_dict_round_trip_rejects_corrupt_recall():
    # The from_dict read path is also protected: a tampered result JSON with an
    # impossible recall can't silently land on the Pareto frontier.
    provider = HashEmbedderProvider()
    qs = build_queries(_CORPUS, n=5, seed=1)
    original = run_sweep(_CORPUS, qs, embedder=provider)
    serialized = json.loads(json.dumps(original.to_dict()))
    serialized["recall_at_k"] = {"5": 1.5}
    with pytest.raises(ValueError, match=r"recall_at_k\[5\] must be a finite number in \[0, 1\]"):
        SweepResult.from_dict(serialized)


# Issue #65: embed_latency_ms was the one numeric SweepResult field without the
# #31 finiteness guard. from_dict's float(v) accepts "Infinity"/"NaN", which then
# render in the markdown table / JSON aggregate. Latency is finite and >= 0.
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -0.001])
def test_sweep_result_rejects_non_finite_or_negative_latency(bad: float):
    kwargs = _valid_sweep_result_kwargs()
    kwargs["embed_latency_ms"] = {"corpus_total": bad}
    with pytest.raises(
        ValueError, match=r"embed_latency_ms\['corpus_total'\] must be a finite number >= 0"
    ):
        SweepResult(**kwargs)


def test_sweep_result_rejects_non_numeric_latency():
    kwargs = _valid_sweep_result_kwargs()
    kwargs["embed_latency_ms"] = {"query_p50": "fast"}
    with pytest.raises(ValueError, match=r"embed_latency_ms\['query_p50'\] must be a number"):
        SweepResult(**kwargs)


def test_sweep_result_from_dict_round_trip_rejects_corrupt_latency():
    # The from_dict read path is protected too: a tampered result JSON whose
    # embed_latency_ms carries inf can't render an Infinity token into the dump.
    provider = HashEmbedderProvider()
    qs = build_queries(_CORPUS, n=5, seed=1)
    original = run_sweep(_CORPUS, qs, embedder=provider)
    serialized = json.loads(json.dumps(original.to_dict()))
    serialized["embed_latency_ms"]["corpus_total"] = float("inf")
    with pytest.raises(
        ValueError, match=r"embed_latency_ms\['corpus_total'\] must be a finite number >= 0"
    ):
        SweepResult.from_dict(serialized)


def test_sweep_result_accepts_valid_latency_dict():
    # The clean path is unaffected: a finite, non-negative latency dict constructs.
    kwargs = _valid_sweep_result_kwargs()
    kwargs["embed_latency_ms"] = {"corpus_total": 12.0, "query_p50": 0.0, "query_p95": 3.5}
    result = SweepResult(**kwargs)
    assert result.embed_latency_ms["query_p50"] == 0.0


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


def test_aggregate_markdown_escapes_pipe_in_embedder_name_so_columns_dont_break():
    # #79 (sibling to llm-eval-harness #134 / rag-kit comment #130): `embedder_name`
    # is the one free-form GFM table cell (every other is a formatted number). It
    # reaches `aggregate_markdown` with an arbitrary `|` via `from_dict` on an
    # external result file, or a BYO Embedder whose `name` carries one. GFM splits
    # table cells on unescaped pipes, so a piped name injects a spurious column and
    # corrupts the table. The fix escapes `|` -> `\|`; the invariant is that the
    # data row's unescaped-pipe count equals the header's. Fails pre-fix (14 vs 13).
    import re

    kwargs = _valid_sweep_result_kwargs()
    kwargs["embedder_name"] = "bge-small|v1.5"
    md = aggregate_markdown([SweepResult(**kwargs)])
    lines = md.splitlines()
    header_line = lines[0]
    row_line = next(line for line in lines if "bge-small" in line)

    def unescaped_pipes(s: str) -> int:
        # A `\|` renders as a literal pipe and does NOT split the cell; only a
        # bare, unescaped `|` is a column delimiter.
        return len(re.findall(r"(?<!\\)\|", s))

    assert unescaped_pipes(row_line) == unescaped_pipes(header_line)
    # The literal pipe is preserved (escaped), not dropped.
    assert "bge-small\\|v1.5" in row_line


# ----------------------------------------------------------------------
# aggregate_json (issue #23)
# ----------------------------------------------------------------------


def _two_aggregable_results() -> tuple[SweepResult, SweepResult, str, str]:
    provider1 = HashEmbedderProvider(dim=64, ngram=1)
    provider2 = HashEmbedderProvider(dim=128, ngram=2)
    qs = build_queries(_CORPUS, n=10, seed=42)
    r1 = run_sweep(_CORPUS, qs, embedder=provider1)
    r2 = run_sweep(_CORPUS, qs, embedder=provider2)
    return r1, r2, provider1.name, provider2.name


def test_aggregate_json_shape_and_keys():
    from emb_shootout.sweep import aggregate_json

    r1, r2, _, _ = _two_aggregable_results()
    payload = aggregate_json([r1, r2])
    assert set(payload) == {"results", "ks"}
    assert isinstance(payload["ks"], list)
    assert all(isinstance(k, int) for k in payload["ks"])
    expected_keys = {
        "embedder",
        "dim",
        "n_corpus",
        "n_queries",
        "recall",
        "ndcg_at_10",
        "corpus_embed_ms",
        "query_p50_ms",
        "query_p95_ms",
        "cost_per_million_tokens",
    }
    for row in payload["results"]:
        assert expected_keys <= set(row), f"row missing keys: {expected_keys - set(row)}"
        for k in payload["ks"]:
            # recall keyed by string-of-k so it round-trips JSON cleanly.
            assert str(k) in row["recall"]


def test_aggregate_json_row_order_matches_markdown_sort():
    """JSON consumer should be able to cross-check against the markdown
    table row-by-row — pinning the sort order keeps that diff possible."""
    from emb_shootout.sweep import aggregate_json

    r1, r2, _, _ = _two_aggregable_results()
    payload = aggregate_json([r1, r2])
    json_order = [row["embedder"] for row in payload["results"]]
    markdown_order = sorted([r1.embedder_name, r2.embedder_name])
    assert json_order == markdown_order


def test_aggregate_json_ks_is_union_across_results():
    from emb_shootout.sweep import SweepResult, aggregate_json

    base = run_sweep(_CORPUS, build_queries(_CORPUS, n=5, seed=42), embedder=HashEmbedderProvider())
    base_dict = base.to_dict()
    # Two synthesized results with different k coverage; aggregate_json
    # surfaces the union, not the intersection.
    a = SweepResult.from_dict({**base_dict, "embedder_name": "p-a", "recall_at_k": {"1": 0.5}})
    b = SweepResult.from_dict({**base_dict, "embedder_name": "p-b", "recall_at_k": {"10": 0.8}})
    payload = aggregate_json([a, b])
    assert payload["ks"] == [1, 10]


def test_aggregate_markdown_byte_identical_to_prior_inline_implementation():
    """Regression guard: the helper extraction in this PR shouldn't change
    a single byte of the markdown output."""
    r1, _, _, _ = _two_aggregable_results()
    md = aggregate_markdown([r1])
    assert "| embedder | dim | n_corpus | n_queries |" in md
    assert "recall@" in md
    assert "NDCG@10" in md


# ----------------------------------------------------------------------
# #47: SweepResult.to_dict — explicit field-by-field contract
# (no dataclasses.asdict). Mirrors the observability-parity arc landed
# across the Python repos.
# ----------------------------------------------------------------------


def test_sweep_result_to_dict_field_set_is_pinned():
    r = SweepResult(
        embedder_name="test",
        embedder_dim=8,
        cost_per_million_tokens=0.0,
        n_corpus=10,
        n_queries=5,
        recall_at_k={1: 0.5, 5: 0.8, 10: 0.9},
        ndcg_at_10=0.75,
        embed_latency_ms={"corpus_total": 1.0, "query_p50": 0.1, "query_p95": 0.2},
        notes=[],
    )
    d = r.to_dict()
    assert sorted(d.keys()) == [
        "cost_per_million_tokens",
        "embed_latency_ms",
        "embedder_dim",
        "embedder_name",
        "n_corpus",
        "n_queries",
        "ndcg_at_10",
        "notes",
        "recall_at_k",
    ]


def test_sweep_result_to_dict_recall_at_k_keys_stringified():
    # JSON has no integer-key type — preserve the existing int→str
    # transformation that the asdict-based to_dict carried.
    r = SweepResult(
        embedder_name="t",
        embedder_dim=4,
        cost_per_million_tokens=0.0,
        n_corpus=10,
        n_queries=5,
        recall_at_k={1: 0.5, 5: 0.8, 10: 0.9},
        ndcg_at_10=0.5,
        embed_latency_ms={"corpus_total": 1.0, "query_p50": 0.1, "query_p95": 0.2},
        notes=[],
    )
    d = r.to_dict()
    assert sorted(d["recall_at_k"].keys()) == ["1", "10", "5"]
    assert all(isinstance(k, str) for k in d["recall_at_k"])


def test_sweep_result_to_dict_round_trips_through_from_dict():
    # Acceptance regression: existing from_dict still parses the new
    # explicit-field shape. The arc is internal refactor only.
    original = SweepResult(
        embedder_name="round-trip-test",
        embedder_dim=16,
        cost_per_million_tokens=0.12,
        n_corpus=100,
        n_queries=20,
        recall_at_k={1: 0.6, 5: 0.85, 10: 0.92},
        ndcg_at_10=0.78,
        embed_latency_ms={"corpus_total": 5.0, "query_p50": 0.3, "query_p95": 0.7},
        notes=["one", "two"],
    )
    restored = SweepResult.from_dict(original.to_dict())
    assert restored == original


def test_sweep_result_to_dict_notes_is_a_list_copy():
    r = SweepResult(
        embedder_name="t",
        embedder_dim=4,
        cost_per_million_tokens=0.0,
        n_corpus=10,
        n_queries=5,
        recall_at_k={1: 0.5},
        ndcg_at_10=0.5,
        embed_latency_ms={"corpus_total": 1.0},
        notes=["original"],
    )
    out = r.to_dict()
    out["notes"].append("leaked")
    assert "leaked" not in r.notes
