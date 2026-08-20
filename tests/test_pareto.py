"""Tests for the Pareto-frontier selection (`emb_shootout.pareto`).

Frontier math is dep-free pure Python — runs in standard CI with no extras.
"""

from __future__ import annotations

import pytest

from emb_shootout.pareto import pareto_frontier
from emb_shootout.sweep import SweepResult


def _make(name: str, cost: float, recall_at_5: float) -> SweepResult:
    return SweepResult(
        embedder_name=name,
        embedder_dim=128,
        cost_per_million_tokens=cost,
        n_corpus=100,
        n_queries=10,
        recall_at_k={1: 0.0, 5: recall_at_5, 10: recall_at_5},
        ndcg_at_10=recall_at_5,
        embed_latency_ms={"corpus_total": 0.0, "query_p50": 0.0, "query_p95": 0.0},
    )


def test_empty_input_returns_empty_list():
    assert pareto_frontier([]) == []


def test_single_point_is_its_own_frontier():
    only = _make("only", cost=1.0, recall_at_5=0.5)
    assert pareto_frontier([only]) == [only]


def test_dominated_point_excluded():
    cheap_good = _make("cheap_good", cost=0.1, recall_at_5=0.9)
    expensive_bad = _make("expensive_bad", cost=10.0, recall_at_5=0.3)
    frontier = pareto_frontier([cheap_good, expensive_bad])
    assert frontier == [cheap_good]


def test_two_non_dominated_points_both_kept():
    # Lower cost but lower recall, and higher cost but higher recall — classic tradeoff.
    cheap_meh = _make("cheap_meh", cost=0.1, recall_at_5=0.6)
    expensive_great = _make("expensive_great", cost=2.0, recall_at_5=0.9)
    frontier = pareto_frontier([cheap_meh, expensive_great])
    names = [r.embedder_name for r in frontier]
    assert set(names) == {"cheap_meh", "expensive_great"}
    # Sorted by cost ascending.
    assert names == ["cheap_meh", "expensive_great"]


def test_dominated_in_middle_excluded():
    a = _make("a", cost=0.1, recall_at_5=0.6)
    b_dominated = _make("b_dominated", cost=1.0, recall_at_5=0.5)  # dominated by both
    c = _make("c", cost=2.0, recall_at_5=0.9)
    frontier = pareto_frontier([a, b_dominated, c])
    assert [r.embedder_name for r in frontier] == ["a", "c"]


def test_identical_points_both_kept():
    # Tie on both axes — neither dominates the other, both stay on the frontier.
    a = _make("provider_a", cost=1.0, recall_at_5=0.8)
    b = _make("provider_b", cost=1.0, recall_at_5=0.8)
    frontier = pareto_frontier([a, b])
    assert {r.embedder_name for r in frontier} == {"provider_a", "provider_b"}


def test_tie_on_cost_higher_recall_dominates():
    cheap = _make("cheap", cost=1.0, recall_at_5=0.5)
    cheap_better = _make("cheap_better", cost=1.0, recall_at_5=0.8)
    frontier = pareto_frontier([cheap, cheap_better])
    assert [r.embedder_name for r in frontier] == ["cheap_better"]


def test_tie_on_recall_lower_cost_dominates():
    expensive = _make("expensive", cost=5.0, recall_at_5=0.8)
    cheap_same = _make("cheap_same", cost=1.0, recall_at_5=0.8)
    frontier = pareto_frontier([expensive, cheap_same])
    assert [r.embedder_name for r in frontier] == ["cheap_same"]


def test_frontier_sorted_by_cost_ascending():
    # Three frontier points returned out-of-order; assert post-sort.
    a = _make("a", cost=2.0, recall_at_5=0.7)
    b = _make("b", cost=0.5, recall_at_5=0.5)
    c = _make("c", cost=5.0, recall_at_5=0.9)
    frontier = pareto_frontier([a, b, c])
    assert [r.embedder_name for r in frontier] == ["b", "a", "c"]


def test_missing_recall_at_5_is_refused_not_treated_as_zero():
    """Deliberate reversal of a previously-asserted behaviour (#123).

    This test used to be `test_missing_recall_at_5_treated_as_zero` and
    asserted the frontier was `["has_recall"]` — i.e. that `no_recall` was
    dominated away. Its comment described the mechanism ("interpreted as 0 — it
    will be dominated by any point with any non-zero recall@5") but gave no
    reason to *want* it, and no core decision endorses it: D-008 fixes the axes
    as cost x recall@5 and says nothing about substituting a value for a
    measurement that was never taken.

    Scoring an unmeasured axis as 0.0 is not neutral — 0.0 is the worst
    possible value, so the substitution actively ranks the model. Measured on
    the pre-fix code, a model with real recall@1/3/10 of 0.95 / 0.97 / 0.99 was
    dropped from the frontier in favour of one scoring 0.12 at k=5. In a
    benchmark repo that is an invented number deciding a published result
    (handoff §10).

    Same scenario as before, opposite expectation.
    """
    no_recall = SweepResult(
        embedder_name="no_recall",
        embedder_dim=128,
        cost_per_million_tokens=1.0,
        n_corpus=100,
        n_queries=10,
        recall_at_k={1: 0.5, 10: 0.5},  # no 5
        ndcg_at_10=0.5,
        embed_latency_ms={"corpus_total": 0.0, "query_p50": 0.0, "query_p95": 0.0},
    )
    has_recall = _make("has_recall", cost=1.0, recall_at_5=0.3)
    with pytest.raises(ValueError, match="no_recall"):
        pareto_frontier([no_recall, has_recall])


def test_real_hash_baseline_alone_is_frontier():
    """The committed hash baseline (recall@5=0.52, cost=$0.000) alone."""
    hash_baseline = _make("hash-embedder-128d-ngram2", cost=0.0, recall_at_5=0.52)
    frontier = pareto_frontier([hash_baseline])
    assert len(frontier) == 1
    assert frontier[0].embedder_name == "hash-embedder-128d-ngram2"


@pytest.mark.parametrize("count", [0, 1, 5, 20])
def test_frontier_size_never_exceeds_input(count: int):
    """Property-style sanity check: |frontier| ≤ |results|."""
    results = [_make(f"p_{i}", cost=float(i), recall_at_5=0.5 + i * 0.01) for i in range(count)]
    frontier = pareto_frontier(results)
    assert len(frontier) <= len(results)
