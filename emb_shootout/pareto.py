"""Pareto frontier selection for the cost-vs-quality plot (issue #3).

The frontier on a (cost, recall@5) plane is the set of points for which no
*other* point is both cheaper AND higher-recall. The math is dep-free pure
Python so it tests in the standard CI matrix without pulling in matplotlib.

The plot renderer (`emb_shootout.plot`) consumes `pareto_frontier(results)`
and draws the connecting polyline only when the frontier has ≥2 distinct
points; with a single result the frontier is trivially that point and the
figure says so honestly.

Axes (D-008):
    x = cost_per_million_tokens (lower is better)
    y = recall_at_k[5]          (higher is better)
"""

from __future__ import annotations

from collections.abc import Sequence

from .sweep import SweepResult


def _recall_at_5(result: SweepResult) -> float:
    """Helper so the recall key access is in one place."""
    return float(result.recall_at_k.get(5, 0.0))


def _dominates(a: SweepResult, b: SweepResult) -> bool:
    """`a` dominates `b` iff a is no worse on both axes and strictly better on
    at least one (lower cost, higher recall@5). Ties on both axes do NOT
    dominate — both points stay on the frontier."""
    a_cost = a.cost_per_million_tokens
    b_cost = b.cost_per_million_tokens
    a_recall = _recall_at_5(a)
    b_recall = _recall_at_5(b)
    no_worse = a_cost <= b_cost and a_recall >= b_recall
    strictly_better = a_cost < b_cost or a_recall > b_recall
    return no_worse and strictly_better


def pareto_frontier(results: Sequence[SweepResult]) -> list[SweepResult]:
    """Return the non-dominated subset of `results`, sorted by cost ascending
    (then by recall@5 descending for stable tie-breaking).

    Empty input returns an empty list. A single result is its own frontier.
    Results with identical (cost, recall@5) coordinates are all kept — none
    dominates the other, and dropping ties silently would hide co-located
    providers from the plot.
    """
    frontier: list[SweepResult] = []
    for candidate in results:
        if any(_dominates(other, candidate) for other in results if other is not candidate):
            continue
        frontier.append(candidate)
    frontier.sort(key=lambda r: (r.cost_per_million_tokens, -_recall_at_5(r)))
    return frontier
