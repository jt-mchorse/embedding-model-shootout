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

#: The frontier's quality axis, fixed by D-008 (`cost_per_million_tokens` x
#: `recall_at_k[5]`). Named rather than inlined so the two places that need it
#: — the accessor and the precondition check — cannot drift apart.
FRONTIER_K = 5


def _recall_at_5(result: SweepResult) -> float:
    """Recall on the frontier's quality axis. Raises if it was never measured.

    This used to be ``recall_at_k.get(5, 0.0)``. ``0.0`` is the *worst possible*
    value on this axis, so a result swept without ``k=5`` was not skipped or
    flagged — it was ranked against every other model as though it had scored
    zero (#123).

    ``run_sweep(..., k_values=...)`` is an exported parameter and
    ``SweepResult.from_dict`` reconstructs from JSON with no requirement that
    ``5`` be present; ``sweep.py``'s own comments note an empty ``recall_at_k``
    is reachable that way.

    Measured, with a model whose real recall@1/3/10 is 0.95 / **0.97** / 0.99
    but which was swept at ``k_values=(1, 3, 10)``:

    - at $0.50/1M it was *dominated by* — and dropped from the frontier in
      favour of — a model scoring **0.12** at k=5.
    - at $0.02/1M it survived on the cost axis alone, and was plotted at
      ``0.0`` on the quality axis of the repo's headline figure.

    Raising is what honouring D-008 looks like. The decision fixes this axis as
    ``recall_at_5``; a result with no such measurement cannot be placed on it,
    and inventing a coordinate is the one thing a benchmark must not do
    (handoff §10). Parameterizing the axis would be a D-008 *revisit*, not a
    fix — see the issue.
    """
    try:
        return float(result.recall_at_k[FRONTIER_K])
    except KeyError:
        raise ValueError(
            f"{result.embedder_name!r} has no recall_at_k[{FRONTIER_K}], which is the "
            f"Pareto frontier's quality axis (D-008); it was measured at "
            f"k={sorted(result.recall_at_k) or 'nothing'}. Re-run the sweep with "
            f"{FRONTIER_K} in k_values — scoring it 0.0 would rank the model against "
            "a measurement that was never taken"
        ) from None


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

    Raises `ValueError` naming the embedder if any result lacks
    `recall_at_k[5]` (#123). Checked up front, before any comparison, so the
    error names the input rather than surfacing partway through a dominance
    scan with half a frontier already built.
    """
    for result in results:
        _recall_at_5(result)  # precondition: every point can be placed on the axis

    frontier: list[SweepResult] = []
    for candidate in results:
        if any(_dominates(other, candidate) for other in results if other is not candidate):
            continue
        frontier.append(candidate)
    frontier.sort(key=lambda r: (r.cost_per_million_tokens, -_recall_at_5(r)))
    return frontier
