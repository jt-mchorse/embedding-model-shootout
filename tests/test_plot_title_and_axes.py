"""The figure's caption and axis limits, checked without matplotlib (#137).

`render_pareto` chose its default title from `len(distinct_frontier_coords) < 2`
— a property of the **frontier** — while the sentence it selected
("all points co-located on at least one axis") claims something about **all
results**. The two coincide only when the frontier is the whole population, and
they come apart in the case this repo exists to produce: when one model
dominates every other, the frontier collapses to one point over results spread
across both axes, and the figure denied that any separation existed.

The same function clamped `set_ylim`'s upper bound to the recall domain and not
its lower, so a run of low recalls drew an axis running into negative recall —
a region `SweepResult.__post_init__` refuses to store.

**Why this file imports no matplotlib.** `emb_shootout/plot.py`'s render path
has no CI coverage (#111, gated on D-008), which is how a caption stating the
opposite of the data survived. The title and limits are pure functions of two
lists, so they are tested here in the standard matrix rather than waiting on
that decision. `render_pareto` itself still needs the `plot` extra; the
decisions it delegates do not.
"""

from __future__ import annotations

import pytest

from emb_shootout.pareto import pareto_frontier
from emb_shootout.plot import (
    _axis_limits,
    _default_title,
    _is_colocated,
    _should_draw_polyline,
)
from emb_shootout.sweep import SweepResult


def _result(name: str, cost: float, recall5: float) -> SweepResult:
    return SweepResult(
        embedder_name=name,
        embedder_dim=64,
        cost_per_million_tokens=cost,
        n_corpus=10,
        n_queries=5,
        recall_at_k={1: recall5, 5: recall5, 10: recall5},
        ndcg_at_10=recall5,
        embed_latency_ms={"p50": 1.0},
        notes=[],
    )


#: (label, results, actually_colocated). The five shapes the defect was
#: measured over. `actually_colocated` is stated here as ground truth so the
#: assertions below compare the title against the *data*, not against a
#: hardcoded string — a title test that pins strings would have passed happily
#: on the wrong caption.
CASES: list[tuple[str, list[SweepResult], bool]] = [
    ("one model dominates the other", [_result("a", 1.0, 0.90), _result("b", 2.0, 0.50)], False),
    (
        "three models, one dominates",
        [_result("a", 1.0, 0.95), _result("b", 2.0, 0.50), _result("c", 3.0, 0.40)],
        False,
    ),
    ("genuine two-point frontier", [_result("a", 1.0, 0.50), _result("b", 2.0, 0.90)], False),
    ("co-located on both axes", [_result("a", 1.0, 0.50), _result("b", 1.0, 0.50)], True),
    ("co-located on cost only", [_result("a", 1.0, 0.50), _result("b", 1.0, 0.90)], True),
    ("co-located on recall only", [_result("a", 1.0, 0.50), _result("b", 2.0, 0.50)], True),
    ("single result", [_result("a", 1.0, 0.50)], True),
]

_IDS = [c[0] for c in CASES]


def test_the_case_table_covers_both_verdicts() -> None:
    """Anti-vacuous: a table that drifted to all-colocated or all-separated
    would make every assertion below pass while proving nothing."""
    colocated = [label for label, _, flag in CASES if flag]
    separated = [label for label, _, flag in CASES if not flag]
    assert len(colocated) >= 3, colocated
    assert len(separated) >= 3, separated


def test_at_least_one_case_has_a_one_point_frontier_over_separated_results() -> None:
    """The defect's exact shape. Without a row like this the whole file is a
    restatement of behaviour that was already correct."""
    matches = [
        label
        for label, results, colocated in CASES
        if not colocated and len(results) > 1 and len(pareto_frontier(results)) == 1
    ]
    assert matches, "no row reproduces a one-point frontier over separated results"


@pytest.mark.parametrize(("label", "results", "colocated"), CASES, ids=_IDS)
def test_is_colocated_matches_the_data(
    label: str, results: list[SweepResult], colocated: bool
) -> None:
    assert _is_colocated(results) is colocated, label


@pytest.mark.parametrize(("label", "results", "colocated"), CASES, ids=_IDS)
def test_the_title_only_claims_colocation_when_the_results_are_colocated(
    label: str, results: list[SweepResult], colocated: bool
) -> None:
    """The assertion the old code failed.

    Stated as an implication against ground truth rather than as an expected
    string: the caption may be reworded, but it may never claim co-location of
    results that are not co-located.
    """
    title = _default_title(results, pareto_frontier(results))
    if not colocated:
        assert "co-located" not in title, f"{label}: {title!r} claims co-location falsely"


@pytest.mark.parametrize(("label", "results", "colocated"), CASES, ids=_IDS)
def test_the_title_does_claim_colocation_when_they_are(
    label: str, results: list[SweepResult], colocated: bool
) -> None:
    """Mirror assertion. Without it, deleting the co-located branch entirely
    passes the test above — and 'never says co-located' is not the fix."""
    if colocated and len(results) > 1:
        title = _default_title(results, pareto_frontier(results))
        assert "co-located" in title, f"{label}: {title!r} lost the co-location caption"


def test_a_dominating_model_is_named_in_the_title() -> None:
    """The case the old caption got backwards now says what happened."""
    results = [_result("voyage-3", 1.0, 0.95), _result("cheap-2", 2.0, 0.50)]
    title = _default_title(results, pareto_frontier(results))
    assert "voyage-3" in title
    assert "dominates" in title
    assert "co-located" not in title


def test_the_single_result_caption_is_unchanged() -> None:
    """Pinned because `docs/pareto.svg` carries this exact sentence today; the
    committed artifact must not churn on an unrelated fix."""
    results = [_result("a", 1.0, 0.5)]
    assert _default_title(results, pareto_frontier(results)) == (
        "Pareto frontier — single point (real-provider runs pending)"
    )


def test_a_caller_supplied_title_is_still_honoured() -> None:
    """`_default_title` is only consulted when `title is None`; this pins that
    the new branch did not become unconditional."""
    import inspect

    from emb_shootout import plot

    src = inspect.getsource(plot.render_pareto)
    assert "if title is None:" in src
    assert "_default_title(results, frontier)" in src


# ---------------------------------------------------------------------------
# Axis limits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "results"),
    [(label, results) for label, results, _ in CASES if len(results) > 1],
    ids=[label for label, results, _ in CASES if len(results) > 1],
)
def test_the_y_axis_never_leaves_the_recall_domain(label: str, results: list[SweepResult]) -> None:
    """`SweepResult.__post_init__` enforces `0.0 <= recall <= 1.0` on every
    value, so the axis showing those values may not run past either end."""
    _, (y_lo, y_hi) = _axis_limits(results)
    assert y_lo >= 0.0, f"{label}: y-axis starts at {y_lo}, below the recall domain"
    assert y_hi <= 1.0, f"{label}: y-axis ends at {y_hi}, above the recall domain"


def test_the_lower_y_clamp_actually_fires_on_low_recalls() -> None:
    """The measured case, and the anti-vacuous arm for the clamp: with recalls
    near zero the padding *wants* to go negative, so a row that never reaches
    the clamp would pass the domain test above without exercising it."""
    results = [_result("a", 1.0, 0.00), _result("b", 2.0, 0.02)]
    _, (y_lo, y_hi) = _axis_limits(results)
    # Unclamped this was (-0.02, 0.04).
    assert y_lo == 0.0
    assert y_hi == pytest.approx(0.04)


def test_the_upper_y_clamp_still_fires_on_high_recalls() -> None:
    """The clamp that already existed must survive the symmetric addition."""
    results = [_result("a", 1.0, 0.95), _result("b", 2.0, 0.99)]
    _, (y_lo, y_hi) = _axis_limits(results)
    assert y_hi == 1.0
    assert y_lo == pytest.approx(0.93)


def test_the_x_axis_never_starts_below_zero_cost() -> None:
    """`cost_per_million_tokens` is validated `>= 0.0`; same argument, cost
    axis. No upper clamp: cost has no documented ceiling."""
    results = [_result("a", 0.0, 0.5), _result("b", 0.01, 0.9)]
    (x_lo, x_hi), _ = _axis_limits(results)
    assert x_lo == 0.0
    assert x_hi > 0.0


def test_padding_still_separates_the_extremes_from_the_frame() -> None:
    """Anti-vacuous partner to the clamps: an `_axis_limits` that returned the
    raw min/max would satisfy every domain assertion above and reintroduce the
    clipped-label problem the padding exists to prevent."""
    results = [_result("a", 1.0, 0.40), _result("b", 3.0, 0.60)]
    (x_lo, x_hi), (y_lo, y_hi) = _axis_limits(results)
    assert x_lo < 1.0
    assert x_hi > 3.0
    assert y_lo < 0.40
    assert y_hi > 0.60


# ---------------------------------------------------------------------------
# The polyline predicate, which must NOT follow the title
# ---------------------------------------------------------------------------
#
# The trap in fixing #137: once the title stops asking a frontier question, the
# tempting next move is to make the polyline "consistent" and ask the results
# question too. Swapping `_should_draw_polyline(frontier)` for
# `not _is_colocated(results)` passed every other assertion in this file — it
# is the plausible neighbouring wrong fix, and these are the rows that separate
# them.


def test_colocated_results_always_collapse_the_frontier_to_one_point() -> None:
    """Measured while writing this file, and it constrains what follows.

    If every result shares one axis value, the best result on the *other* axis
    dominates all of them, so the frontier holds exactly one **distinct
    coordinate**. That means `_is_colocated(results)` implies
    `not _should_draw_polyline(frontier)` — the two predicates cannot disagree
    in that direction at all.

    Distinct *coordinates*, not frontier *length*: results at identical
    coordinates all stay on the frontier (`pareto_frontier`'s docstring —
    "dropping ties silently would hide co-located providers from the plot"), so
    the co-located-on-both-axes row has a two-element frontier and one point to
    draw. Confusing those two units is the defect this whole issue is about,
    one level down.
    """
    for label, results, colocated in CASES:
        if colocated and len(results) > 1:
            frontier = pareto_frontier(results)
            coords = {(r.cost_per_million_tokens, r.recall_at_k[5]) for r in frontier}
            assert len(coords) == 1, f"{label}: {len(coords)} distinct frontier coordinates"
            assert _should_draw_polyline(frontier) is False, label


def test_the_predicates_disagree_exactly_on_the_dominating_case() -> None:
    """The one direction they *do* come apart, stated precisely.

    Substituting `not _is_colocated(results)` for `_should_draw_polyline(
    frontier)` — the tempting "make it consistent" move once the title stops
    asking a frontier question — draws a polyline through a **single** frontier
    point whenever one model dominates separated results. The rendered result
    is a degenerate line, so nothing looks wrong; the parametrized test below is
    what actually fails on the substitution.

    Asserting the disagreement here rather than a harm, because there is no
    visual harm to assert and inventing one would be worse than saying so.
    """
    results = [_result("a", 1.0, 0.90), _result("b", 2.0, 0.50)]
    frontier = pareto_frontier(results)
    assert len(frontier) == 1
    assert _is_colocated(results) is False  # results predicate says "draw"
    assert _should_draw_polyline(frontier) is False  # frontier predicate says "don't"


@pytest.mark.parametrize(("label", "results", "colocated"), CASES, ids=_IDS)
def test_polyline_predicate_matches_distinct_frontier_coordinates(
    label: str, results: list[SweepResult], colocated: bool
) -> None:
    """The predicate is exactly "≥2 distinct frontier coordinates" — computed
    here independently rather than by calling the function under test."""
    frontier = pareto_frontier(results)
    coords = {(r.cost_per_million_tokens, r.recall_at_k[5]) for r in frontier}
    assert _should_draw_polyline(frontier) is (len(coords) >= 2), label
