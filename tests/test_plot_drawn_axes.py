"""The axes the figure actually draws, not the ones `_axis_limits` returns (#139).

`docs/pareto.svg` — the figure this repo ships and the README links — carried x
ticks at **−0.04** and **−0.02**: negative cost per million tokens, a value
`SweepResult.__post_init__` refuses to store.

`_axis_limits` was right. It was not called::

    # Pad axes so labels don't clip.
    if len(results) > 1:
        (x_lo, x_hi), (y_lo, y_hi) = _axis_limits(results)

With one result no clamp applied at all and matplotlib picked the range, and
the committed result has ``cost_per_million_tokens: 0.0`` — so the shipped
figure was drawn on exactly that branch. Measured with the drawn axes captured:

    single result   drawn ylim          drawn xlim
    recall 0.85     (0.8032, 0.8968)    (0.0189, 0.0211)
    recall 0.99     (0.9356, 1.0445)    (0.0189, 0.0211)   <- above 1.0
    recall 1.0      (0.945,  1.055)     (0.0189, 0.0211)   <- above 1.0
    cost 0.0        (0.4725, 0.5275)    (-0.055, 0.055)    <- below 0.0
    two (control)   (0.83,   0.93)      (0.0,    0.18)     <- clamp working

**Why this file asserts on the drawn axes.** `tests/test_plot_title_and_axes.py`
tests `_axis_limits` directly and passes against the unfixed code, because the
function was never the problem — its *call site* was. A test of the return value
would be a vacuous regression here, so this one captures the `Axes` object
`render_pareto` builds and reads the limits off it.

That costs the `plot` extra, so the render-level tests skip without it. The
committed-artifact lock below does not: it reads the SVG as text and needs
nothing installed, which makes it the check that runs everywhere and the one
that would have caught this.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from emb_shootout.plot import _RECALL_DOMAIN, _axis_limits
from emb_shootout.sweep import SweepResult

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMITTED_SVG = REPO_ROOT / "docs" / "pareto.svg"

#: matplotlib writes each tick label into the SVG as an XML comment alongside
#: the glyph path, and uses U+2212 MINUS SIGN rather than ASCII `-`.
_TICK = re.compile(r"<!-- (−?-?[\d.]+) -->")


def _result(name: str, cost: float, recall5: float) -> SweepResult:
    return SweepResult(
        embedder_name=name,
        embedder_dim=64,
        cost_per_million_tokens=cost,
        n_corpus=100,
        n_queries=10,
        recall_at_k={1: recall5 * 0.8, 5: recall5, 10: recall5},
        ndcg_at_10=recall5,
        embed_latency_ms={"p50": 12.0, "p95": 20.0},
        notes="",
    )


def _drawn_limits(results: list[SweepResult], tmp_path: Path) -> tuple[tuple[float, float], ...]:
    """Render for real and read the limits off the `Axes` that was built.

    `render_pareto` closes its figure, so the axes are captured as it makes
    them rather than fetched afterwards from `plt.gcf()`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from emb_shootout.plot import render_pareto

    captured: list[object] = []
    real_subplots = plt.subplots

    def spy(*args: object, **kwargs: object) -> tuple[object, object]:
        fig, ax = real_subplots(*args, **kwargs)  # type: ignore[arg-type]
        captured.append(ax)
        return fig, ax

    plt.subplots = spy  # type: ignore[assignment]
    try:
        render_pareto(results, out_svg=tmp_path / "p.svg")
    finally:
        plt.subplots = real_subplots  # type: ignore[assignment]
    ax = captured[-1]
    return tuple(ax.get_xlim()), tuple(ax.get_ylim())  # type: ignore[attr-defined]


#: (case id, results). Every single-result row the issue measured, plus the
#: multi-result control that shows the clamp already worked there.
_CASES: list[tuple[str, list[SweepResult]]] = [
    ("single-mid-recall", [_result("a", 0.02, 0.85)]),
    ("single-low-recall", [_result("a", 0.02, 0.02)]),
    ("single-high-recall", [_result("a", 0.02, 0.99)]),
    ("single-perfect-recall", [_result("a", 0.02, 1.0)]),
    ("single-zero-recall", [_result("a", 0.02, 0.0)]),
    ("single-zero-cost", [_result("a", 0.0, 0.5)]),
    ("single-zero-cost-perfect-recall", [_result("a", 0.0, 1.0)]),
    ("two-results", [_result("a", 0.02, 0.85), _result("b", 0.13, 0.91)]),
    ("two-results-both-extremes", [_result("a", 0.0, 0.0), _result("b", 0.4, 1.0)]),
]


@pytest.mark.parametrize(("case", "results"), _CASES, ids=[c for c, _ in _CASES])
def test_the_drawn_axes_stay_inside_the_domains(
    case: str, results: list[SweepResult], tmp_path: Path
) -> None:
    pytest.importorskip("matplotlib")
    (x_lo, x_hi), (y_lo, y_hi) = _drawn_limits(results, tmp_path)
    lo, hi = _RECALL_DOMAIN
    assert x_lo >= 0.0, f"{case}: x axis runs to {x_lo}, below the cost floor"
    assert y_lo >= lo, f"{case}: y axis runs to {y_lo}, below the recall domain"
    assert y_hi <= hi, f"{case}: y axis runs to {y_hi}, above the recall domain"


@pytest.mark.parametrize(("case", "results"), _CASES, ids=[c for c, _ in _CASES])
def test_the_drawn_axes_are_exactly_what_axis_limits_returns(
    case: str, results: list[SweepResult], tmp_path: Path
) -> None:
    """The call site delegates rather than deciding.

    A neighbour that clamps at the call site instead — `max(0.0, x_lo)` next to
    `set_xlim` — satisfies the domain assertions above and puts a second copy
    of the domain in the module.
    """
    pytest.importorskip("matplotlib")
    assert _drawn_limits(results, tmp_path) == _axis_limits(results)


@pytest.mark.parametrize(("case", "results"), _CASES, ids=[c for c, _ in _CASES])
def test_a_single_point_still_gets_a_readable_range(
    case: str, results: list[SweepResult], tmp_path: Path
) -> None:
    """The reason the pads have floors, asserted rather than assumed.

    Applying `_axis_limits` unconditionally is only safe because a zero spread
    still yields a real range — `max(0.05, spread * 0.08)` and
    `max(0.02, spread * 0.08)`. Without those floors a single point would draw a
    zero-width axis, which is the failure mode the `len(results) > 1` condition
    reads as guarding against.
    """
    pytest.importorskip("matplotlib")
    (x_lo, x_hi), (y_lo, y_hi) = _drawn_limits(results, tmp_path)
    assert x_hi > x_lo, f"{case}: zero-width x axis"
    assert y_hi > y_lo, f"{case}: zero-width y axis"


def test_the_multi_result_output_is_unchanged() -> None:
    """The fix touches the shared path, so pin the branch that already worked."""
    results = [_result("a", 0.02, 0.85), _result("b", 0.13, 0.91)]
    assert _axis_limits(results) == ((0.0, 0.18), (0.83, 0.93))


# ---------------------------------------------------------------------------
# The committed artifact
# ---------------------------------------------------------------------------


def test_the_shipped_figure_has_no_tick_outside_a_domain() -> None:
    """The check that would have caught this, and it needs nothing installed.

    `docs/pareto.svg` is committed and linked from the README, and it carried
    `−0.04` / `−0.02` on the cost axis. Reading the artifact as text is the only
    check here that runs in the matrix without the `plot` extra — the render
    tests above skip without matplotlib, which is how a wrong figure shipped.
    """
    assert COMMITTED_SVG.exists(), "docs/pareto.svg is committed and linked from the README"
    ticks = _TICK.findall(COMMITTED_SVG.read_text(encoding="utf-8"))
    # Anti-vacuous: a regex that stopped matching would pass on an empty list,
    # and this lock's whole value is that it reads the shipped bytes.
    assert len(ticks) >= 6, f"found {len(ticks)} tick labels; the scan is not reading the figure"

    negative = [t for t in ticks if t.startswith("−") or t.startswith("-")]
    assert negative == [], (
        "docs/pareto.svg has negative axis ticks; neither cost nor recall can be "
        f"negative, so the figure is drawn outside the data's own domain: {negative}"
    )
    above_one = [t for t in ticks if float(t.replace("−", "-")) > 1.0]
    # Cost has no ceiling, so a >1.0 tick is only wrong on the recall axis; this
    # asserts the committed figure's numbers are all in range for *both* axes,
    # which holds while the sole committed result is a sub-dollar cost.
    assert above_one == [], f"unexpected tick above 1.0 in the committed figure: {above_one}"


def test_the_tick_scan_recognises_a_bad_figure() -> None:
    """The lock's own falsification.

    A scan asserting "no negative ticks" over a figure that has none is
    satisfied by a broken regex. This feeds it the shape the repo actually
    shipped — matplotlib's U+2212 minus, not an ASCII hyphen, which is the
    detail a hand-written check gets wrong.
    """
    shipped_before = "<!-- −0.04 --> <!-- −0.02 --> <!-- 0.00 -->"
    ticks = _TICK.findall(shipped_before)
    assert ticks == ["−0.04", "−0.02", "0.00"]
    assert [t for t in ticks if t.startswith("−")] == ["−0.04", "−0.02"]
