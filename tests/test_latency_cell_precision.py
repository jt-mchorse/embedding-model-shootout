"""A measured latency must not be published as `0.0` (#145).

`#127` gave the latency columns an em dash for an **absent** measurement, and its
docstring states the reason in full:

    for latency the fabricated default is worse than merely wrong, because `0.0`
    is the *best possible value*. A provider that reported no timings won any
    "which is fastest" read of the published benchmark [...] A default landing at
    an extreme of a comparison does not abstain, it ranks.

That argument is about the **observable**, and `#127` closed only the path where
`0.0` arrives as a default. A *present* measurement below half of `10**-places`
reaches the identical cell by arithmetic. Measured on ``295a88b`` at ``places=1``,
which is what the markdown table uses for p50/p95::

    ABSENT                                    ' — '
    present 0.0135 ms  (committed query_p50)  ' 0.0 '
    present 0.0171 ms  (committed query_p95)  ' 0.0 '
    present 0.04 ms                           ' 0.0 '
    a genuine 0.0 ms                          ' 0.0 '   <- the collision
    present 8.1 ms     (#127's own control)   ' 8.1 '

Both of the committed result's query latencies sit in that band, so the published
table read ``0.0 | 0.0`` for a provider that measured 0.0135 and 0.0171 ms — while
``README.md`` quotes the honest ``0.017 ms`` for the same measurement, and only the
README half was locked (``test_readme_snapshot.py``).

The fix is significant figures, not a wider fixed ``places``: a wider fixed width
**moves** the collision band instead of removing it, so ``places=3`` would publish
a 0.0004 ms value as ``0.000``. That neighbour is built and run below.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from emb_shootout.sweep import ABSENT_RECALL_CELL, _format_latency, _latency_cell

_REPO_ROOT = Path(__file__).resolve().parent.parent
HASH_RESULTS = _REPO_ROOT / "results" / "hash.json"
README = _REPO_ROOT / "README.md"
BENCHMARKS_MD = _REPO_ROOT / "docs" / "benchmarks.md"


def _cell(value: float | None, places: int = 1) -> str:
    payload: dict[str, float] = {} if value is None else {"query_p50": value}
    return _latency_cell(payload, "query_p50", places=places).strip(" |")


# --- the collision ----------------------------------------------------------

#: Realistic magnitudes spanning a hash embedder (microseconds) to a remote API
#: call (tens of ms). Every pair of DIFFERENT values must render differently.
_MAGNITUDES = [0.0, 0.0004, 0.0135, 0.0171, 0.04, 0.05, 0.5, 1.0, 8.1, 19.4, 429.23]


def test_no_two_different_latencies_render_to_the_same_cell() -> None:
    """The property, over a swept table rather than the two rows that bit us.

    A test naming only `0.0171` and `0.0` would be satisfied by a fix that moves
    the collision band somewhere else in the range, which is what a wider fixed
    `places` does.
    """
    rendered = {value: _cell(value) for value in _MAGNITUDES}
    collisions: dict[str, list[float]] = {}
    for value, cell in rendered.items():
        collisions.setdefault(cell, []).append(value)
    clashing = {cell: vs for cell, vs in collisions.items() if len(vs) > 1}
    assert clashing == {}, (
        f"different measured latencies render to the same published cell: {clashing}. "
        f"`0.0` is the best possible value in this column, so a collision there does "
        f"not abstain, it ranks (#127's own argument, #145)."
    )


def test_a_genuine_zero_is_distinguishable_from_a_small_measurement() -> None:
    assert _cell(0.0) != _cell(0.0135)
    assert _cell(0.0) != _cell(0.04)
    # And from the absent case, which is #127's contribution and must survive.
    assert _cell(None) == ABSENT_RECALL_CELL
    assert _cell(0.0) != _cell(None)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(8.1, "8.1", id="127-control-8.1"),
        pytest.param(19.4, "19.4", id="127-control-19.4"),
        pytest.param(0.5, "0.5", id="half-a-ms"),
        pytest.param(1.0, "1.0", id="one-ms"),
    ],
)
def test_every_cell_127_reasoned_about_is_unchanged(value: float, expected: str) -> None:
    """The fix must not churn the column it is not about.

    #127's docstring shows a worked table with `8.1` and `19.4`; those have to
    render identically or this change is a reformat wearing a bug fix's clothes.
    """
    assert _cell(value) == expected


def test_the_corpus_total_column_is_unchanged() -> None:
    """`places=0`, the widest column, must also be untouched for normal values."""
    payload = {"corpus_total": 429.2312499601394}
    assert _latency_cell(payload, "corpus_total", places=0).strip(" |") == "429"


def test_a_non_finite_latency_renders_the_absent_cell_rather_than_raising() -> None:
    """`from_dict` accepts an external result file, and `log10(nan)` raises.

    A renderer is the wrong place to discover a malformed input, and a `nan` cell
    would be a published non-number either way.
    """
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert _cell(bad) == ABSENT_RECALL_CELL


# --- the two published surfaces must agree ----------------------------------


def test_the_table_cell_matches_the_number_the_readme_quotes() -> None:
    """README and table, for one measurement.

    The README says "per-query p95 of **0.017 ms**" and `test_readme_snapshot.py`
    pins that to `results/hash.json`. Nothing pinned the table, so the README said
    `0.017` and the table said `0.0` — both committed, both published, and the lock
    watched only the one that happened to be right.
    """
    measured = json.loads(HASH_RESULTS.read_text(encoding="utf-8"))["embed_latency_ms"]["query_p95"]
    cell = _latency_cell(measured_dict := {"query_p95": measured}, "query_p95", places=1).strip(
        " |"
    )
    assert measured_dict  # keep the binding readable
    readme = README.read_text(encoding="utf-8")
    quoted = re.search(r"per-query p95 of \*\*([0-9.]+) ms\*\*", readme)
    assert quoted, "the README's p95 claim moved; this lock pairs it with the table cell"
    assert cell == quoted.group(1), (
        f"the table renders {cell!r} for query_p95 while the README quotes "
        f"{quoted.group(1)!r} for the same measurement"
    )
    assert cell in BENCHMARKS_MD.read_text(encoding="utf-8"), (
        "the committed table does not carry the cell this renderer produces; "
        "regenerate docs/benchmarks.md"
    )


# --- the neighbours ---------------------------------------------------------


def test_a_wider_fixed_places_only_moves_the_collision_band() -> None:
    """The plausible fix, built and run.

    `places=3` separates the two rows that prompted #145 and re-creates the same
    defect one decade down — which is why the shipped rule is significant figures.
    """
    fixed3 = lambda v: f"{v:.3f}"  # noqa: E731 - the neighbour, inline on purpose
    assert fixed3(0.0135) != fixed3(0.0)  # it does fix the rows we noticed
    assert fixed3(0.0004) == fixed3(0.0) == "0.000", (
        "a wider fixed width must still collide somewhere, or this neighbour is "
        "not the one #145 argues against"
    )
    # The shipped rule does not.
    assert _cell(0.0004) != _cell(0.0)


def test_significant_figures_is_the_rule_and_not_a_hardcoded_pair() -> None:
    """Discovered over decades, so the rule holds away from the observed values.

    Two claims with deliberately different scopes. Non-collision with a genuine
    zero is asserted at *every* magnitude — that is the defect. Round-trip
    fidelity is asserted only where the widening applies: at `places=1` a value of
    `0.17` renders `0.2` and should, because that is the column's declared
    precision and `0.2` is not zero. Asserting fidelity everywhere would be
    asserting that the column has no precision at all, and a first draft of this
    test did exactly that and failed on `0.17`.
    """
    for exponent in range(-6, 3):
        value = 1.7 * 10.0**exponent
        cell = _cell(value)
        assert cell != _cell(0.0), f"1.7e{exponent} collided with a genuine zero"
        if float(f"{value:.1f}") == 0.0:
            # The widened band: the cell must carry the measurement, not a
            # rounding of it to nothing.
            assert float(cell) == pytest.approx(value, rel=0.1), (
                f"1.7e{exponent} rendered as {cell!r}, which is not that number"
            )
        else:
            # Outside it, the narrow form is correct by definition.
            assert cell == f"{value:.1f}"


def test_format_latency_keeps_at_least_places_decimals() -> None:
    """The fix widens, never narrows: an existing column cannot get shorter."""
    for places in (0, 1, 3):
        for value in (8.1, 19.4, 429.23, 1.0):
            out = _format_latency(value, places=places)
            decimals = len(out.partition(".")[2])
            assert decimals >= places, f"{value} at places={places} rendered {out!r}"
            assert math.isclose(float(out), value, rel_tol=0.05)
