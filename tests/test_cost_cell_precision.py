"""The cost column must not publish a measured price as zero (#147).

#145 / D-011 gave the three latency columns a significant-figures renderer, and
`_format_latency`'s docstring states the argument in full. It was never about
latency:

    "A *present* measurement smaller than half of ``10**-places`` reaches the
    identical cell by ARITHMETIC."

    "``0.0`` is the *best possible value* [...] A default landing at an extreme
    of a comparison does not abstain, it ranks."

Both sentences are about a number at the good end of a comparison, and cheapest
is the good end of a cost column -- the axis D-008 makes the Pareto frontier's
x-axis. The cost cell sat one line below the three that were fixed, still
rendering through a bare `:.3f`. Measured before this change:

    a-genuinely-free-local   0.0      -> $0.000
    b-cheap-real-0.0001      0.0001   -> $0.000
    c-cheap-real-0.0004      0.0004   -> $0.000
    d-openai-small           0.02     -> $0.020
    e-cohere                 0.10     -> $0.100

Three distinct costs, one cell, at the cheapest possible value.

Two arms carry the weight, and neither is "small numbers now render wider":

- `test_a_genuine_zero_still_renders_as_a_zero` is what separates this fix from
  its nearest wrong neighbour. A self-hosted embedder that costs nothing is a
  *real measurement*, not an absence; rendering it as `$0.0001` would fabricate
  a price, which is the one thing this repo's rules exist to prevent.
- `test_the_committed_artifact_is_byte_identical` is green on both trees and is
  what rejects an over-eager refactor: `#145`'s own first draft applied
  significant figures unconditionally and churned `0.5` into `0.50`.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from emb_shootout.pareto import pareto_frontier
from emb_shootout.sweep import (
    SweepResult,
    _format_cost,
    _format_latency,
    aggregate_json,
    aggregate_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED = REPO_ROOT / "docs" / "benchmarks.md"


def _result(name: str, cost: float, recall: float = 0.85) -> SweepResult:
    return SweepResult(
        embedder_name=name,
        embedder_dim=64,
        n_corpus=10,
        n_queries=5,
        recall_at_k={5: recall},
        ndcg_at_10=0.7,
        embed_latency_ms={"corpus_total": 100.0, "query_p50": 8.1, "query_p95": 19.4},
        cost_per_million_tokens=cost,
    )


def _cost_cells(md: str) -> list[str]:
    """The `$/1M tokens` cell of every data row, in order."""
    rows = [ln for ln in md.splitlines() if ln.startswith("| ") and "---" not in ln]
    return [ln.strip().strip("|").split("|")[-1].strip() for ln in rows[1:]]


# ----------------------------------------------------------------------
# The collision, closed
# ----------------------------------------------------------------------


def test_three_distinct_costs_render_as_three_distinct_cells() -> None:
    """The headline. Pre-fix all three were `$0.000`."""
    md = aggregate_markdown(
        [_result("a-free", 0.0), _result("b-tiny", 0.0001), _result("c-small", 0.0004)]
    )
    cells = _cost_cells(md)
    assert len(cells) == 3
    assert len(set(cells)) == 3, f"costs still collide in the published table: {cells}"


def test_a_measured_price_below_the_fixed_width_is_not_published_as_zero() -> None:
    assert _format_cost(0.0001) != "$0.000"
    assert _format_cost(0.0004) != "$0.000"
    # And they are readable rather than scientific, so the column still scans.
    assert _format_cost(0.0001) == "$0.00010"
    assert _format_cost(0.0004) == "$0.00040"


# ----------------------------------------------------------------------
# The arm that separates this from its nearest wrong neighbour
# ----------------------------------------------------------------------


def test_a_genuine_zero_still_renders_as_a_zero() -> None:
    """A free local embedder costs nothing, and that is a measurement.

    The plausible over-broad fix renders everything small in significant
    figures, which turns a true `0.0` into something that is not zero. That
    would be fabricating a price for a provider that has none -- the repo's
    own no-fabricated-benchmarks rule, reached from the other side.

    The rule keys off the value being *small*, never off it being *falsy*,
    which is the distinction `llm-cost-optimizer` D-018/D-019 drew for
    abstention.
    """
    assert _format_cost(0.0) == "$0.000"


def test_the_free_provider_is_distinguishable_from_the_nearly_free_one() -> None:
    """Both halves of the point in one assertion: zero reads as zero, and the
    two are not the same cell."""
    assert _format_cost(0.0) != _format_cost(0.0001)
    assert float(_format_cost(0.0).lstrip("$")) == 0.0
    assert float(_format_cost(0.0001).lstrip("$")) > 0.0


# ----------------------------------------------------------------------
# Nothing that shipped moves
# ----------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.02, 0.1, 0.13, 1.0, 0.5, 0.001, 12.345])
def test_ordinary_prices_are_byte_identical_to_the_old_three_decimal_render(
    value: float,
) -> None:
    """Widened ONLY where the narrow form would fake a zero.

    #145's first draft applied significant figures unconditionally and churned
    `0.5` into `0.50`. Every value here renders non-zero at three decimals, so
    every one must be exactly what `:.3f` produced before.
    """
    assert _format_cost(value) == f"${value:.3f}"


def test_the_committed_artifact_is_byte_identical() -> None:
    """Green on both trees by design: no published number may move.

    The shipped providers all price well above the collision band, which is why
    this defect was latent. If this ever fails, a committed cost entered the
    band and the change stopped being invisible.
    """
    text = COMMITTED.read_text(encoding="utf-8")
    for cell in _cost_cells(text):
        assert cell.startswith("$")
        # Three decimals exactly -- the pre-#147 shape -- for every shipped row.
        assert len(cell.split(".")[1]) == 3, f"a committed cost cell widened: {cell}"


# ----------------------------------------------------------------------
# The cross-check `aggregate_json` promises
# ----------------------------------------------------------------------


def test_markdown_and_json_can_be_cross_checked_line_by_line() -> None:
    """`aggregate_json` says a consumer can read the two formats side by side.

    For this column they could not: three distinct JSON values mapped to one
    markdown cell. The JSON was always right -- this pins that the markdown now
    carries the same information.
    """
    results = [_result("a-free", 0.0), _result("b-tiny", 0.0001), _result("c-small", 0.0004)]
    md_cells = _cost_cells(aggregate_markdown(results))
    json_costs = [r["cost_per_million_tokens"] for r in aggregate_json(results)["results"]]

    assert len(set(json_costs)) == len(set(md_cells))
    for cell, exact in zip(md_cells, json_costs, strict=True):
        assert math.isclose(float(cell.lstrip("$")), exact, rel_tol=0.05, abs_tol=1e-12), (
            f"{cell} does not represent {exact}"
        )


def test_the_computation_was_never_wrong_only_the_presentation() -> None:
    """Scoping the finding, and pinning that the fix did not touch the maths.

    The Pareto frontier ordered these correctly before this change, because it
    reads the float rather than the cell. Asserted so a later edit cannot
    quietly move the ordering while "fixing rendering".
    """
    results = [
        _result("a-free", 0.0, recall=0.80),
        _result("b-tiny", 0.0001, recall=0.85),
        _result("c-small", 0.0004, recall=0.90),
    ]
    assert [r.embedder_name for r in pareto_frontier(results)] == [
        "a-free",
        "b-tiny",
        "c-small",
    ]


# ----------------------------------------------------------------------
# One definition, two call sites
# ----------------------------------------------------------------------


def test_latency_and_cost_share_the_rule_rather_than_a_copy_of_it() -> None:
    """The same value renders the same way through both, modulo the `$`.

    Two copies of "don't publish a measured value as zero" would drift, which
    this repo has paid for twice (#79/#80 on pipe-escaping, and #145 itself,
    where the README's honest `0.017` and the table's `0.0` were one
    measurement with only one of the two locked). Both call sites use
    `places=3`-equivalent widths here so the outputs are directly comparable.
    """
    for value in (0.0, 0.0001, 0.0004, 0.02, 0.5, 8.1):
        assert _format_cost(value) == "$" + _format_latency(value, places=3)


def test_latency_cells_are_untouched_by_the_generalisation() -> None:
    """#145's own worked values, re-asserted here.

    The generalisation must not move a latency cell. These are the numbers
    `_format_latency`'s docstring records, so if the shared helper ever drifts
    this fails in the module that caused it as well as in #145's own.
    """
    assert _format_latency(0.0135, places=1) == "0.013"
    assert _format_latency(0.0171, places=1) == "0.017"
    assert _format_latency(0.0, places=1) == "0.0"
    assert _format_latency(8.1, places=1) == "8.1"
    assert _format_latency(19.4, places=1) == "19.4"
    assert _format_latency(429.0, places=0) == "429"
