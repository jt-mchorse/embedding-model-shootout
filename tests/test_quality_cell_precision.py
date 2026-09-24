"""`recall@k` and `NDCG@10` do not publish a measured value as zero (#149).

`#145` gave the latency columns a significant-figures renderer; `#147` gave the
cost column the same one, and stated the rule generally -- "`#145` wrote the
argument for latency and it was never about latency". Four columns on the same
row were still a bare `.3f`: one `recall@k` cell per `k`, and `NDCG@10`.

Measured before the fix, through the public constructor::

    a-genuinely-zero      recall 0.0      ndcg 0.0       -> 0.000 / 0.000
    b-one-hit-in-4000     recall 0.00025  ndcg 7.23e-05  -> 0.000 / 0.000
    c-four-hits-in-4000   recall 0.001    ndcg 0.000289  -> 0.001 / 0.000

`b` found the gold document for one query in four thousand and published a row
**byte-identical** to the one that found nothing, while `aggregate_json` carried
`0.00025` and `7.23e-05` for it throughout.

The direction differs from `#145`/`#147` and the difference is written into
`_format_quality`'s docstring rather than inherited silently: `0.000` is the
*worst* value on these columns, so truncation understates rather than flatters.
The half of the argument that transfers is the arithmetic one -- two distinct
present measurements, and a genuine zero, reaching one cell.
"""

from __future__ import annotations

import pytest

from emb_shootout.sweep import (
    ABSENT_RECALL_CELL,
    SweepResult,
    _format_quality,
    aggregate_json,
    aggregate_markdown,
)

_LATENCY = {"corpus_total": 10.0, "query_p50": 1.0, "query_p95": 2.0}


def _result(name: str, recall: float, ndcg: float, **kw) -> SweepResult:
    return SweepResult(
        embedder_name=name,
        embedder_dim=8,
        n_corpus=100,
        n_queries=4000,
        recall_at_k=kw.pop("recall_at_k", {1: recall, 5: recall, 10: recall}),
        ndcg_at_10=ndcg,
        embed_latency_ms=_LATENCY,
        cost_per_million_tokens=0.02,
        **kw,
    )


GENUINE_ZERO = _result("a-genuinely-zero", 0.0, 0.0)
ONE_HIT = _result("b-one-hit-in-4000", 0.00025, 7.23e-05)
FOUR_HITS = _result("c-four-hits-in-4000", 0.001, 0.000289)


def _row(result: SweepResult, table: str) -> str:
    [line] = [ln for ln in table.splitlines() if ln.startswith(f"| {result.embedder_name} ")]
    return line


# ----------------------------------------------------------------------
# The defect
# ----------------------------------------------------------------------


def test_a_measured_score_is_not_published_as_a_genuine_zero() -> None:
    """The row that found one document in four thousand is not the row that
    found none. Red against the pre-#149 renderer, where the two are equal."""
    table = aggregate_markdown([GENUINE_ZERO, ONE_HIT])
    zero_cells = _row(GENUINE_ZERO, table).split("|")[5:9]
    hit_cells = _row(ONE_HIT, table).split("|")[5:9]
    assert zero_cells != hit_cells
    assert [c.strip() for c in zero_cells] == ["0.000", "0.000", "0.000", "0.000"]
    assert [c.strip() for c in hit_cells] == ["0.00025", "0.00025", "0.00025", "0.000072"]


def test_ndcg_collapses_at_a_value_where_recall_survives() -> None:
    """The nDCG column is not redundant with the recall columns.

    `c` retrieved four documents in four thousand: recall renders `0.001` at the
    narrow width and survived the old code, while its nDCG of `0.000289` did
    not. An arm that only exercised recall would have left nDCG unfixed -- which
    is the shape this issue is itself a sibling of.
    """
    table = aggregate_markdown([FOUR_HITS])
    cells = [c.strip() for c in _row(FOUR_HITS, table).split("|")[5:9]]
    assert cells[:3] == ["0.001", "0.001", "0.001"]
    assert cells[3] == "0.00029"


# ----------------------------------------------------------------------
# What must NOT move: #127's three-way distinction
# ----------------------------------------------------------------------


def test_a_genuine_zero_keeps_the_narrow_form() -> None:
    """An embedder that retrieved nothing is a real measurement.

    Widening it to `0.00000` would say the opposite of what `_recall_cell`
    exists to say. Green on both trees, and it is what rejects the neighbour
    that widens unconditionally.
    """
    assert _format_quality(0.0) == "0.000"
    table = aggregate_markdown([GENUINE_ZERO])
    assert "| 0.000 | 0.000 | 0.000 | 0.000 |" in _row(GENUINE_ZERO, table)


def test_an_absent_k_still_renders_the_em_dash() -> None:
    """`#127`'s absent case never reaches the renderer.

    `_recall_cell` returns the em dash before calling `_format_quality`, so the
    three-way distinction — absent / genuine zero / small measurement — is
    intact. Green on both trees; it rejects the neighbour that routes the absent
    branch through the widening too.
    """
    partial = _result("d-partial", 0.0, 0.5, recall_at_k={1: 0.0, 10: 0.0})
    full = _result("e-full", 0.0, 0.5)
    table = aggregate_markdown([partial, full])
    assert ABSENT_RECALL_CELL in _row(partial, table)
    assert ABSENT_RECALL_CELL not in _row(full, table)


def test_ordinary_scores_are_byte_identical_to_the_narrow_form() -> None:
    """Nothing that never collided moves. This is why the committed artifact
    regenerates unchanged."""
    for value in (0.001, 0.32, 0.52, 0.62, 0.4486406869414832, 0.85, 1.0):
        assert _format_quality(value) == f"{value:.3f}"


# ----------------------------------------------------------------------
# The promise `aggregate_json` makes
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "result", [GENUINE_ZERO, ONE_HIT, FOUR_HITS], ids=lambda r: r.embedder_name
)
def test_the_two_formats_agree_on_these_rows(result: SweepResult) -> None:
    """`aggregate_json`'s docstring promises a consumer "can cross-check the two
    formats line-by-line". On these rows they could not: JSON said `0.00025`,
    markdown said `0.000`, and a *different* row also said `0.000`.

    Checked as "the markdown cell is the JSON value rendered", which is the
    strongest form the promise can take across a float and a fixed-width string.
    """
    rows = [GENUINE_ZERO, ONE_HIT, FOUR_HITS]
    table = aggregate_markdown(rows)
    payload = {r["embedder"]: r for r in aggregate_json(rows)["results"]}[result.embedder_name]
    cells = [c.strip() for c in _row(result, table).split("|")[5:9]]
    expected = [_format_quality(payload["recall"][str(k)]) for k in (1, 5, 10)]
    expected.append(_format_quality(payload["ndcg_at_10"]))
    assert cells == expected
