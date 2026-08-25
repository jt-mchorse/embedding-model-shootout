"""The two aggregate formats must agree on which cells were never measured (#127).

`#123` established the rule: `.get(k, 0.0)` "PUBLISHED A NUMBER FOR A MEASUREMENT
NEVER TAKEN … `0.000` … is indistinguishable from a real measurement of zero.
Handoff §10: do not invent benchmark numbers." It shipped `_recall_cell`, which
renders an em dash — for `recall_at_k` in `aggregate_markdown` only.

Enumerating the sites the rule applies to, that was one cell of eight:

    |                     | recall_at_k | corpus_total | query_p50 | query_p95 |
    | aggregate_markdown  | fixed (—)   | .get(…, 0.0) | .get(…,0.0)| .get(…,0.0)|
    | aggregate_json      | .get(k,0.0) | .get(…, 0.0) | .get(…,0.0)| .get(…,0.0)|

Two harms followed. The formats disagreed about the same cell — markdown `—`,
JSON `0.0` — while `aggregate_json`'s docstring promises a consumer can
"cross-check the two formats line-by-line". And for latency the fabricated `0.0`
is the *best possible value*, so a provider that reported no timings won any
"which is fastest" read of the published benchmark:

    | b-no-latency | ... | 0.850 |    0 | 0.0 |  0.0 | $0.100 |
    | full-model   | ... | 0.720 | 1234 | 8.1 | 19.4 | $0.130 |

This module asserts the parity *differentially* — it derives the absent cells from
each format's own output and compares them — rather than restating the rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emb_shootout.sweep import (
    ABSENT_RECALL_CELL,
    SweepResult,
    aggregate_json,
    aggregate_markdown,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Column label -> JSON key, for the three non-recall columns under test.
_LATENCY_COLUMNS = {
    "corpus embed (ms)": "corpus_embed_ms",
    "query p50 (ms)": "query_p50_ms",
    "query p95 (ms)": "query_p95_ms",
}


def _result(name: str, *, recall: dict[int, float], latency: dict[str, float]) -> SweepResult:
    return SweepResult(
        embedder_name=name,
        embedder_dim=768,
        cost_per_million_tokens=0.13,
        n_corpus=100,
        n_queries=50,
        recall_at_k=recall,
        ndcg_at_10=0.72,
        embed_latency_ms=latency,
    )


FULL_RECALL = {1: 0.4, 5: 0.7, 10: 0.8}
FULL_LATENCY = {"corpus_total": 1234.0, "query_p50": 8.1, "query_p95": 19.4}

# Every row is a real shape `SweepResult.from_dict` accepts off an external
# result file, which is what `sweep aggregate` does to every `results/*.json`.
RESULTS = [
    _result("a-complete", recall=FULL_RECALL, latency=FULL_LATENCY),
    _result("b-no-k5", recall={1: 0.35, 10: 0.75}, latency=FULL_LATENCY),
    _result("c-no-latency", recall=FULL_RECALL, latency={}),
    _result("d-partial-latency", recall=FULL_RECALL, latency={"corpus_total": 900.0}),
    _result("e-no-recall-at-all", recall={}, latency=FULL_LATENCY),
]


def _markdown_cells(results: list[SweepResult]) -> dict[tuple[str, str], str]:
    """`(embedder, column label) -> rendered cell`, parsed back out of the table."""
    lines = aggregate_markdown(results).strip().splitlines()
    headers = [h.strip() for h in lines[0].strip("|").split("|")]
    cells: dict[tuple[str, str], str] = {}
    for line in lines[2:]:
        values = [v.strip() for v in line.strip("|").split("|")]
        assert len(values) == len(headers), (
            f"row has {len(values)} cells but the header has {len(headers)}: {line!r}"
        )
        name = values[0]
        for header, value in zip(headers[1:], values[1:], strict=True):
            cells[(name, header)] = value
    return cells


def test_the_fixture_covers_both_absent_kinds_and_a_control() -> None:
    """Anti-vacuous: without an all-measured row, a change that marked everything
    absent would satisfy every parity assertion below."""
    assert any(r.recall_at_k == FULL_RECALL and r.embed_latency_ms == FULL_LATENCY for r in RESULTS)
    assert any(5 not in r.recall_at_k for r in RESULTS)
    assert any(not r.embed_latency_ms for r in RESULTS)
    assert any(r.embed_latency_ms and "query_p50" not in r.embed_latency_ms for r in RESULTS)


def test_markdown_and_json_agree_on_which_recall_cells_are_absent() -> None:
    md = _markdown_cells(RESULTS)
    payload = aggregate_json(RESULTS)
    by_name = {row["embedder"]: row for row in payload["results"]}

    compared = 0
    for name, row in by_name.items():
        for k in payload["ks"]:
            md_absent = md[(name, f"recall@{k}")] == ABSENT_RECALL_CELL
            json_absent = row["recall"][str(k)] is None
            assert md_absent == json_absent, (
                f"{name} recall@{k}: markdown absent={md_absent}, json absent={json_absent}"
            )
            compared += 1
    assert compared >= 12, "expected the cross-product of results and ks"


def test_markdown_and_json_agree_on_which_latency_cells_are_absent() -> None:
    md = _markdown_cells(RESULTS)
    by_name = {row["embedder"]: row for row in aggregate_json(RESULTS)["results"]}

    for name, row in by_name.items():
        for label, key in _LATENCY_COLUMNS.items():
            md_absent = md[(name, label)] == ABSENT_RECALL_CELL
            json_absent = row[key] is None
            assert md_absent == json_absent, (
                f"{name} {label}: markdown absent={md_absent}, json absent={json_absent}"
            )


def test_a_measured_zero_is_still_rendered_as_zero() -> None:
    """The distinction the whole issue is about: absent is not zero, and zero is
    not absent. A guard that rendered every falsy value as absent would pass the
    parity tests above and be just as wrong."""
    zeroed = _result(
        "z-measured-zero",
        recall={1: 0.0, 5: 0.0, 10: 0.0},
        latency={"corpus_total": 0.0, "query_p50": 0.0, "query_p95": 0.0},
    )
    md = _markdown_cells([zeroed])
    row = aggregate_json([zeroed])["results"][0]

    for k in (1, 5, 10):
        assert md[("z-measured-zero", f"recall@{k}")] == "0.000"
        assert row["recall"][str(k)] == 0.0
    for label, key in _LATENCY_COLUMNS.items():
        assert md[("z-measured-zero", label)] not in {ABSENT_RECALL_CELL, ""}
        assert row[key] == 0.0


def test_json_rows_always_carry_every_key() -> None:
    """`null`, not an omitted key. A missing key and a null key are different
    contracts: a consumer reading `row["recall"]["5"]` gets a `KeyError` from
    omission but a readable `None` from null, and the column set is a property of
    the aggregate (the union of every result's k), not of the individual row."""
    payload = aggregate_json(RESULTS)
    for row in payload["results"]:
        assert set(row["recall"]) == {str(k) for k in payload["ks"]}
        for key in _LATENCY_COLUMNS.values():
            assert key in row


def test_the_table_stays_column_aligned_for_every_row() -> None:
    """`#83`'s alignment invariant, re-asserted because this change rewrote how
    the latency cells are spliced. `_markdown_cells` asserts it per row; this
    names it, and covers the empty-`ks` case #83 was about."""
    _markdown_cells(RESULTS)
    _markdown_cells([_result("only", recall={}, latency={})])


@pytest.mark.parametrize("fmt", ["markdown", "json"])
def test_the_committed_artifact_does_not_move(fmt: str) -> None:
    """`results/hash.json` carries all three ks and all three latency keys, so it
    has no absent cells and nothing published should change."""
    files = sorted((REPO_ROOT / "results").glob("*.json"))
    assert files, "expected a committed result to compare against"
    results = [SweepResult.from_dict(json.loads(p.read_text(encoding="utf-8"))) for p in files]

    if fmt == "markdown":
        rendered = aggregate_markdown(results)
        assert ABSENT_RECALL_CELL not in rendered
        assert rendered in (REPO_ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    else:
        payload = aggregate_json(results)
        flat = json.dumps(payload)
        assert "null" not in flat, "the committed result has no unmeasured cells"
