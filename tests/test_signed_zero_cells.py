"""A negative zero publishes as a zero, not as a negative number (#224).

`-0.0 == 0.0`, so it passes `SweepResult`'s `>= 0.0` guards and ties with
`0.0` on the Pareto comparator -- but `format` keeps the sign bit, and the
shared cell renderer formatted the value it was given. Measured on cc40a48:

    _format_cost(-0.0)               '$-0.000'
    _format_quality(-0.0)            '-0.000'
    _format_latency(-0.0, places=0)  '-0'
    _format_latency(-0.0, places=1)  '-0.0'

A hand-edited result file (the workflow #75 invites) with
`"cost_per_million_tokens": -0.0` put a negative price into the table #29's
guard exists to keep non-negative.

`test_a_negative_zero_is_a_valid_result` is the anti-vacuity arm: if the
constructor ever starts refusing `-0.0`, the rendering arms below would be
testing an unreachable input and should be revisited rather than trusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emb_shootout.cli import main
from emb_shootout.sweep import (
    SweepResult,
    _format_cost,
    _format_latency,
    _format_quality,
    aggregate_json,
    aggregate_markdown,
)

_ROOT = Path(__file__).resolve().parents[1]


def _result(name: str, zero: float) -> SweepResult:
    return SweepResult(
        embedder_name=name,
        embedder_dim=8,
        cost_per_million_tokens=zero,
        n_corpus=10,
        n_queries=4,
        recall_at_k={1: zero, 5: zero, 10: zero},
        ndcg_at_10=zero,
        embed_latency_ms={"corpus_total": zero, "query_p50": zero, "query_p95": zero},
    )


def test_a_negative_zero_is_a_valid_result() -> None:
    r = _result("x", -0.0)
    assert str(r.cost_per_million_tokens) == "-0.0"


@pytest.mark.parametrize(
    ("render", "expected"),
    [
        (_format_cost, "$0.000"),
        (_format_quality, "0.000"),
        (lambda v: _format_latency(v, places=0), "0"),
        (lambda v: _format_latency(v, places=1), "0.0"),
    ],
    ids=["cost", "quality", "latency-places-0", "latency-places-1"],
)
def test_each_renderer_publishes_a_negative_zero_as_a_zero(render, expected: str) -> None:
    assert render(-0.0) == expected
    assert render(-0.0) == render(0.0)


def test_a_negative_zero_row_is_the_zero_row() -> None:
    table = aggregate_markdown([_result("a", 0.0), _result("b", -0.0)])
    rows = [line for line in table.splitlines() if line.startswith(("| a |", "| b |"))]
    assert len(rows) == 2
    assert rows[0].replace("| a |", "") == rows[1].replace("| b |", "")
    assert "-" not in rows[1].replace("| b |", "")


def test_the_json_aggregate_keeps_the_stored_value() -> None:
    """Scope: only the rendered cell changed. The JSON carries the stored float."""
    row = aggregate_json([_result("b", -0.0)])["results"][0]
    assert str(row["cost_per_million_tokens"]) == "-0.0"


def test_sweep_aggregate_on_a_hand_edited_negative_zero_cost(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    baseline = json.loads((_ROOT / "results" / "hash.json").read_text(encoding="utf-8"))
    (results / "hash.json").write_text(json.dumps(baseline), encoding="utf-8")
    edited = dict(baseline, embedder_name="self-hosted", cost_per_million_tokens=-0.0)
    (results / "self-hosted.json").write_text(json.dumps(edited), encoding="utf-8")
    out = tmp_path / "t.md"
    assert main(["sweep", "aggregate", "--results-dir", str(results), "--out", str(out)]) == 0
    table = out.read_text(encoding="utf-8")
    assert "$-0" not in table
    cost_cells = [line.rsplit("|", 2)[1].strip() for line in table.splitlines()[2:]]
    assert cost_cells == ["$0.000", "$0.000"]
