"""A near-perfect recall/nDCG is not published as a perfect one (#178).

#149 widened a quality cell whose `.3f` form would fabricate a ZERO, arguing
that end only: "`0.000` is the *worst* value, so truncation understates rather
than flatters." `1.0` is the best value, and there `.3f` rounds UP into it.
Measured on main (cc36d9b): `_format_quality(0.9995)` -> `'1.000'`, so a sweep
that missed one query in 2,000 printed a row byte-identical to a perfect one,
while `aggregate_json` carried 0.9995. The `sweep run` summary line printed a
bare `.3f` and so had neither end's rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emb_shootout import sweep as sweep_mod
from emb_shootout.cli import main
from emb_shootout.sweep import SweepResult, _format_quality, aggregate_markdown

_REPO_ROOT = Path(__file__).resolve().parents[1]
_COMMITTED = _REPO_ROOT / "results" / "hash.json"


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        (1.0, "1.000"),  # a genuine perfect score keeps the narrow form
        (0.9995, "0.9995"),
        (0.99951, "0.9995"),
        (0.9998, "0.9998"),
        (0.99999, "0.99999"),
        (0.99949, "0.999"),  # never rounded into 1.0: unchanged
        (0.5, "0.500"),
        (0.0, "0.000"),
        (0.00025, "0.00025"),  # #149's end still holds
    ],
)
def test_format_quality(value: float, rendered: str) -> None:
    assert _format_quality(value) == rendered


def _result(name: str, recall_at_10: float, n_queries: int = 2000) -> SweepResult:
    payload = json.loads(_COMMITTED.read_text())
    payload["embedder_name"] = name
    payload["n_queries"] = n_queries
    payload["recall_at_k"]["10"] = recall_at_10
    return SweepResult.from_dict(payload)


def _row_cells(md: str, name: str) -> list[str]:
    (line,) = [line for line in md.splitlines() if line.startswith(f"| {name} ")]
    return [c.strip() for c in line.strip().strip("|").split("|")][1:]


def test_a_missed_query_in_2000_and_a_perfect_sweep_render_different_rows() -> None:
    md = aggregate_markdown([_result("model-perfect", 1.0), _result("model-missed-one", 0.9995)])
    assert _row_cells(md, "model-perfect") != _row_cells(md, "model-missed-one")
    assert "1.000" in _row_cells(md, "model-perfect")
    assert "0.9995" in _row_cells(md, "model-missed-one")


@pytest.mark.parametrize(
    ("recall_at_5", "ndcg", "shown"),
    [
        (0.9995, 0.9998, "recall@5=0.9995 NDCG@10=0.9998"),
        (0.00025, 7.23e-05, "recall@5=0.00025 NDCG@10=0.000072"),
        (1.0, 0.0, "recall@5=1.000 NDCG@10=0.000"),
    ],
)
def test_the_sweep_run_summary_line_uses_the_table_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    recall_at_5: float,
    ndcg: float,
    shown: str,
) -> None:
    payload = json.loads(_COMMITTED.read_text())
    payload["recall_at_k"]["5"] = recall_at_5
    payload["ndcg_at_10"] = ndcg
    stub = SweepResult.from_dict(payload)
    monkeypatch.setattr(sweep_mod, "run_sweep", lambda *a, **k: stub)
    out = tmp_path / "r.json"
    rc = main(
        [
            "sweep",
            "run",
            "--provider",
            "hash",
            "--corpus",
            str(_REPO_ROOT / "data" / "corpus.jsonl"),
            "--queries",
            "5",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    assert f"{stub.embedder_name}: {shown} " in capsys.readouterr().out
