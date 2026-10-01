"""The Pareto frontier refuses rows measured on different query sets (#167).

`require_comparable` (#155, D-014) was called by `aggregate_markdown` and
`aggregate_json` and not by `pareto_frontier`. Measured on `17de55c`: with
recall@5 from 5 queries beside recall@5 from 200, `sweep aggregate` exited 2
and `sweep plot` exited 0 with the title "hash-embedder-128d-ngram2 dominates
every other model".
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import emb_shootout
from emb_shootout.cli import main
from emb_shootout.pareto import pareto_frontier
from emb_shootout.sweep import SweepResult

PKG = Path(emb_shootout.__file__).resolve().parent


def _result(
    name: str, *, cost: float, recall5: float, n_queries: int = 50, n_corpus: int = 100
) -> SweepResult:
    return SweepResult(
        embedder_name=name,
        embedder_dim=8,
        cost_per_million_tokens=cost,
        n_corpus=n_corpus,
        n_queries=n_queries,
        recall_at_k={1: recall5 / 2, 5: recall5, 10: recall5},
        ndcg_at_10=0.5,
        embed_latency_ms={"corpus_total": 1.0, "query_p50": 0.1, "query_p95": 0.2},
        notes=[],
    )


MIXES = [{"n_queries": 200}, {"n_corpus": 101}]


@pytest.mark.parametrize("other", MIXES, ids=["queries", "corpus"])
def test_the_frontier_refuses_mixed_query_sets(other: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="different query sets"):
        pareto_frontier(
            [
                _result(
                    "cheap", cost=0.0, recall5=1.0, n_queries=5 if "n_queries" in other else 50
                ),
                _result("paid", cost=0.02, recall5=0.985, **other),
            ]
        )


def test_one_query_set_still_has_a_frontier() -> None:
    frontier = pareto_frontier(
        [_result("cheap", cost=0.0, recall5=0.9), _result("paid", cost=0.02, recall5=0.95)]
    )
    assert [r.embedder_name for r in frontier] == ["cheap", "paid"]


@pytest.mark.parametrize("other", MIXES, ids=["queries", "corpus"])
def test_sweep_plot_exits_two_and_writes_no_image(
    other: dict[str, int], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    for r in (
        _result("cheap", cost=0.0, recall5=1.0),
        _result("paid", cost=0.02, recall5=0.985, **other),
    ):
        (results / f"{r.embedder_name}.json").write_text(json.dumps(r.to_dict()), encoding="utf-8")
    png = tmp_path / "p.png"
    rc = main(["sweep", "plot", "--results-dir", str(results), "--out-png", str(png)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "different query sets" in err
    assert not png.exists()


def _calls(fn: ast.FunctionDef) -> set[str]:
    return {
        n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }


def test_every_ranking_or_tabulating_consumer_goes_through_the_check() -> None:
    """The population: D-014 named two consumers and there were three. Every
    module-level function that takes a sequence of results and ranks or
    tabulates them must reach `require_comparable`, directly or through
    `pareto_frontier` / the aggregators."""
    gates = {"require_comparable", "pareto_frontier", "aggregate_markdown", "aggregate_json"}
    consumers = {
        "sweep.py": {"aggregate_markdown", "aggregate_json"},
        "pareto.py": {"pareto_frontier"},
        "plot.py": {"render_pareto"},
    }
    for module, names in consumers.items():
        tree = ast.parse((PKG / module).read_text(encoding="utf-8"))
        fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in names:
            assert _calls(fns[name]) & (gates - {name}), f"{module}:{name} skips require_comparable"


def test_the_refusal_does_not_depend_on_the_plotting_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An incomparable set is bad input (exit 2) whether or not matplotlib is
    installed; checked before the import, so it is never reported as
    "matplotlib missing" (exit 3) instead."""
    from emb_shootout import plot

    def no_matplotlib():
        raise RuntimeError("matplotlib is not installed")

    monkeypatch.setattr(plot, "_import_matplotlib", no_matplotlib)
    with pytest.raises(ValueError, match="different query sets"):
        plot.render_pareto(
            [
                _result("cheap", cost=0.0, recall5=1.0, n_queries=5),
                _result("paid", cost=0.02, recall5=0.985, n_queries=200),
            ],
            out_png=tmp_path / "p.png",
        )
