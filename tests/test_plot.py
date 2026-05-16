"""Tests for the Pareto plot renderer (`emb_shootout.plot`).

Skipped automatically if matplotlib (the `[plot]` extra) isn't installed,
so the standard CI matrix without extras still passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emb_shootout.cli import main
from emb_shootout.sweep import SweepResult

pytest.importorskip("matplotlib")


from emb_shootout.plot import render_pareto  # noqa: E402  (after importorskip)


def _make(name: str, cost: float, recall_at_5: float) -> SweepResult:
    return SweepResult(
        embedder_name=name,
        embedder_dim=128,
        cost_per_million_tokens=cost,
        n_corpus=100,
        n_queries=10,
        recall_at_k={1: 0.0, 5: recall_at_5, 10: recall_at_5},
        ndcg_at_10=recall_at_5,
        embed_latency_ms={"corpus_total": 0.0, "query_p50": 0.0, "query_p95": 0.0},
    )


def test_render_writes_png_and_svg(tmp_path: Path):
    results = [
        _make("hash", cost=0.0, recall_at_5=0.52),
        _make("openai-3-small", cost=0.02, recall_at_5=0.80),
        _make("openai-3-large", cost=0.13, recall_at_5=0.88),
    ]
    png = tmp_path / "out.png"
    svg = tmp_path / "out.svg"
    frontier, png_path, svg_path = render_pareto(results, out_png=png, out_svg=svg)
    assert png.exists()
    assert png.stat().st_size > 0
    assert svg.exists()
    assert svg.stat().st_size > 0
    assert png_path == png
    assert svg_path == svg
    # All three points are non-dominated → all on frontier (cost rises, recall rises).
    assert {r.embedder_name for r in frontier} == {"hash", "openai-3-small", "openai-3-large"}


def test_render_creates_parent_dirs(tmp_path: Path):
    deep = tmp_path / "a" / "b" / "c" / "out.png"
    results = [_make("only", cost=0.0, recall_at_5=0.5)]
    render_pareto(results, out_png=deep)
    assert deep.exists()


def test_render_only_png(tmp_path: Path):
    results = [_make("only", cost=0.0, recall_at_5=0.5)]
    _frontier, png_path, svg_path = render_pareto(results, out_png=tmp_path / "only.png")
    assert png_path is not None
    assert png_path.exists()
    assert svg_path is None


def test_render_only_svg(tmp_path: Path):
    results = [_make("only", cost=0.0, recall_at_5=0.5)]
    _frontier, png_path, svg_path = render_pareto(results, out_svg=tmp_path / "only.svg")
    assert svg_path is not None
    assert svg_path.exists()
    assert png_path is None


def test_render_rejects_no_outputs():
    results = [_make("only", cost=0.0, recall_at_5=0.5)]
    with pytest.raises(ValueError, match="must provide at least one"):
        render_pareto(results)


def test_render_rejects_empty_results(tmp_path: Path):
    with pytest.raises(ValueError, match="non-empty"):
        render_pareto([], out_png=tmp_path / "x.png")


def test_render_with_dominated_point_keeps_only_frontier_red(tmp_path: Path):
    # Dominated point excluded from `frontier` return value.
    cheap_good = _make("cheap_good", cost=0.1, recall_at_5=0.9)
    expensive_bad = _make("expensive_bad", cost=10.0, recall_at_5=0.3)
    frontier, _png, _svg = render_pareto([cheap_good, expensive_bad], out_png=tmp_path / "dom.png")
    assert [r.embedder_name for r in frontier] == ["cheap_good"]


def test_cli_plot_subcommand_writes_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    payload = _make("hash-embedder-128d-ngram2", cost=0.0, recall_at_5=0.52).to_dict()
    (results_dir / "hash.json").write_text(json.dumps(payload), encoding="utf-8")

    out_png = tmp_path / "out.png"
    out_svg = tmp_path / "out.svg"
    rc = main(
        [
            "sweep",
            "plot",
            "--results-dir",
            str(results_dir),
            "--out-png",
            str(out_png),
            "--out-svg",
            str(out_svg),
        ]
    )
    assert rc == 0
    assert out_png.exists()
    assert out_svg.exists()
    summary = json.loads(capsys.readouterr().out)
    assert summary["results"] == 1
    assert summary["frontier_size"] == 1
    assert summary["frontier"] == ["hash-embedder-128d-ngram2"]
    assert summary["png"] == str(out_png)
    assert summary["svg"] == str(out_svg)


def test_cli_plot_requires_at_least_one_output(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    payload = _make("hash-embedder-128d-ngram2", cost=0.0, recall_at_5=0.52).to_dict()
    (results_dir / "hash.json").write_text(json.dumps(payload), encoding="utf-8")
    rc = main(["sweep", "plot", "--results-dir", str(results_dir)])
    assert rc == 2


def test_cli_plot_missing_results_dir(tmp_path: Path):
    rc = main(
        [
            "sweep",
            "plot",
            "--results-dir",
            str(tmp_path / "nope"),
            "--out-png",
            str(tmp_path / "x.png"),
        ]
    )
    assert rc == 2


def test_cli_plot_empty_results_dir(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    rc = main(
        [
            "sweep",
            "plot",
            "--results-dir",
            str(results_dir),
            "--out-png",
            str(tmp_path / "x.png"),
        ]
    )
    assert rc == 2
