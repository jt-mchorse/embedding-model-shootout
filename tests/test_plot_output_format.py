"""`--out-png` writes PNG and `--out-svg` writes SVG, at exactly the path given (#220).

`render_pareto` called `fig.savefig(p, dpi=150)` with no `format=`, so
matplotlib chose the format from the extension. Measured on `main` with the
`plot` extra: `--out-svg pareto` wrote a PNG to `pareto.png` while the summary
reported `"svg": ".../pareto"`, and `--out-png chart.svg` wrote an SVG.

The fake-figure arms run in CI, which has no `plot` extra (#111, D-008); the
real-matplotlib arms check the bytes where it is installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emb_shootout.cli import main
from emb_shootout.plot import render_pareto
from emb_shootout.sweep import SweepResult


def _result() -> SweepResult:
    return SweepResult(
        embedder_name="m",
        embedder_dim=4,
        cost_per_million_tokens=0.0,
        n_corpus=10,
        n_queries=5,
        recall_at_k={1: 0.4, 5: 0.6, 10: 0.8},
        ndcg_at_10=0.5,
        embed_latency_ms={"corpus_total": 1.0, "query_p50": 1.0, "query_p95": 1.0},
    )


class _Fig:
    def __init__(self, saved: list[tuple[Path, object]]) -> None:
        self._saved = saved

    def tight_layout(self, *a: object, **k: object) -> None: ...

    def savefig(self, path: Path, *a: object, **k: object) -> None:
        self._saved.append((Path(path), k.get("format")))


class _Ax:
    def __getattr__(self, _name: str):
        return lambda *a, **k: None


class _Plt:
    def __init__(self, saved: list[tuple[Path, object]]) -> None:
        self._saved = saved

    def subplots(self, *a: object, **k: object):
        return _Fig(self._saved), _Ax()

    def close(self, *a: object, **k: object) -> None: ...


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, object]]:
    calls: list[tuple[Path, object]] = []
    monkeypatch.setattr("emb_shootout.plot._import_matplotlib", lambda: _Plt(calls))
    return calls


def test_each_flag_pins_its_own_format_at_the_exact_path(
    tmp_path: Path, saved: list[tuple[Path, object]]
) -> None:
    png, svg = tmp_path / "chart", tmp_path / "pareto"
    _, png_out, svg_out = render_pareto([_result()], out_png=png, out_svg=svg)
    assert saved == [(png, "png"), (svg, "svg")]
    assert (png_out, svg_out) == (png, svg)


def test_matching_suffix_is_accepted_case_insensitively(
    tmp_path: Path, saved: list[tuple[Path, object]]
) -> None:
    render_pareto([_result()], out_png=tmp_path / "a.PNG", out_svg=tmp_path / "b.svg")
    assert saved == [(tmp_path / "a.PNG", "png"), (tmp_path / "b.svg", "svg")]


@pytest.mark.parametrize(
    ("flag", "name"),
    [("out_png", "chart.svg"), ("out_svg", "pareto.png"), ("out_png", "chart.pdf")],
    ids=["png-flag-svg-path", "svg-flag-png-path", "png-flag-pdf-path"],
)
def test_suffix_naming_another_format_is_refused_before_rendering(
    tmp_path: Path, saved: list[tuple[Path, object]], flag: str, name: str
) -> None:
    with pytest.raises(ValueError, match=r"output path .* ends in"):
        render_pareto([_result()], **{flag: tmp_path / name})
    assert saved == []
    assert not (tmp_path / name).exists()


def _write_results(tmp_path: Path) -> Path:
    d = tmp_path / "results"
    d.mkdir()
    (d / "m.json").write_text(json.dumps(_result().to_dict()), encoding="utf-8")
    return d


def test_cli_refuses_a_crossed_suffix_at_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], saved: list[tuple[Path, object]]
) -> None:
    d = _write_results(tmp_path)
    rc = main(["sweep", "plot", "--results-dir", str(d), "--out-png", str(tmp_path / "x.svg")])
    assert rc == 2
    assert "ends in '.svg'" in capsys.readouterr().err
    assert saved == []


def test_cli_summary_names_the_paths_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], saved: list[tuple[Path, object]]
) -> None:
    d = _write_results(tmp_path)
    svg = tmp_path / "pareto"
    rc = main(["sweep", "plot", "--results-dir", str(d), "--out-svg", str(svg)])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["svg"] == str(svg)
    assert saved == [(svg, "svg")]


def test_real_matplotlib_writes_the_flag_format_at_the_exact_path(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    png, svg = tmp_path / "chart", tmp_path / "pareto"
    render_pareto([_result()], out_png=png, out_svg=svg)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["chart", "pareto"]
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert b"<svg" in svg.read_bytes()[:1000]
