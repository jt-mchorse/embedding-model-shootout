"""Same results, same chart bytes: the SVG carries no clock and no random ids (#266).

`render_pareto` called `fig.savefig(buf, dpi=150, format=fmt)` with no metadata
and no `svg.hashsalt`. Measured on `main` (5fb9d89, matplotlib 3.11.2): two runs
of the README's regen command over unchanged `results/` gave identical PNGs and
SVGs differing in 37 lines -- `<dc:date>` and every `clip-path` id, which
matplotlib salts with `uuid4()` unless `svg.hashsalt` is set. So every regen
rewrote `docs/pareto.svg`, against the README's "every number reproducible
from a fresh clone".

The fake-pyplot arms run in CI, which installs only `.[dev]` (#111, D-008); the
real-matplotlib arms check the bytes where the `plot` extra is installed.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterator
from pathlib import Path

import pytest

from emb_shootout.plot import render_pareto
from emb_shootout.sweep import SweepResult
from tests.test_plot_output_format import _Ax


def _result(name: str, cost: float, recall: float) -> SweepResult:
    return SweepResult(
        embedder_name=name,
        embedder_dim=4,
        cost_per_million_tokens=cost,
        n_corpus=10,
        n_queries=5,
        recall_at_k={1: recall / 2, 5: recall, 10: recall},
        ndcg_at_10=0.5,
        embed_latency_ms={"corpus_total": 1.0, "query_p50": 1.0, "query_p95": 1.0},
    )


# Two frontier points and a dominated one: a polyline, three markers, and the
# axes clip path -- every element kind that carries a generated id.
RESULTS = [_result("cheap", 0.0, 0.6), _result("mid", 0.02, 0.9), _result("worse", 0.05, 0.7)]


# ----------------------------------------------------------------------
# What savefig was handed (runs in CI, no `plot` extra needed)
# ----------------------------------------------------------------------


class _Fig:
    def __init__(self, rc: dict[str, object], calls: list[tuple[str, dict, dict]]) -> None:
        self._rc, self._calls = rc, calls

    def tight_layout(self, *a: object, **k: object) -> None: ...

    def savefig(self, fname: io.BytesIO, *a: object, **k: object) -> None:
        # The rc params in force *while rendering* are what decide the ids.
        self._calls.append((str(k["format"]), dict(k), dict(self._rc)))
        fname.write(b"x")


class _Plt:
    def __init__(self, calls: list[tuple[str, dict, dict]]) -> None:
        self._rc: dict[str, object] = {}
        self._calls = calls

    def subplots(self, *a: object, **k: object):
        return _Fig(self._rc, self._calls), _Ax()

    def close(self, *a: object, **k: object) -> None: ...

    @contextlib.contextmanager
    def rc_context(self, rc: dict[str, object] | None = None) -> Iterator[None]:
        saved = dict(self._rc)
        self._rc.update(rc or {})
        try:
            yield
        finally:
            self._rc.clear()
            self._rc.update(saved)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict, dict]]:
    recorded: list[tuple[str, dict, dict]] = []
    monkeypatch.setattr("emb_shootout.plot._import_matplotlib", lambda: _Plt(recorded))
    return recorded


def test_svg_is_rendered_with_a_fixed_hashsalt(
    tmp_path: Path, calls: list[tuple[str, dict, dict]]
) -> None:
    render_pareto(RESULTS, out_png=tmp_path / "p.png", out_svg=tmp_path / "p.svg")
    salts = {fmt: rc.get("svg.hashsalt") for fmt, _k, rc in calls}
    assert set(salts) == {"png", "svg"}
    assert isinstance(salts["svg"], str), salts
    assert salts["svg"]


def test_svg_drops_the_wall_clock_date(tmp_path: Path, calls: list[tuple[str, dict, dict]]) -> None:
    render_pareto(RESULTS, out_svg=tmp_path / "p.svg")
    [(fmt, kwargs, _rc)] = calls
    assert fmt == "svg"
    assert kwargs.get("metadata") == {"Date": None}, kwargs


def test_png_keywords_are_unchanged(tmp_path: Path, calls: list[tuple[str, dict, dict]]) -> None:
    """GREEN against the unfixed tree: the PNG was already deterministic, and
    `Date` is not a PNG metadata key, so it is not passed one."""
    render_pareto(RESULTS, out_png=tmp_path / "p.png")
    [(fmt, kwargs, _rc)] = calls
    assert fmt == "png"
    assert "metadata" not in kwargs, kwargs


# ----------------------------------------------------------------------
# The bytes (needs the `plot` extra)
# ----------------------------------------------------------------------


def _render_twice(tmp_path: Path) -> tuple[bytes, bytes, bytes, bytes]:
    out = []
    for i in (1, 2):
        png, svg = tmp_path / f"p{i}.png", tmp_path / f"p{i}.svg"
        render_pareto(RESULTS, out_png=png, out_svg=svg)
        out.append((png.read_bytes(), svg.read_bytes()))
    (png1, svg1), (png2, svg2) = out
    return png1, png2, svg1, svg2


def test_two_renders_give_identical_svg_bytes(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    _png1, _png2, svg1, svg2 = _render_twice(tmp_path)
    assert b"clip-path" in svg1  # the ids this is about are actually drawn
    assert svg1 == svg2


def test_svg_carries_no_date(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    _png1, _png2, svg1, _svg2 = _render_twice(tmp_path)
    assert b"<dc:date>" not in svg1


def test_two_renders_give_identical_png_bytes(tmp_path: Path) -> None:
    """GREEN against the unfixed tree: a control that the PNG stays as it was."""
    pytest.importorskip("matplotlib")
    png1, png2, _svg1, _svg2 = _render_twice(tmp_path)
    assert png1 == png2
