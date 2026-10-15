"""The Pareto plot draws an `embedder_name` as the literal name, not as mathtext (#214).

matplotlib parses any text holding a pair of unescaped `$` as mathtext, and
`render_pareto` handed every point label (and the auto-chosen title, which
quotes the single winner's label since #151) to matplotlib with that default.
Measured on `origin/main` with the built `Axes` captured:

    'tier $0.02$ small'   drawn as mathtext: the `$` gone, the span italic
    'cost $\\frac$ model'   ParseSyntaxException (a ValueError) -- `sweep plot`
                          printed it as a bare `error:` at exit 2, naming no
                          file, for a result `sweep aggregate` accepts at exit 0
    'plain'               drawn literally (control)

Both `SweepResult` seams accept such a name, correctly: #191's
`refuse_unrenderable_name` is about names *no* surface can show, and a `$` is
an ordinary character everywhere except this one reader.

The CI arm is the fake-matplotlib one (the `plot` extra is not installed in CI,
D-008), so it reads what `annotate`/`set_title` were *handed*. The real-render
arms read the `Text` objects matplotlib built, and skip without the extra.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from emb_shootout.cli import main
from emb_shootout.plot import render_pareto
from emb_shootout.sweep import SweepResult

#: A well-formed math span: silently re-typeset before #214.
PAIRED = r"tier $0.02$ small"
#: A malformed math span: raised out of `render_pareto` before #214.
MALFORMED = r"cost $\frac$ model"


def _result(name: str, cost: float = 0.1, recall: float = 0.5) -> SweepResult:
    return SweepResult(
        embedder_name=name,
        embedder_dim=8,
        cost_per_million_tokens=cost,
        n_corpus=10,
        n_queries=5,
        recall_at_k={5: recall},
        ndcg_at_10=recall,
        embed_latency_ms={},
        notes=[],
    )


# ----------------------------------------------------------------------
# What matplotlib was handed (runs in CI, no `plot` extra needed)
# ----------------------------------------------------------------------


class _RecordingAx:
    def __init__(self) -> None:
        self.annotations: list[tuple[str, dict[str, object]]] = []
        self.titles: list[tuple[str, dict[str, object]]] = []

    def scatter(self, *a: object, **k: object) -> None: ...
    def plot(self, *a: object, **k: object) -> None: ...
    def set_xlabel(self, *a: object, **k: object) -> None: ...
    def set_ylabel(self, *a: object, **k: object) -> None: ...
    def grid(self, *a: object, **k: object) -> None: ...
    def set_xlim(self, *a: object, **k: object) -> None: ...
    def set_ylim(self, *a: object, **k: object) -> None: ...

    def set_title(self, title: str, *a: object, **k: object) -> None:
        self.titles.append((title, k))

    def annotate(self, label: str, *a: object, **k: object) -> None:
        self.annotations.append((label, k))


class _RecordingFig:
    def savefig(self, *a: object, **k: object) -> None: ...
    def tight_layout(self, *a: object, **k: object) -> None: ...


def _draw_fake(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    results: list[SweepResult],
    title: str | None = None,
) -> _RecordingAx:
    ax = _RecordingAx()
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.subplots = lambda **k: (_RecordingFig(), ax)  # type: ignore[attr-defined]
    pyplot.close = lambda fig: None  # type: ignore[attr-defined]
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.use = lambda backend: None  # type: ignore[attr-defined]
    matplotlib.pyplot = pyplot  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    render_pareto(results, out_png=tmp_path / "p.png", title=title)
    return ax


def test_every_point_label_is_handed_over_as_literal_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    results = [_result(PAIRED, cost=0.1, recall=0.9), _result(MALFORMED, cost=10.0, recall=0.95)]
    ax = _draw_fake(monkeypatch, tmp_path, results)
    assert [label for label, _ in ax.annotations] == [PAIRED, MALFORMED]
    for label, kwargs in ax.annotations:
        assert kwargs.get("parse_math") is False, (label, kwargs)


def test_the_auto_chosen_title_is_handed_over_as_literal_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # One dominates the other, so the title quotes the winner's label (#151).
    results = [_result(PAIRED, cost=0.1, recall=0.9), _result("other", cost=10.0, recall=0.3)]
    ax = _draw_fake(monkeypatch, tmp_path, results)
    [(title, kwargs)] = ax.titles
    assert title == f"Cost vs recall@5 — {PAIRED} dominates every other model", title
    assert kwargs.get("parse_math") is False, kwargs


def test_an_operator_title_keeps_matplotlibs_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """GREEN against the unfixed tree: `--title` is the operator's own markup."""
    ax = _draw_fake(monkeypatch, tmp_path, [_result("a")], title=r"$x^2$ vs cost")
    [(title, kwargs)] = ax.titles
    assert title == r"$x^2$ vs cost"
    assert "parse_math" not in kwargs, kwargs


# ----------------------------------------------------------------------
# What matplotlib built (needs the `plot` extra)
# ----------------------------------------------------------------------


def _draw_real(results: list[SweepResult], tmp_path: Path, title: str | None = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    captured: list[object] = []
    real_subplots = plt.subplots

    def spy(*args: object, **kwargs: object) -> tuple[object, object]:
        fig, ax = real_subplots(*args, **kwargs)  # type: ignore[arg-type]
        captured.append(ax)
        return fig, ax

    plt.subplots = spy  # type: ignore[assignment]
    try:
        render_pareto(results, out_svg=tmp_path / "p.svg", title=title)
    finally:
        plt.subplots = real_subplots  # type: ignore[assignment]
    return captured[-1]


def _is_math(text: object) -> bool:
    return bool(text._preprocess_math(text.get_text())[1])  # type: ignore[attr-defined]


@pytest.mark.parametrize("name", [PAIRED, MALFORMED, "plain"], ids=["paired", "malformed", "plain"])
def test_the_drawn_label_is_the_literal_name(name: str, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    ax = _draw_real([_result(name)], tmp_path)
    [text] = ax.texts  # type: ignore[attr-defined]
    assert text.get_text() == name
    assert not _is_math(text), f"{name!r} was drawn as mathtext"


def test_the_drawn_winner_title_is_literal(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    results = [_result(MALFORMED, cost=0.1, recall=0.9), _result("other", cost=10.0, recall=0.3)]
    ax = _draw_real(results, tmp_path)
    title = ax.title  # type: ignore[attr-defined]
    assert title.get_text() == f"Cost vs recall@5 — {MALFORMED} dominates every other model"
    assert not _is_math(title)


def test_a_drawn_operator_title_can_still_be_math(tmp_path: Path) -> None:
    """GREEN against the unfixed tree: the opt-in path is unchanged."""
    pytest.importorskip("matplotlib")
    ax = _draw_real([_result("a")], tmp_path, title=r"$x^2$ vs cost")
    assert _is_math(ax.title)  # type: ignore[attr-defined]


def test_sweep_plot_cli_renders_a_name_with_a_malformed_math_span(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("matplotlib")
    results = tmp_path / "results"
    results.mkdir()
    (results / "r.json").write_text(json.dumps(_result(MALFORMED).to_dict()), encoding="utf-8")
    out = tmp_path / "pareto.png"
    rc = main(["sweep", "plot", "--results-dir", str(results), "--out-png", str(out)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert out.exists()
    assert json.loads(captured.out)["frontier"] == [MALFORMED]
