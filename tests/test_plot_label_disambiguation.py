"""No two points in one Pareto chart carry the same annotation (#151).

#69 established — and `plot.py` still says so in a comment — that two distinct
`SweepResult`s can share an `embedder_name`: D-007 writes one file per run, so
running the same provider twice yields two same-named results. #69 stopped
keying the frontier **colour** on that name. The annotation eleven lines below
it still did, and so did `_default_title`.

    frontier size: 1
      cost=0.1   recall=0.9  colour=#d62728 (FRONTIER)    label='openai-3-small'
      cost=10.0  recall=0.3  colour=#7f7f7f (dominated)   label='openai-3-small'

Post-#69 the colouring is right and the chart says "openai-3-small is on the
frontier" *and* "openai-3-small is dominated", with nothing to attribute either
point to a run.

**The fix is sparse, and that is the property most of these arms protect.** A
name that appears once is untouched; only a collision is decorated. Decorating
every point would churn every chart this repo has produced, to disambiguate two
of them. That is the deliberate difference from `llm-cost-optimizer` D-021,
whose set-wide widening is uniform because a column of numbers at mixed
precision reads as mixed quantities.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from emb_shootout.plot import _default_title, disambiguated_labels, pareto_frontier, render_pareto
from emb_shootout.sweep import SweepResult


def _result(name: str, cost: float = 1.0, recall: float = 0.5) -> SweepResult:
    return SweepResult(
        embedder_name=name,
        embedder_dim=1536,
        cost_per_million_tokens=cost,
        n_corpus=100,
        n_queries=10,
        recall_at_k={5: recall},
        ndcg_at_10=0.5,
        embed_latency_ms={"mean": 1.0},
        notes=[],
    )


# ----------------------------------------------------------------------
# The rule
# ----------------------------------------------------------------------


def test_unique_names_are_returned_untouched() -> None:
    """GREEN against the unfixed tree, deliberately.

    This is the arm that rejects "append an index to every label", which would
    satisfy every uniqueness arm below while churning every ordinary chart.
    """
    results = [_result("openai"), _result("voyage"), _result("cohere")]
    assert disambiguated_labels(results) == ["openai", "voyage", "cohere"]


@pytest.mark.parametrize(
    "names",
    [
        pytest.param(["a", "a"], id="two-of-one"),
        pytest.param(["a", "b", "a"], id="separated-by-another"),
        pytest.param(["a", "a", "a"], id="three-of-one"),
        pytest.param(["a", "a", "b", "b"], id="two-colliding-pairs"),
        pytest.param(["a"], id="single"),
        pytest.param([], id="empty"),
    ],
)
def test_no_two_labels_are_equal(names: list[str]) -> None:
    labels = disambiguated_labels([_result(n) for n in names])
    assert len(labels) == len(names)
    assert len(set(labels)) == len(labels), labels


def test_only_the_colliding_names_are_decorated() -> None:
    """The sparseness property, stated directly rather than inferred."""
    results = [_result("a"), _result("b"), _result("a")]
    assert disambiguated_labels(results) == ["a #1", "b", "a #3"]


def test_the_ordinal_is_the_sequence_position_and_gaps_are_meaningful() -> None:
    """`["a", "b", "a"]` is `a #1` / `a #3`, not `a #1` / `a #2`.

    A per-name occurrence counter reads more naturally and points at nothing:
    "the second `a`" is not a file you can open, while "the third result" maps
    to the third entry of `sorted(results_dir.glob("*.json"))`. The gap is the
    feature; this arm exists so a later "tidy-up" does not remove it.
    """
    labels = disambiguated_labels([_result("a"), _result("b"), _result("a")])
    assert "a #2" not in labels
    assert labels[2] == "a #3"


# ----------------------------------------------------------------------
# The title — the second surface that interpolates a name
# ----------------------------------------------------------------------


def test_the_single_winner_title_disambiguates_too() -> None:
    """`_default_title` says "<name> dominates every other model".

    With two runs of one provider and one dominating the other, that sentence
    named a string appearing twice on the chart — once on the frontier and once
    dominated. Answering AC4 by fixing the surface rather than noting it.
    """
    winner = _result("openai", cost=0.1, recall=0.9)
    loser = _result("openai", cost=10.0, recall=0.3)
    assert pareto_frontier([winner, loser]) == [winner]
    title = _default_title([winner, loser], [winner])
    assert title == "Cost vs recall@5 — openai #1 dominates every other model", title


def test_the_ordinary_single_winner_title_is_unchanged() -> None:
    """GREEN against the unfixed tree. The ordinary caption must not move."""
    winner = _result("openai", cost=0.1, recall=0.9)
    other = _result("voyage", cost=10.0, recall=0.3)
    title = _default_title([winner, other], [winner])
    assert title == "Cost vs recall@5 — openai dominates every other model", title


def test_the_title_matches_the_winner_by_identity_not_by_name() -> None:
    """The same trap #69 fixed one function down, in the surface it left behind.

    Picking the winner's label by name would find the *first* result carrying
    that name, which is not necessarily the frontier point.
    """
    loser = _result("openai", cost=10.0, recall=0.3)
    winner = _result("openai", cost=0.1, recall=0.9)
    # Loser first, so a name-keyed lookup would return its label.
    assert pareto_frontier([loser, winner]) == [winner]
    title = _default_title([loser, winner], [winner])
    assert title == "Cost vs recall@5 — openai #2 dominates every other model", title


# ----------------------------------------------------------------------
# The drawn thing, not the computed one
# ----------------------------------------------------------------------


class _RecordingAx:
    def __init__(self) -> None:
        self.annotations: list[str] = []
        self.title = ""

    def scatter(self, *a: object, **k: object) -> None: ...
    def plot(self, *a: object, **k: object) -> None: ...
    def set_xlabel(self, *a: object, **k: object) -> None: ...
    def set_ylabel(self, *a: object, **k: object) -> None: ...
    def grid(self, *a: object, **k: object) -> None: ...
    def set_xlim(self, *a: object, **k: object) -> None: ...
    def set_ylim(self, *a: object, **k: object) -> None: ...
    def margins(self, *a: object, **k: object) -> None: ...

    def set_title(self, title: str, *a: object, **k: object) -> None:
        self.title = title

    def annotate(self, label: str, *a: object, **k: object) -> None:
        self.annotations.append(label)


class _RecordingFig:
    def savefig(self, *a: object, **k: object) -> None: ...
    def tight_layout(self, *a: object, **k: object) -> None: ...


def _draw(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, results: list[SweepResult]):
    """Drive `render_pareto` and return the axes it actually drew on.

    A **fake matplotlib injected into `sys.modules`**, not the real one. This
    render path has no CI coverage — `_default_title`'s own docstring says so
    ("gated on D-008 ... which is how a caption stating the opposite of the data
    survived") — so an `importorskip` arm would skip in exactly the place that
    let the last caption bug through. Technique carried from
    `llm-cost-optimizer#227` tonight, where formatter-level arms stayed green
    against a call-site revert (0 red) until an arm read what the chart was
    handed.
    """
    ax = _RecordingAx()
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.subplots = lambda **k: (_RecordingFig(), ax)  # type: ignore[attr-defined]
    pyplot.close = lambda fig: None  # type: ignore[attr-defined]
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.use = lambda backend: None  # type: ignore[attr-defined]
    matplotlib.pyplot = pyplot  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    render_pareto(results, out_png=tmp_path / "p.png")
    return ax


def test_the_chart_actually_drawn_has_no_duplicate_annotations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reads the labels `ax.annotate` received, not the helper's return."""
    results = [
        _result("openai-3-small", cost=0.1, recall=0.9),
        _result("openai-3-small", cost=10.0, recall=0.3),
    ]
    ax = _draw(monkeypatch, tmp_path, results)
    assert len(ax.annotations) == 2, ax.annotations
    assert len(set(ax.annotations)) == 2, ax.annotations
    assert ax.annotations == ["openai-3-small #1", "openai-3-small #2"], ax.annotations


def test_the_chart_actually_drawn_is_unchanged_without_collisions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """GREEN against the unfixed tree — the ordinary chart must not move."""
    results = [
        _result("openai-3-small", cost=0.1, recall=0.9),
        _result("voyage-3", cost=10.0, recall=0.3),
    ]
    ax = _draw(monkeypatch, tmp_path, results)
    assert ax.annotations == ["openai-3-small", "voyage-3"], ax.annotations


def test_the_drawn_title_also_disambiguates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    results = [
        _result("openai-3-small", cost=0.1, recall=0.9),
        _result("openai-3-small", cost=10.0, recall=0.3),
    ]
    ax = _draw(monkeypatch, tmp_path, results)
    assert ax.title == "Cost vs recall@5 — openai-3-small #1 dominates every other model", ax.title


def test_every_name_interpolation_in_plot_goes_through_the_helper() -> None:
    """The population, discovered rather than listed.

    Two surfaces interpolated `embedder_name` and #69 fixed neither — it fixed
    the *colour*. A third (an axis label, a legend, a caption) is the obvious
    next change to this module, so the rule is over the file: no
    `.embedder_name` read outside `disambiguated_labels` and the fallback inside
    `_default_title`.
    """
    source = (Path(__file__).resolve().parents[1] / "emb_shootout" / "plot.py").read_text(
        encoding="utf-8"
    )
    code = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))

    # Exempt the helper by slicing its function out by position, not by matching
    # the text of the lines inside it. A text-keyed exemption is a wildcard: it
    # keeps exempting whatever happens to look like the string it was written
    # for, and stops exempting the helper the moment the helper is rewritten.
    # (Measured: two neighbour probes that only changed the helper's internals
    # tripped an earlier text-keyed version of this arm for no real reason.)
    helper_start = code.index("def disambiguated_labels(")
    helper_end = code.index("\ndef ", helper_start)
    inside_helper = code[helper_start:helper_end]
    outside_helper = code[:helper_start] + code[helper_end:]

    offenders = [
        line.strip()
        for line in outside_helper.splitlines()
        # The `_default_title` fallback passes the bare name to `next(...)` as
        # its default, which is correct: it is what the label *would* be when no
        # identity match is found, and it is already the disambiguated value in
        # every collision-free chart.
        if ".embedder_name" in line and "frontier[0].embedder_name," not in line
    ]
    assert offenders == [], (
        f"plot.py reads `.embedder_name` outside the label helper at {offenders}; "
        "route it through `disambiguated_labels` — the name is not unique (#69, #151)"
    )
    # Anti-vacuity, two ways: the slice must be non-empty, and the helper must
    # really contain the read this rule would otherwise reject.
    assert inside_helper.strip(), "the helper slice is empty; this arm walks nothing"
    assert ".embedder_name" in inside_helper
    assert "disambiguated_labels(results)" in outside_helper
