"""An unmeasured recall@5 is refused, not scored as zero (#123).

`pareto._recall_at_5` was `recall_at_k.get(5, 0.0)`. `0.0` is the *worst
possible* value on the frontier's quality axis, so a result swept without
`k=5` was not skipped or flagged — it was ranked against every other model as
though it had scored zero.

`run_sweep(..., k_values=...)` is an exported parameter and
`SweepResult.from_dict` reconstructs from JSON with no requirement that `5` be
present; `sweep.py`'s own comments note an empty `recall_at_k` is reachable
that way.

The two measured consequences, recorded as the two `TestMeasured*` classes:

- a model with real recall@1/3/10 of 0.95 / **0.97** / 0.99 was *dominated by*
  and dropped from the frontier in favour of one scoring **0.12** at k=5;
- when it was the cheapest point it survived on the cost axis alone, and was
  plotted at `0.0` on the quality axis of the headline figure.

Raising is what honouring D-008 looks like. That decision fixes the axis as
`recall_at_5`; a result with no such measurement cannot be placed on it, and
inventing a coordinate is the one thing a benchmark must not do.
"""

from __future__ import annotations

import pytest

from emb_shootout.pareto import FRONTIER_K, _dominates, pareto_frontier
from emb_shootout.sweep import ABSENT_RECALL_CELL, SweepResult, aggregate_markdown


def _result(name: str, cost: float, recall: dict[int, float]) -> SweepResult:
    return SweepResult(
        embedder_name=name,
        embedder_dim=8,
        cost_per_million_tokens=cost,
        n_corpus=10,
        n_queries=4,
        recall_at_k=dict(recall),
        ndcg_at_10=0.5,
        embed_latency_ms={"corpus_total": 1.0, "query_p50": 1.0, "query_p95": 2.0},
        notes=[],
    )


# The model at the centre of both measured cases: strong, but swept at
# k_values=(1, 3, 10) so it has no recall@5.
STRONG_NO_5 = {1: 0.95, 3: 0.97, 10: 0.99}


class TestMeasuredCase2DroppedFromTheFrontier:
    def test_a_097_model_is_no_longer_beaten_by_a_012_one(self) -> None:
        strong = _result("strong-no-5", 0.50, STRONG_NO_5)
        weak = _result("cheap-weak", 0.10, {1: 0.10, 5: 0.12, 10: 0.15})
        mid = _result("mid", 0.30, {1: 0.40, 5: 0.55, 10: 0.60})

        with pytest.raises(ValueError, match="strong-no-5") as exc:
            pareto_frontier([strong, weak, mid])
        message = str(exc.value)
        assert "strong-no-5" in message, "the error must name the offending embedder"
        assert "[1, 3, 10]" in message, "and say what it WAS measured at"

    def test_the_pre_fix_domination_really_did_happen(self) -> None:
        """Pins the mechanism, so the fix's rationale stays checkable.

        `_dominates` is unchanged — it is correct Pareto domination. The defect
        was the coordinate handed to it. With an explicit 0.0 (what `.get`
        used to supply) the 0.12 model really does dominate the 0.97 one.
        """
        strong_scored_zero = _result("strong-no-5", 0.50, {**STRONG_NO_5, 5: 0.0})
        weak = _result("cheap-weak", 0.10, {1: 0.10, 5: 0.12, 10: 0.15})
        assert _dominates(weak, strong_scored_zero) is True


class TestMeasuredCase1PlottedAtZero:
    def test_the_cheapest_no_5_model_is_no_longer_plotted_at_zero(self) -> None:
        strong = _result("strong-no-5", 0.02, STRONG_NO_5)
        mid = _result("mid", 0.30, {1: 0.40, 5: 0.55, 10: 0.60})
        with pytest.raises(ValueError, match="strong-no-5"):
            pareto_frontier([strong, mid])

    def test_an_empty_recall_map_is_refused_too(self) -> None:
        # sweep.py's own comments call this reachable via `from_dict`.
        empty = _result("empty", 0.40, {})
        with pytest.raises(ValueError, match="nothing"):
            pareto_frontier([empty])


class TestExistingBehaviourIsUnchanged:
    """Everything that already carried k=5 must behave exactly as before."""

    def test_a_normal_frontier_is_unchanged(self) -> None:
        a = _result("cheap-weak", 0.10, {5: 0.12})
        b = _result("mid", 0.30, {5: 0.55})
        c = _result("dear-worse", 0.90, {5: 0.50})
        frontier = [r.embedder_name for r in pareto_frontier([a, b, c])]
        assert frontier == ["cheap-weak", "mid"]

    def test_ties_on_both_axes_both_stay(self) -> None:
        # The docstring promises this; the precondition loop must not disturb it.
        a = _result("a", 0.20, {5: 0.60})
        b = _result("b", 0.20, {5: 0.60})
        assert len(pareto_frontier([a, b])) == 2

    def test_empty_input_is_still_an_empty_frontier(self) -> None:
        assert pareto_frontier([]) == []

    def test_a_single_result_is_its_own_frontier(self) -> None:
        only = _result("only", 0.20, {5: 0.60})
        assert [r.embedder_name for r in pareto_frontier([only])] == ["only"]

    def test_a_genuine_zero_recall_is_still_a_valid_point(self) -> None:
        """A real 0.0 measurement is data, not a missing axis."""
        zero = _result("genuinely-zero", 0.01, {5: 0.0})
        other = _result("other", 0.50, {5: 0.90})
        assert len(pareto_frontier([zero, other])) == 2

    def test_the_frontier_k_constant_matches_D_008(self) -> None:
        assert FRONTIER_K == 5


class TestAggregateMarkdownDoesNotFabricate:
    def test_an_unmeasured_cell_is_not_a_number(self) -> None:
        strong = _result("strong-no-5", 0.02, STRONG_NO_5)
        mid = _result("mid", 0.30, {1: 0.40, 5: 0.55, 10: 0.60})
        rows = [
            line
            for line in aggregate_markdown([strong, mid]).splitlines()
            if line.startswith("| ") and "---" not in line and not line.startswith("| embedder")
        ]
        strong_row = next(r for r in rows if "strong-no-5" in r)
        mid_row = next(r for r in rows if "| mid " in r)

        assert "0.970" in strong_row, "its real recall@3 must still be reported"
        assert "0.000" not in strong_row, (
            "recall@5 was never measured for this model; pre-fix it read 0.000, "
            "indistinguishable from a measured zero"
        )
        assert ABSENT_RECALL_CELL in strong_row
        # `mid` was never measured at 3 — the same defect in the other direction.
        assert ABSENT_RECALL_CELL in mid_row
        assert "0.550" in mid_row

    def test_a_genuine_zero_still_renders_as_zero(self) -> None:
        """Otherwise the fix has traded a false number for a false absence."""
        zero = _result("genuinely-zero", 0.01, {1: 0.0, 5: 0.0, 10: 0.0})
        row = next(
            line
            for line in aggregate_markdown([zero]).splitlines()
            if line.startswith("| genuinely-zero")
        )
        assert "0.000" in row
        assert ABSENT_RECALL_CELL not in row

    def test_a_uniform_sweep_renders_with_no_placeholder(self) -> None:
        # What the CLI produces: k_values=(1, 5, 10) for every provider.
        a = _result("a", 0.10, {1: 0.1, 5: 0.2, 10: 0.3})
        b = _result("b", 0.20, {1: 0.4, 5: 0.5, 10: 0.6})
        table = aggregate_markdown([a, b])
        assert ABSENT_RECALL_CELL not in table, (
            "the canonical CLI path must render exactly as before; the committed "
            "aggregate artifact depends on it"
        )
        assert "0.100" in table
        assert "0.600" in table


class TestTheRendererCannotFabricateEither:
    """`plot.py` had four more `.get(5, 0.0)` sites (#123, second pass).

    They were already unreachable once `pareto_frontier` gained its
    precondition — `render_pareto` calls it first — but only by *call order*.
    A reordering, or a future path that skips the frontier, would bring the
    fabricated coordinate back. Routing them through the shared accessor makes
    the guarantee structural instead of positional.

    `#111` records that the plot/CLI render tests do not run in CI, so the
    source-level lock below is the part that always executes.
    """

    def test_no_module_substitutes_a_default_for_the_frontier_axis(self) -> None:
        """Scans the AST, not the text.

        A substring scan also matches the docstrings and comments that *quote*
        the old expression to explain why it was removed — prose about a bug is
        not the bug. Parsing means the lock only ever sees real calls.
        """
        import ast
        from pathlib import Path

        pkg = Path(__file__).resolve().parents[1] / "emb_shootout"
        offenders = []
        for rel in ("plot.py", "cli.py", "pareto.py", "sweep.py"):
            tree = ast.parse((pkg / rel).read_text(encoding="utf-8"), filename=rel)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (isinstance(func, ast.Attribute) and func.attr == "get"):
                    continue
                if len(node.args) != 2:
                    continue
                key, default = node.args
                if not (isinstance(key, ast.Constant) and key.value == FRONTIER_K):
                    continue
                if isinstance(default, ast.Constant) and default.value == 0:
                    offenders.append(f"{rel}:{node.lineno}")
        assert offenders == [], (
            "these calls substitute a default for the Pareto frontier's quality "
            "axis; 0.0 is the worst possible value there, so the default ranks "
            "the model rather than abstaining: " + ", ".join(offenders)
        )

    def test_the_lock_is_looking_at_real_files(self) -> None:
        # Guards the guard: a moved package would make the scan vacuous.
        from pathlib import Path

        pkg = Path(__file__).resolve().parents[1] / "emb_shootout"
        for rel in ("plot.py", "cli.py", "pareto.py", "sweep.py"):
            assert (pkg / rel).is_file(), f"{rel} not found — the lock scans nothing"

    def test_render_pareto_refuses_a_missing_axis(self, tmp_path) -> None:
        pytest.importorskip("matplotlib")
        from emb_shootout.plot import render_pareto

        strong = _result("strong-no-5", 0.02, STRONG_NO_5)
        mid = _result("mid", 0.30, {1: 0.40, 5: 0.55, 10: 0.60})
        with pytest.raises(ValueError, match="strong-no-5"):
            render_pareto([strong, mid], out_png=str(tmp_path / "p.png"))

    def test_render_pareto_still_works_on_a_normal_sweep(self, tmp_path) -> None:
        pytest.importorskip("matplotlib")
        from emb_shootout.plot import render_pareto

        a = _result("a", 0.10, {1: 0.1, 5: 0.2, 10: 0.3})
        b = _result("b", 0.20, {1: 0.4, 5: 0.5, 10: 0.6})
        out = tmp_path / "p.png"
        render_pareto([a, b], out_png=str(out))
        assert out.is_file()
        assert out.stat().st_size > 0
