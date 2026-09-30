"""Rows in one comparison table share a query set (#155, D-014).

`docs/benchmarks.md` said "all providers run against the same queries by
construction, so cross-provider rows in this table are apples-to-apples" --
directly under a reproduce command running `--queries 200` with no seed, beside
a committed baseline measured at `--queries 50 --seed 42`. Nothing constructed
the claim: `SweepResult` records no seed, and the aggregators rendered whatever
rows they were handed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from emb_shootout.cli import main
from emb_shootout.sweep import SweepResult, aggregate_json, aggregate_markdown, require_comparable

_ROOT = Path(__file__).resolve().parents[1]
_BASELINE = json.loads((_ROOT / "results" / "hash.json").read_text(encoding="utf-8"))


def _result(name: str, *, n_queries: int = 50, n_corpus: int = 12010) -> SweepResult:
    return SweepResult(
        embedder_name=name,
        embedder_dim=8,
        cost_per_million_tokens=0.0,
        n_corpus=n_corpus,
        n_queries=n_queries,
        recall_at_k={1: 0.5, 5: 0.6, 10: 0.7},
        ndcg_at_10=0.5,
        embed_latency_ms={"corpus_total": 1.0, "query_p50": 0.1, "query_p95": 0.2},
        notes=[],
    )


# ----------------------------------------------------------------------
# The aggregators
# ----------------------------------------------------------------------


@pytest.mark.parametrize("aggregate", [aggregate_markdown, aggregate_json])
@pytest.mark.parametrize(
    "other",
    [{"n_queries": 200}, {"n_corpus": 11999}, {"n_queries": 200, "n_corpus": 11999}],
    ids=["queries", "corpus", "both"],
)
def test_rows_from_different_query_sets_are_refused(aggregate: Any, other: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="different query sets") as excinfo:
        aggregate([_result("hash"), _result("openai", **other)])
    # Named, so the operator knows which file to re-run.
    assert "hash" in str(excinfo.value)
    assert "openai" in str(excinfo.value)


@pytest.mark.parametrize("aggregate", [aggregate_markdown, aggregate_json])
def test_rows_from_one_query_set_still_aggregate(aggregate: Any) -> None:
    aggregate([_result("hash"), _result("openai"), _result("voyage")])


def test_a_single_row_is_trivially_comparable() -> None:
    require_comparable([_result("hash", n_queries=7)])


def test_the_cli_exits_two_and_leaves_the_table_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    for r in (_result("hash"), _result("openai", n_queries=200)):
        (results / f"{r.embedder_name}.json").write_text(json.dumps(r.to_dict()), encoding="utf-8")
    out = tmp_path / "benchmarks.md"
    out.write_text("untouched\n", encoding="utf-8")
    rc = main(["sweep", "aggregate", "--results-dir", str(results), "--out", str(out)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "different query sets" in err
    assert out.read_text(encoding="utf-8") == "untouched\n"


def test_the_committed_results_still_aggregate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "b.md"
    assert (
        main(["sweep", "aggregate", "--results-dir", str(_ROOT / "results"), "--out", str(out)])
        == 0
    )
    _ = capsys.readouterr()


# ----------------------------------------------------------------------
# The documented commands
# ----------------------------------------------------------------------


def _documented_provider_commands() -> list[tuple[str, str]]:
    """`(file, command)` for every documented non-demo `sweep run`, with
    backslash continuations and `printf` wrappers flattened."""
    found = []
    for rel in ("README.md", "docs/benchmarks.md", "scripts/capture_demo.sh"):
        text = (_ROOT / rel).read_text(encoding="utf-8")
        text = re.sub(r"printf '([^']*)'", r"\1", text).replace("\\n", "\n")
        text = re.sub(r"\\\s*\n\s*", " ", text)
        for line in text.splitlines():
            if "sweep run" in line and "results/" in line and "$" not in line.split("sweep run")[1]:
                found.append((rel, line.strip()))
    return found


def test_the_documented_commands_are_found() -> None:
    """A pass over no commands is not a pass: README carries two, the
    benchmarks doc one, and the demo script prints one."""
    files = [f for f, _ in _documented_provider_commands()]
    assert files.count("README.md") >= 2
    assert "docs/benchmarks.md" in files
    assert "scripts/capture_demo.sh" in files


@pytest.mark.parametrize(
    ("where", "command"), _documented_provider_commands(), ids=lambda v: str(v)[:40]
)
def test_every_documented_command_uses_the_baselines_query_set(where: str, command: str) -> None:
    """`--queries` is derived from the committed baseline, not retyped, so
    re-measuring the baseline at a different size moves every doc with it."""
    queries = re.search(r"--queries (\d+)", command)
    seed = re.search(r"--seed (\d+)", command)
    assert queries, f"{where}: no --queries in {command!r} (the CLI default is 200)"
    assert int(queries.group(1)) == _BASELINE["n_queries"], f"{where}: {command!r}"
    assert seed, f"{where}: no --seed in {command!r}"
    assert int(seed.group(1)) == 42, f"{where}: {command!r}"
