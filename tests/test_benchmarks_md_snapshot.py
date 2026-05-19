"""Snapshot test for `docs/benchmarks.md`'s aggregator-produced table.

`docs/benchmarks.md` is the operator-facing artifact when real-provider runs
are committed. Its own opening paragraph says the file is **regenerated** by
``emb-shootout sweep aggregate`` from JSON files under ``results/``.

The numbers in the committed row are indirectly locked by
``test_readme_snapshot.py`` (which asserts ``results/hash.json``'s cells and
the README prose quoting them). What's *not* locked anywhere is the
aggregator's **rendering**: column order, decimal precision, sort key,
header signature, separator alignment. A tweak to
``emb_shootout.sweep.aggregate_markdown`` would silently desync
``docs/benchmarks.md`` from what the CLI now emits.

Same hygiene pattern as ``test_readme_snapshot.py`` (this repo),
``test_rewriter_bench_snapshot.py`` + ``test_eval_bench_snapshot.py``
(rag-production-kit), the savings snapshot in llm-cost-optimizer, and the
regression-demo snapshot in prompt-regression-suite.

When the snapshot fails, regenerate with::

    emb-shootout sweep aggregate --results-dir results --out docs/benchmarks.md

…then ``git diff docs/benchmarks.md`` before committing.
"""

from __future__ import annotations

import json
from pathlib import Path

from emb_shootout.sweep import SweepResult, aggregate_markdown

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
BENCHMARKS_MD = REPO_ROOT / "docs" / "benchmarks.md"

REGEN_HINT = (
    "Regenerate docs/benchmarks.md:\n"
    "  emb-shootout sweep aggregate --results-dir results --out docs/benchmarks.md\n"
    "Inspect with `git diff docs/benchmarks.md` before committing."
)

# Header signature matches the column names the aggregator always emits.
# Used as a presence guard so a missing table fails loudly rather than the
# substring check producing a confusing error.
HEADER_SIGNATURE = "| embedder | dim | n_corpus | n_queries |"


def _load_committed_results() -> list[SweepResult]:
    """Mirror what ``emb-shootout sweep aggregate`` feeds the aggregator."""
    files = sorted(RESULTS_DIR.glob("*.json"))
    assert files, (
        f"No result JSON files under {RESULTS_DIR}. The aggregator snapshot "
        f"needs at least one committed run to verify against."
    )
    return [SweepResult.from_dict(json.loads(p.read_text(encoding="utf-8"))) for p in files]


def test_committed_benchmarks_md_contains_aggregator_output() -> None:
    """The exact aggregator-produced table block must appear in docs/benchmarks.md."""
    results = _load_committed_results()
    expected_md = aggregate_markdown(results)
    actual = BENCHMARKS_MD.read_text(encoding="utf-8")

    # The aggregator output is a header + separator + N data rows followed
    # by a trailing newline. We assert the full block appears verbatim;
    # surrounding prose in docs/benchmarks.md is free to evolve.
    assert expected_md.strip() in actual, (
        "docs/benchmarks.md does not contain the current `aggregate_markdown(results/)` "
        "output verbatim. Either the aggregator changed shape or a cell was hand-edited.\n\n"
        f"Aggregator produced:\n{expected_md}\n"
        f"{REGEN_HINT}"
    )


def test_committed_benchmarks_md_has_header_signature() -> None:
    """Guard against silently dropping the table entirely from docs/benchmarks.md."""
    actual = BENCHMARKS_MD.read_text(encoding="utf-8")
    assert HEADER_SIGNATURE in actual, (
        f"docs/benchmarks.md is missing the aggregator table header "
        f"(`{HEADER_SIGNATURE}`). If the header signature changed intentionally, "
        f"update HEADER_SIGNATURE in this test.\n{REGEN_HINT}"
    )
