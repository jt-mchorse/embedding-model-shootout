"""The documented regen command must not delete the artifact's prose (#145, D-011).

`docs/benchmarks.md` opens by saying it "is **regenerated** by
``emb-shootout sweep aggregate``" and "Don't hand-edit". Run as documented on
``295a88b``::

    $ emb-shootout sweep aggregate --results-dir results --out docs/benchmarks.md
    aggregated 1 results → docs/benchmarks.md
    file is now 3 lines (was 44)

    $ grep -c "no-fabricated-benchmarks" docs/benchmarks.md
    0

    $ pytest
    873 passed

It deleted the `## Current results` framing, the interpretation paragraph, the
whole `## Reproducing` section, the apples-to-apples note, and **"Per the
no-fabricated-benchmarks rule, this README does not carry placeholder numbers for
those providers"** — the sentence that encodes this portfolio's first quality rule.

And the suite stayed green, which is the part that matters.
``tests/test_benchmarks_md_snapshot.py`` locks this artifact by **containment**:
it asserts the aggregator's table is *in* the file. A file truncated *to* the table
still contains the table. A containment lock cannot see a deletion, and a green
suite is exactly why an operator would believe the regeneration had gone fine.

This module adds the two checks that can see it: an **equality** lock over the
spliced result, and a test that runs the documented command and asserts the prose
survives it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from emb_shootout.sweep import (
    TABLE_BEGIN_MARKER,
    TABLE_END_MARKER,
    SweepResult,
    aggregate_markdown,
    splice_markdown_table,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_MD = _REPO_ROOT / "docs" / "benchmarks.md"
RESULTS_DIR = _REPO_ROOT / "results"

#: Prose that must survive any regeneration. The first entry is load-bearing well
#: beyond this file: it is the repo's statement of the no-fabricated-benchmarks
#: rule, which is the one thing the portfolio handoff forbids breaking.
REQUIRED_PROSE = (
    "no-fabricated-benchmarks",
    "## Current results",
    "## Reproducing",
    "apples-to-apples",
)


def _committed_results() -> list[SweepResult]:
    return [
        SweepResult.from_dict(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(RESULTS_DIR.glob("*.json"))
    ]


def test_the_committed_artifact_carries_the_markers() -> None:
    """Anti-vacuous for everything below: without markers the splice is a no-op
    and the preservation tests would pass against a file nothing protects."""
    text = BENCHMARKS_MD.read_text(encoding="utf-8")
    assert text.count(TABLE_BEGIN_MARKER) == 1, "docs/benchmarks.md needs exactly one begin marker"
    assert text.count(TABLE_END_MARKER) == 1, "docs/benchmarks.md needs exactly one end marker"
    assert text.index(TABLE_BEGIN_MARKER) < text.index(TABLE_END_MARKER)


def test_the_committed_artifact_equals_the_spliced_regeneration() -> None:
    """EQUALITY, not containment.

    `test_benchmarks_md_snapshot.py` asserts the table is *in* the file, which a
    file truncated to the table also satisfies. This asserts the file *is* what
    regenerating it produces, so a deletion anywhere outside the table is visible.
    """
    committed = BENCHMARKS_MD.read_text(encoding="utf-8")
    expected = splice_markdown_table(committed, aggregate_markdown(_committed_results()))
    assert committed == expected, (
        "docs/benchmarks.md is not what regenerating it produces. Regenerate with:\n"
        "  emb-shootout sweep aggregate --results-dir results --out docs/benchmarks.md"
    )


def test_the_documented_command_preserves_the_prose(tmp_path: Path) -> None:
    """Run the command the file's own opening paragraph names, and check the file.

    Against a copy, so the committed artifact is untouched whatever the outcome.
    """
    target = tmp_path / "benchmarks.md"
    shutil.copyfile(BENCHMARKS_MD, target)
    before = target.read_text(encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "emb_shootout.cli",
            "sweep",
            "aggregate",
            "--results-dir",
            str(RESULTS_DIR),
            "--out",
            str(target),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"{proc.returncode}\n{proc.stdout}{proc.stderr}"

    after = target.read_text(encoding="utf-8")
    for needle in REQUIRED_PROSE:
        assert needle in after, (
            f"regenerating docs/benchmarks.md deleted {needle!r}. The file's own "
            f"opening paragraph tells the operator to run this command (#145)."
        )
    assert len(after.splitlines()) >= len(before.splitlines()), (
        f"regeneration shrank the file from {len(before.splitlines())} to "
        f"{len(after.splitlines())} lines"
    )
    assert after == before, "regenerating an already-current artifact must be a no-op"


def test_regeneration_is_idempotent(tmp_path: Path) -> None:
    """Two runs must agree, or `git diff` after a regen is noise and gets ignored."""
    target = tmp_path / "benchmarks.md"
    shutil.copyfile(BENCHMARKS_MD, target)
    rendered = aggregate_markdown(_committed_results())
    once = splice_markdown_table(target.read_text(encoding="utf-8"), rendered)
    twice = splice_markdown_table(once, rendered)
    assert once == twice


# --- the splice contract ----------------------------------------------------


def test_splice_returns_the_render_when_there_are_no_markers() -> None:
    """A scratch `--out` must behave exactly as it did before #145.

    This is what keeps the change from being a breaking one: every existing
    caller that writes to a fresh path, or to a file it intends to be table-only,
    is unaffected.
    """
    assert splice_markdown_table("", "TABLE\n") == "TABLE\n"
    assert splice_markdown_table("prose with no markers\n", "TABLE\n") == "TABLE\n"


def test_splice_replaces_only_the_marked_region() -> None:
    existing = f"HEAD\n{TABLE_BEGIN_MARKER}\nold table\n{TABLE_END_MARKER}\nTAIL\n"
    out = splice_markdown_table(existing, "new table\n")
    assert out == f"HEAD\n{TABLE_BEGIN_MARKER}\nnew table\n{TABLE_END_MARKER}\nTAIL\n"
    assert "old table" not in out, "the generator must be able to shrink its own region"


def test_splice_falls_back_on_an_unclosed_marker() -> None:
    """Declared behaviour, not modelled: an opening marker with no close is a
    damaged artifact, and guessing where the region ends would be worse than
    falling back to the pre-#145 whole-file write, which the snapshot lock then
    reports."""
    existing = f"HEAD\n{TABLE_BEGIN_MARKER}\nhalf-edited\n"
    assert splice_markdown_table(existing, "TABLE\n") == "TABLE\n"


def test_the_markers_are_invisible_in_rendered_markdown() -> None:
    """HTML comments, so GitHub's view of the file is unchanged."""
    for marker in (TABLE_BEGIN_MARKER, TABLE_END_MARKER):
        assert marker.startswith("<!--")
        assert marker.endswith("-->")


def test_the_destructive_behaviour_is_actually_gone(tmp_path: Path) -> None:
    """The falsification, run rather than asserted.

    Reproduces the pre-#145 shape — a marker-less copy of the artifact — and
    confirms it still truncates, so the preservation above is demonstrably coming
    from the markers and not from some unrelated change in the renderer.
    """
    text = BENCHMARKS_MD.read_text(encoding="utf-8")
    stripped = text.replace(TABLE_BEGIN_MARKER + "\n", "").replace(TABLE_END_MARKER + "\n", "")
    rendered = aggregate_markdown(_committed_results())
    assert splice_markdown_table(stripped, rendered) == rendered, (
        "without markers the whole file is still replaced — that is the pre-#145 "
        "behaviour and the reason the markers exist"
    )
    assert "no-fabricated-benchmarks" not in splice_markdown_table(stripped, rendered)
