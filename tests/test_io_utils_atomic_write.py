"""Atomicity contract for `emb_shootout.io_utils.atomic_write_text` (issue #37).

Until this PR, four production write sites in this repo used
`Path.write_text` or the streaming `open("w") + f.write()` shape, both
non-atomic: SIGINT/SIGTERM/disk-full/OOM between the implicit
`open(..., "w")` truncate and `close()` flush leaves the destination
zero-length or partial.

The harm shape is amplified by D-007's per-provider result-JSON
partitioning: the aggregator and the plot subcommand scan `*.json`
files in a directory, so a single half-written sweep result either
poisons the aggregator (`json.JSONDecodeError`) or silently truncates
the plotted Pareto frontier. The corpus write path is worse: rows are
parsed one-per-line, so a truncation at a row boundary passes the
parser silently — quality numbers drift down without a loud signal.

This PR routes all four sites through a new public helper at
`emb_shootout.io_utils.atomic_write_text`, matching the portfolio
standard (`rag_kit/io_utils.atomic_write_text` from
`rag-production-kit#44/#45`, `eval_harness/io_utils.atomic_write_text`
from `llm-eval-harness#50` D-015, and similar). D-009 in this repo
codifies the pattern.

What this file pins:

1. **Helper unit contract** (6 tests): happy path, parent-dir
   creation, overwrite, the three load-bearing failure invariants —
   destination-absent on rename failure for new files, no leftover
   `.tmp` siblings, pre-existing-file-unchanged on overwrite failure.
2. **Per-call-site integration** (4 tests): `sweep run --output`,
   `sweep aggregate --out`, `corpus build --out` (via `write_jsonl`),
   and the notebook build via `notebooks/_build_notebook.py`. Each
   asserts that monkey-patching `io_utils.os.replace` to raise leaves
   the destination untouched (or pre-existing content intact for the
   overwrite case).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from emb_shootout import io_utils as io_utils_mod
from emb_shootout.cli import main as cli_main
from emb_shootout.corpus import Chunk, write_jsonl
from emb_shootout.io_utils import atomic_write_text

# ---------------------------------------------------------------------------
# Unit tests on the helper itself.
# ---------------------------------------------------------------------------


def test_atomic_write_text_happy_path(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    atomic_write_text(out, "hello\nworld\n")
    assert out.read_text(encoding="utf-8") == "hello\nworld\n"


def test_atomic_write_text_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "x.json"
    assert not out.parent.exists()
    atomic_write_text(out, "{}")
    assert out.read_text(encoding="utf-8") == "{}"


def test_atomic_write_text_overwrites_existing_file(tmp_path: Path) -> None:
    """Stale content is wholly replaced, not appended."""
    out = tmp_path / "out.txt"
    out.write_text("STALE-CONTENT-MUST-NOT-SURVIVE", encoding="utf-8")
    atomic_write_text(out, "fresh")
    body = out.read_text(encoding="utf-8")
    assert body == "fresh"
    assert "STALE" not in body


def test_atomic_write_text_replace_failure_leaves_destination_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If `os.replace` raises (EXDEV, SIGINT between fsync and rename,
    PermissionError), the destination must not exist. The helper must
    never touch the destination directly — only via the atomic rename.
    """
    out = tmp_path / "result.json"

    def boom(*_args, **_kwargs):
        raise OSError("simulated mid-rename failure")

    monkeypatch.setattr(io_utils_mod.os, "replace", boom)
    with pytest.raises(OSError, match="simulated mid-rename failure"):
        atomic_write_text(out, '{"k": "v"}')

    assert not out.exists(), "destination must remain absent when os.replace fails"


def test_atomic_write_text_replace_failure_cleans_up_tmp_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No leftover `.tmp` siblings after a failed atomic write."""
    out = tmp_path / "artifacts" / "delta.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def boom(*_args, **_kwargs):
        raise OSError("simulated mid-rename failure")

    monkeypatch.setattr(io_utils_mod.os, "replace", boom)
    with pytest.raises(OSError, match="simulated mid-rename failure"):
        atomic_write_text(out, '{"k": "v"}')

    siblings = list(out.parent.iterdir())
    assert siblings == [], f"expected no temp leftovers in {out.parent}, got {siblings}"


def test_atomic_write_text_destination_unchanged_when_overwriting_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed overwrite must leave the pre-existing destination intact.
    The property `Path.write_text` and the `open("w") + write` shape
    could never offer.
    """
    out = tmp_path / "existing.json"
    out.write_text('{"keep": true}', encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise OSError("simulated")

    monkeypatch.setattr(io_utils_mod.os, "replace", boom)
    with pytest.raises(OSError, match="simulated"):
        atomic_write_text(out, '{"overwrite": true}')

    assert out.read_text(encoding="utf-8") == '{"keep": true}'


# ---------------------------------------------------------------------------
# Integration: each call site routes through atomic_write_text.
# Pattern: monkeypatch io_utils.os.replace, exercise the surface, assert
# destination unchanged.
# ---------------------------------------------------------------------------


def _make_corpus_for_sweep(tmp_path: Path) -> Path:
    """Write a tiny dep-free JSONL corpus for hash-provider sweep runs.

    Uses the same shape as ``test_cli_sweep_run_out_alias._make_corpus``
    so this test exercises the real sweep path end-to-end.
    """
    corpus_path = tmp_path / "corpus.jsonl"
    rows = [
        {"chunk_id": f"chunk-{i:03d}", "text": f"document number {i} discusses topic {i % 3}"}
        for i in range(20)
    ]
    corpus_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return corpus_path


def test_sweep_run_output_routes_through_atomic_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`emb-shootout sweep run --output PATH` must route through the helper.

    D-007 says the aggregator scans `*.json` and merges; a half-written
    per-provider result either crashes the aggregator with a
    `JSONDecodeError` or, in the plot subcommand's silent-skip-on-bad-file
    path, vanishes a provider from the Pareto plot.
    """
    corpus = _make_corpus_for_sweep(tmp_path)
    out = tmp_path / "results" / "hash.json"

    def boom(*_args, **_kwargs):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(io_utils_mod.os, "replace", boom)
    with pytest.raises(OSError, match="simulated rename failure"):
        cli_main(
            [
                "sweep",
                "run",
                "--provider",
                "hash",
                "--corpus",
                str(corpus),
                "--queries",
                "5",
                "--seed",
                "42",
                "--output",
                str(out),
            ]
        )

    assert not out.exists(), "sweep run --output must not write destination on replace failure"


def test_sweep_aggregate_out_routes_through_atomic_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`emb-shootout sweep aggregate --out PATH` must route through the helper.

    The README's "Benchmarks" section is rendered from `docs/benchmarks.md`
    — a half-written aggregate is a front-page failure on GitHub.
    """
    # Seed: one real sweep result, written via the (now-atomic) sweep run.
    corpus = _make_corpus_for_sweep(tmp_path)
    results_dir = tmp_path / "results"
    rc = cli_main(
        [
            "sweep",
            "run",
            "--provider",
            "hash",
            "--corpus",
            str(corpus),
            "--queries",
            "5",
            "--seed",
            "42",
            "--output",
            str(results_dir / "hash.json"),
        ]
    )
    assert rc == 0

    out = tmp_path / "docs" / "benchmarks.md"

    def boom(*_args, **_kwargs):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(io_utils_mod.os, "replace", boom)
    with pytest.raises(OSError, match="simulated rename failure"):
        cli_main(
            [
                "sweep",
                "aggregate",
                "--results-dir",
                str(results_dir),
                "--format",
                "markdown",
                "--out",
                str(out),
            ]
        )

    assert not out.exists(), "sweep aggregate --out must not write destination on replace failure"


def test_corpus_write_jsonl_routes_through_atomic_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`corpus.write_jsonl(chunks, path)` must route through the helper.

    The corpus is row-oriented, so a truncation at a row boundary passes
    the parser silently — quality numbers drift down without a loud
    signal. Overwrite-atomicity is load-bearing here: a re-build that
    fails mid-write must leave the prior corpus intact so subsequent
    sweep runs aren't silently degraded.
    """
    chunks = [
        Chunk(
            chunk_id=f"stdlib.example.member_{i}",
            text=f"docstring of member {i}",
            module="stdlib.example",
            qualname=f"example.member_{i}",
            kind="function",
            source="python-stdlib",
        )
        for i in range(8)
    ]

    out = tmp_path / "corpus.jsonl"
    out.write_text("PRE-EXISTING-CORPUS\n", encoding="utf-8")  # for overwrite invariant

    def boom(*_args, **_kwargs):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(io_utils_mod.os, "replace", boom)
    with pytest.raises(OSError, match="simulated rename failure"):
        write_jsonl(chunks, out)

    # The pre-existing corpus must remain intact when the atomic rename
    # fails — the property the prior `open("w") + f.write()` shape could
    # never offer (it would have truncated the destination before the
    # first row's bytes hit disk).
    assert out.read_text(encoding="utf-8") == "PRE-EXISTING-CORPUS\n", (
        "write_jsonl must leave existing file intact on rename failure"
    )


def test_notebook_build_routes_through_atomic_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`notebooks/_build_notebook.py`'s notebook write must route through
    the helper. A half-written `.ipynb` breaks `jupyter nbconvert` and
    the README "Notebook" link.

    Loads the script as a module so the test exercises the real `main()`
    entry point. Redirects `NOTEBOOK_PATH` to a tmp file so we don't
    accidentally trash the committed notebook on disk.
    """
    script_path = Path(__file__).resolve().parent.parent / "notebooks" / "_build_notebook.py"
    spec = importlib.util.spec_from_file_location("_build_notebook_under_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    redirected = tmp_path / "redirected.ipynb"
    monkeypatch.setattr(module, "NOTEBOOK_PATH", redirected)

    def boom(*_args, **_kwargs):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(io_utils_mod.os, "replace", boom)
    with pytest.raises(OSError, match="simulated rename failure"):
        module.main()

    assert not redirected.exists(), "notebook build must not write destination on rename failure"
