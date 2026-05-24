"""Tests for `emb-shootout sweep run --out PATH` (issue #25).

The `sweep run` subcommand pre-#25 required `--output PATH` while every
other path-emitting subcommand in this CLI uses `--out` (`corpus build`,
`sweep aggregate`, `sweep plot --out-png` / `--out-svg`). The fix adds
`--out` as an alias for `--output` so both work; the existing CI /
Makefile callers that pass `--output` keep working unchanged.

This module asserts:
- `--out PATH` produces the same RunResult-shape file as the existing
  `--output PATH` path.
- `--output PATH` still works (back-compat regression guard).
- Supplying neither flag exits 2 (argparse still requires the argument
  under one of its two names).

The hash provider runs against a small JSONL corpus written in tmp_path
so the test is hermetic — no network, no provider dep beyond the
dep-free `hash` baseline embedder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emb_shootout.cli import main


def _make_corpus(tmp_path: Path) -> Path:
    """Write a tiny dep-free JSONL corpus for hash-provider sweep runs."""
    corpus_path = tmp_path / "corpus.jsonl"
    rows = [
        {"chunk_id": f"chunk-{i:03d}", "text": f"document number {i} discusses topic {i % 3}"}
        for i in range(20)
    ]
    corpus_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return corpus_path


def test_sweep_run_with_out_writes_result_json(tmp_path: Path) -> None:
    """The new `--out` alias produces the same RunResult shape as `--output`."""
    corpus = _make_corpus(tmp_path)
    out_path = tmp_path / "results" / "hash.json"
    rc = main(
        [
            "sweep",
            "run",
            "--provider",
            "hash",
            "--corpus",
            str(corpus),
            "--queries",
            "10",
            "--seed",
            "1",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    # Parent dir was auto-created (no manual mkdir in the test).
    assert out_path.exists()
    parsed = json.loads(out_path.read_text(encoding="utf-8"))
    # Shape contract from RunResult.to_dict — the embedder_name field
    # is always present; recall_at_k is a dict keyed by k.
    assert "embedder_name" in parsed
    assert "recall_at_k" in parsed
    assert isinstance(parsed["recall_at_k"], dict)


def test_sweep_run_with_output_still_works(tmp_path: Path) -> None:
    """Regression guard: existing callers passing `--output` must keep working.

    CI / Makefile / shell scripts already invoke `sweep run --output …`;
    the alias is additive — `--output` is not deprecated, only joined.
    """
    corpus = _make_corpus(tmp_path)
    out_path = tmp_path / "results" / "hash-via-output.json"
    rc = main(
        [
            "sweep",
            "run",
            "--provider",
            "hash",
            "--corpus",
            str(corpus),
            "--queries",
            "10",
            "--seed",
            "1",
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
    parsed = json.loads(out_path.read_text(encoding="utf-8"))
    assert parsed["embedder_name"]


def test_sweep_run_requires_one_of_out_or_output(tmp_path: Path) -> None:
    """Neither flag set → argparse exits 2 (the existing required-arg behaviour)."""
    corpus = _make_corpus(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "sweep",
                "run",
                "--provider",
                "hash",
                "--corpus",
                str(corpus),
                "--queries",
                "10",
                "--seed",
                "1",
            ]
        )
    # argparse calls sys.exit(2) on required-arg failure.
    assert excinfo.value.code == 2
