"""CLI smoke tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from emb_shootout.cli import main


def test_cli_corpus_build_subset(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    out = tmp_path / "corpus.jsonl"
    rc = main(
        [
            "corpus",
            "build",
            "--out",
            str(out),
            "--module",
            "json",
            "--module",
            "math",
        ]
    )
    assert rc == 0
    assert out.exists()
    captured = capsys.readouterr().out
    summary = json.loads(captured)
    assert summary["chunk_count"] > 0
    assert summary["out"] == str(out)
    # Every line of the output JSONL must parse.
    for line in out.read_text(encoding="utf-8").splitlines():
        json.loads(line)


def test_cli_invoked_as_module(tmp_path: Path):
    """`python -m emb_shootout.cli ...` works (entry point for the README)."""
    out = tmp_path / "corpus.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "emb_shootout.cli",
            "corpus",
            "build",
            "--out",
            str(out),
            "--module",
            "math",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    assert summary["chunk_count"] > 0
