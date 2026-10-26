"""`corpus build` refuses to write a corpus with zero chunks (#260).

`_cmd_corpus_build` wrote before it computed the skipped set and never
checked for an empty result. Measured on `main` 835ae11:

    corpus build --out good.jsonl --module json   -> 21831 bytes
    corpus build --out good.jsonl --module jsno   -> rc 0, good.jsonl 0 bytes

The writer emitted what its own validator ("contains no rows") and `sweep run`
("corpus must be non-empty") refuse.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from emb_shootout.cli import main


def _build(out: Path, *modules: str) -> int:
    argv = ["corpus", "build", "--out", str(out)]
    for m in modules:
        argv += ["--module", m]
    return main(argv)


def test_an_all_typo_build_leaves_an_existing_corpus_byte_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "good.jsonl"
    assert _build(out, "json") == 0
    before = out.read_bytes()
    assert before  # the control: a real corpus is there to lose
    capsys.readouterr()

    rc = _build(out, "jsno")
    captured = capsys.readouterr()
    assert rc == 2
    assert out.read_bytes() == before
    assert "skipped module 'jsno'" in captured.err
    assert "produced no chunks" in captured.err
    assert captured.out == ""  # no success summary for a build that wrote nothing


def test_an_all_typo_build_creates_no_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "new.jsonl"
    assert _build(out, "jsno", "csvv") == 2
    assert not out.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == []  # no stray temp file
    err = capsys.readouterr().err
    assert "skipped module 'csvv'" in err
    assert "skipped module 'jsno'" in err


def test_an_importable_module_with_no_members_is_refused_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Zero chunks with NOTHING skipped: the module imports but documents no
    # members. The refusal keys on the chunk count, not on the skipped set.
    monkeypatch.setitem(sys.modules, "ems_260_empty", types.ModuleType("ems_260_empty"))
    out = tmp_path / "c.jsonl"
    out.write_text("sentinel\n", encoding="utf-8")
    assert _build(out, "ems_260_empty") == 2
    assert out.read_text(encoding="utf-8") == "sentinel\n"
    err = capsys.readouterr().err
    assert "skipped module" not in err
    assert "produced no chunks" in err


def test_a_partial_skip_still_writes_and_exits_0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Control: #172's behaviour for a partial skip is unchanged.
    out = tmp_path / "c.jsonl"
    assert _build(out, "json", "csvv") == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out.strip().splitlines()[-1])
    assert summary["chunk_count"] > 0
    assert summary["modules_skipped"] == ["csvv"]
    assert out.stat().st_size > 0
    assert "skipped module 'csvv'" in captured.err
