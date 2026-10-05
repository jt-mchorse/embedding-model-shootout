"""`corpus build` reports the modules it skipped (#172).

`build_corpus` skips a module that fails to import, by design, and its
docstring promises "the set of skipped modules is reported by the CLI in JSON
output". Measured on `main`:

    corpus build --module json --module csvv
    {"chunk_count": 20, "modules_requested": 2, "out": ...}     exit 0

The typo vanished.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emb_shootout.cli import main
from emb_shootout.corpus import DEFAULT_MODULES, build_corpus, unimportable_modules


def _build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], *modules: str
) -> tuple[int, dict, str]:
    argv = ["corpus", "build", "--out", str(tmp_path / "c.jsonl")]
    for m in modules:
        argv += ["--module", m]
    rc = main(argv)
    out = capsys.readouterr()
    return rc, json.loads(out.out.strip().splitlines()[-1]), out.err


def test_a_mistyped_module_is_reported_in_the_json_and_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, summary, err = _build(tmp_path, capsys, "json", "csvv")
    assert rc == 0  # skipping is the documented behaviour; it is now visible
    assert summary["modules_requested"] == 2
    assert summary["modules_skipped"] == ["csvv"]
    assert "skipped module 'csvv'" in err


def test_all_importable_modules_report_an_empty_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, summary, err = _build(tmp_path, capsys, "json", "csv")
    assert rc == 0
    assert summary["modules_skipped"] == []
    assert err == ""


def test_the_default_list_reports_exactly_what_build_corpus_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Whatever this interpreter lacks (telnetlib is gone on 3.13+), the report
    # names exactly the modules that contributed no chunks.
    _, summary, _ = _build(tmp_path, capsys)
    built = {c.module for c in build_corpus(DEFAULT_MODULES)}
    assert summary["modules_skipped"] == unimportable_modules(DEFAULT_MODULES)
    assert set(summary["modules_skipped"]).isdisjoint(built)
