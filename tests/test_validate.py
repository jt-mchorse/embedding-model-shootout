"""Tests for ``validate_corpus`` and the ``emb-shootout corpus validate`` CLI (#45).

Coverage matrix:

- happy path on a small clean corpus → ``ok=True``, no findings.
- accumulating-errors path: a synthetic file with three different bad
  rows + one valid row surfaces three findings in line-number order,
  not failing fast on the first.
- one positive case per finding code (10 codes including ``empty``).
- duplicate-``chunk_id`` detection: validator reports the duplicate
  and does not count the shadowed row as a second valid row.
- blank lines silently skipped: present in input but absent from
  ``n_rows`` and from the findings.
- missing file: ``FileNotFoundError`` propagates from the library; CLI
  surfaces exit code 2.
- ``ValidationReport.to_dict`` is JSON-stable: top-level keys, per-
  finding keys, monotonically-increasing ``line_no`` within ``findings``.
- CLI: clean fixture exits 0 with a one-line ``ok:`` summary on stdout
  and nothing on stderr.
- CLI: malformed fixture exits 1 with one stderr line per finding and a
  ``fail:`` summary on stdout.
- CLI: ``--json`` emits the report dict on stdout and still respects
  the exit code.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from emb_shootout.validate import (
    ValidationFinding,
    ValidationReport,
    validate_corpus,
)


def _write_jsonl(path: Path, rows: list[dict | str]) -> None:
    """Write rows to ``path``. ``str`` rows are emitted verbatim so a test
    can inject a malformed line; ``dict`` rows go through ``json.dumps``.
    """
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            if isinstance(row, str):
                fh.write(row + "\n")
            else:
                fh.write(json.dumps(row) + "\n")


def _valid_row(chunk_id: str = "a", text: str = "hello") -> dict[str, str]:
    return {"chunk_id": chunk_id, "text": text}


# ---------------------------------------------------------------------------
# Library: validate_corpus
# ---------------------------------------------------------------------------


def test_happy_path_clean_corpus(tmp_path: Path) -> None:
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [_valid_row("a", "hello"), _valid_row("b", "world")])
    report = validate_corpus(p)
    assert report.ok
    assert report.n_rows == 2
    assert report.n_valid == 2
    assert report.findings == ()
    assert report.path == str(p)


def test_collects_multiple_findings_not_failing_fast(tmp_path: Path) -> None:
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(
        p,
        [
            _valid_row("ok", "first"),  # line 1: valid
            "{not valid json",  # line 2: malformed_json
            {"chunk_id": "missing_text_row"},  # line 3: missing_text
            {"text": "missing_id_row"},  # line 4: missing_chunk_id
            _valid_row("ok2", "fourth_valid"),  # line 5: valid
        ],
    )
    report = validate_corpus(p)
    assert not report.ok
    assert report.n_rows == 5
    assert report.n_valid == 2
    codes = [f.code for f in report.findings]
    line_nos = [f.line_no for f in report.findings]
    assert codes == ["malformed_json", "missing_text", "missing_chunk_id"]
    assert line_nos == [2, 3, 4]


@pytest.mark.parametrize(
    ("row", "code"),
    [
        ("{not valid json", "malformed_json"),
        ('"just_a_string"', "not_an_object"),
        ({"text": "no_id"}, "missing_chunk_id"),
        ({"chunk_id": "no_text"}, "missing_text"),
        ({"chunk_id": 123, "text": "x"}, "non_string_chunk_id"),
        ({"chunk_id": "x", "text": 123}, "non_string_text"),
        ({"chunk_id": "", "text": "x"}, "empty_chunk_id"),
        ({"chunk_id": "x", "text": ""}, "empty_text"),
    ],
)
def test_one_positive_case_per_finding_code(tmp_path: Path, row: dict | str, code: str) -> None:
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [row])
    report = validate_corpus(p)
    assert not report.ok
    assert len(report.findings) == 1
    assert report.findings[0].code == code
    assert report.findings[0].line_no == 1
    assert report.n_valid == 0


def test_duplicate_chunk_id_reports_second_occurrence(tmp_path: Path) -> None:
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(
        p,
        [
            _valid_row("dup", "first"),
            _valid_row("unique", "second"),
            _valid_row("dup", "third_shadowed"),
        ],
    )
    report = validate_corpus(p)
    assert not report.ok
    assert report.n_rows == 3
    # First two are valid; third shadows the first chunk_id and does not
    # count as a second valid row for ``dup``.
    assert report.n_valid == 2
    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.code == "duplicate_chunk_id"
    assert finding.line_no == 3
    assert "line 1" in finding.reason  # first-seen-at message


def test_empty_file_surfaces_empty_finding(tmp_path: Path) -> None:
    p = tmp_path / "corpus.jsonl"
    p.write_text("", encoding="utf-8")
    report = validate_corpus(p)
    assert not report.ok
    assert report.n_rows == 0
    assert report.n_valid == 0
    assert len(report.findings) == 1
    assert report.findings[0].code == "empty"
    assert report.findings[0].line_no == 0


def test_blank_lines_are_silently_skipped(tmp_path: Path) -> None:
    """Match ``_read_corpus_jsonl`` so validate-and-then-run agree.

    Blank lines in the input do not count toward ``n_rows`` and do not
    produce findings. A file of only blank lines is treated as empty.
    """
    p = tmp_path / "corpus.jsonl"
    # One valid row sandwiched between blank lines.
    p.write_text("\n\n" + json.dumps(_valid_row()) + "\n\n", encoding="utf-8")
    report = validate_corpus(p)
    assert report.ok
    assert report.n_rows == 1
    assert report.n_valid == 1
    assert report.findings == ()


def test_only_blank_lines_treated_as_empty(tmp_path: Path) -> None:
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n\n\n", encoding="utf-8")
    report = validate_corpus(p)
    assert not report.ok
    assert report.n_rows == 0
    assert len(report.findings) == 1
    assert report.findings[0].code == "empty"


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    p = tmp_path / "does_not_exist.jsonl"
    with pytest.raises(FileNotFoundError):
        validate_corpus(p)


def test_report_to_dict_is_json_stable(tmp_path: Path) -> None:
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [_valid_row("a"), {"text": "no_id"}])
    report = validate_corpus(p)
    d = report.to_dict()
    # Top-level keys are exactly the published surface.
    assert set(d.keys()) == {"path", "ok", "n_rows", "n_valid", "findings"}
    assert d["ok"] is False
    assert d["n_rows"] == 2
    assert d["n_valid"] == 1
    # Findings each carry the same three keys.
    assert all(set(f.keys()) == {"line_no", "reason", "code"} for f in d["findings"])
    # Whole dict round-trips through json without raising.
    assert json.loads(json.dumps(d)) == d


def test_finding_and_report_are_frozen_dataclasses() -> None:
    """Dataclass shape: ``frozen=True`` so callers can hash / cache reports.

    Mirrors the harness and prompt-snap conventions; if a future edit
    drops ``frozen=True`` this catches it.
    """
    assert dataclasses.is_dataclass(ValidationFinding)
    assert dataclasses.is_dataclass(ValidationReport)
    finding = ValidationFinding(line_no=1, reason="x", code="malformed_json")
    with pytest.raises(dataclasses.FrozenInstanceError):
        finding.line_no = 2  # type: ignore[misc]
    report = ValidationReport(path="x", n_rows=0, n_valid=0, findings=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.n_rows = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CLI: emb-shootout corpus validate
# ---------------------------------------------------------------------------


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "emb_shootout.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_clean_corpus_exits_zero(tmp_path: Path) -> None:
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [_valid_row("a"), _valid_row("b")])
    proc = _run_cli("corpus", "validate", str(p))
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""
    assert proc.stdout.startswith("ok:")
    assert "rows=2" in proc.stdout
    assert "valid=2" in proc.stdout
    assert "findings=0" in proc.stdout


def test_cli_malformed_corpus_exits_one_with_per_finding_stderr(tmp_path: Path) -> None:
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(
        p,
        [
            _valid_row("ok"),
            "{not valid json",
            {"text": "no_id"},
        ],
    )
    proc = _run_cli("corpus", "validate", str(p))
    assert proc.returncode == 1
    assert proc.stdout.startswith("fail:")
    assert "rows=3" in proc.stdout
    assert "valid=1" in proc.stdout
    assert "findings=2" in proc.stdout
    # One stderr line per finding, line-numbered.
    err_lines = [ln for ln in proc.stderr.splitlines() if ln.strip()]
    assert len(err_lines) == 2
    assert err_lines[0].startswith("line 2 [malformed_json]")
    assert err_lines[1].startswith("line 3 [missing_chunk_id]")


def test_cli_json_flag_emits_report_dict_and_respects_exit_code(tmp_path: Path) -> None:
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [_valid_row("ok"), {"text": "no_id"}])
    proc = _run_cli("corpus", "validate", str(p), "--json")
    assert proc.returncode == 1
    # Stderr is empty under --json (the JSON is the full report).
    assert proc.stderr == ""
    parsed = json.loads(proc.stdout)
    assert parsed["ok"] is False
    assert parsed["n_rows"] == 2
    assert parsed["n_valid"] == 1
    assert len(parsed["findings"]) == 1
    assert parsed["findings"][0]["code"] == "missing_chunk_id"
    assert parsed["findings"][0]["line_no"] == 2


def test_cli_missing_file_exits_two(tmp_path: Path) -> None:
    p = tmp_path / "does_not_exist.jsonl"
    proc = _run_cli("corpus", "validate", str(p))
    assert proc.returncode == 2
    assert "corpus not found" in proc.stderr
