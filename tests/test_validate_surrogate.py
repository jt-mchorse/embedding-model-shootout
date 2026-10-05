"""A corpus `validate` accepts must not crash `sweep run` on encoding (#176)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from emb_shootout.validate import validate_corpus

_ROWS = [
    {"chunk_id": "a", "doc_id": "d1", "text": "alpha beta gamma delta epsilon"},
    {"chunk_id": "b", "doc_id": "d2", "text": "zeta eta theta iota kappa lambda"},
]


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    # `json.dumps` escapes a lone surrogate as `\ud800`, which is exactly the
    # on-disk shape that loads cleanly and then fails to encode.
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "emb_shootout.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_finding_names_codepoint_and_index(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    _write(p, [{"chunk_id": "a", "text": "ab\ud800cd\udc00"}])
    (finding,) = validate_corpus(p).findings
    assert finding.code == "unencodable_text"
    assert "U+D800 at index 2" in finding.reason


def test_both_fields_reported_on_one_row(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    _write(p, [{"chunk_id": "\udc80", "text": "\ud800"}])
    codes = [f.code for f in validate_corpus(p).findings]
    assert codes == ["unencodable_chunk_id", "unencodable_text"]


def test_surrogate_pair_in_json_is_one_character_and_valid(tmp_path: Path) -> None:
    # Control: a non-BMP character written as an escaped PAIR decodes to one
    # real character (category So), so it must stay valid.
    p = tmp_path / "c.jsonl"
    p.write_text('{"chunk_id": "a", "text": "smile \\ud83d\\ude00"}\n', encoding="utf-8")
    report = validate_corpus(p)
    assert report.ok, report.findings


def test_cli_exits_one_on_surrogate_corpus(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    rows = [dict(_ROWS[0], text=_ROWS[0]["text"] + " \ud800"), _ROWS[1]]
    _write(p, rows)
    proc = _cli("corpus", "validate", str(p))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "unencodable_text" in proc.stderr


def test_validate_clean_corpus_runs_the_sweep(tmp_path: Path) -> None:
    # The contract the pre-flight exists for, end to end through the real CLI:
    # the rejected corpus crashes `sweep run`; the accepted one completes.
    bad = tmp_path / "bad.jsonl"
    _write(bad, [dict(_ROWS[0], text=_ROWS[0]["text"] + " \ud800"), _ROWS[1]])
    good = tmp_path / "good.jsonl"
    _write(good, _ROWS)
    for path, validate_rc in ((bad, 1), (good, 0)):
        assert _cli("corpus", "validate", str(path)).returncode == validate_rc
    out = tmp_path / "o.json"
    sweep = ("sweep", "run", "--provider", "hash", "--queries", "2", "--out", str(out))
    assert _cli(*sweep, "--corpus", str(bad)).returncode != 0
    proc = _cli(*sweep, "--corpus", str(good))
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
