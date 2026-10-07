"""`sweep run`'s corpus reader refuses a lone surrogate, like `validate_corpus` (#189).

#176 found that a lone surrogate (`\\ud800`) loads from JSON but cannot be
encoded as UTF-8, and #177 added the check to `validate_corpus`. The fail-fast
reader `sweep run` uses (`cli._read_corpus_jsonl`) says it mirrors the
validator "so the two loaders agree on a valid row", and did not get the rule.
Measured on `main` (a hunt agent, re-run here) on a 21-row corpus with one
surrogate: `corpus validate` exits 1 with `unencodable_text`; the reader accepts
all 21 rows; and an embedder that escapes the character the way an HTTP SDK
does embedded 26 texts (21 corpus + 5 queries) before `fingerprint_corpus`
raised `UnicodeEncodeError` -- the paid-work-then-crash harm #176 named.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emb_shootout.cli import _read_corpus_jsonl, main
from emb_shootout.validate import first_lone_surrogate, validate_corpus


def _corpus(tmp_path: Path, bad_field: str) -> Path:
    rows = [{"chunk_id": f"c{i}", "text": f"chunk text number {i}"} for i in range(20)]
    rows.append({"chunk_id": "c20", "text": "fine"})
    rows[-1][bad_field] = "x\ud800y"
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


@pytest.mark.parametrize("field", ["text", "chunk_id"])
def test_the_reader_refuses_a_lone_surrogate_with_its_line_and_field(
    tmp_path: Path, field: str
) -> None:
    with pytest.raises(
        ValueError, match=rf":21: field '{field}' contains the lone surrogate U\+D800 at index 1"
    ):
        _read_corpus_jsonl(_corpus(tmp_path, field))


def test_the_validator_and_the_reader_agree(tmp_path: Path) -> None:
    path = _corpus(tmp_path, "text")
    codes = [f.code for f in validate_corpus(path).findings]
    assert "unencodable_text" in codes
    with pytest.raises(ValueError, match="lone surrogate"):
        _read_corpus_jsonl(path)


def test_sweep_run_exits_2_before_embedding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "sweep",
            "run",
            "--provider",
            "hash",
            "--corpus",
            str(_corpus(tmp_path, "text")),
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 2
    assert "lone surrogate" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize(
    ("value", "expected"),
    [("abc", None), ("a\ud800", (1, 0xD800)), ("\udfff", (0, 0xDFFF)), ("😀", None)],
)
def test_first_lone_surrogate(value: str, expected: object) -> None:
    assert first_lone_surrogate(value) == expected
