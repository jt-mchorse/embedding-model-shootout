"""Corpus-JSONL collecting-mode lint (#45).

``_read_corpus_jsonl`` in ``cli.py`` is fail-fast: the first malformed
line, missing field, or non-string value aborts the read partway through.
An operator who hand-edits ``data/corpus.jsonl`` (an invited workflow —
corpus is reproducible-from-source per D-002, but a small operator-supplied
slice is the obvious next step) sees one error per ``sweep run`` attempt:
fix, retry, fix, retry.

``validate_corpus(path)`` walks the same file in *collecting* mode and
returns every finding in one pass. Same shape as ``validate_dataset`` in
``llm-eval-harness`` (#56/#57) and ``validate_snapshots`` in
``prompt-regression-suite`` (#49/#50) — a frozen-dataclass report with
``ok`` flag, ``n_rows`` / ``n_valid`` counters, and a tuple of findings,
each ``(line_no, reason, code)``.

Finding codes (line numbers are 1-indexed; blank lines are silently
skipped to match ``_read_corpus_jsonl``):

- ``malformed_json``      — ``json.loads`` raised.
- ``not_an_object``       — parsed JSON, but the row is not a JSON object
                            (e.g., a bare string or array).
- ``missing_chunk_id``    — required ``chunk_id`` field absent.
- ``missing_text``        — required ``text`` field absent.
- ``non_string_chunk_id`` — ``chunk_id`` present but not a string.
- ``non_string_text``     — ``text`` present but not a string.
- ``empty_chunk_id``      — ``chunk_id`` is an empty string.
- ``empty_text``          — ``text`` is an empty string.
- ``duplicate_chunk_id``  — same ``chunk_id`` seen at multiple lines; the
                            duplicate row is reported (first occurrence
                            line is named in the message).
- ``empty``               — file contained zero rows; reported once with
                            ``line_no=0``.

A missing path raises ``FileNotFoundError`` so the CLI can surface it as
exit 2 alongside other I/O errors — same convention as the harness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationFinding:
    """One row-level issue surfaced by ``validate_corpus``.

    Mirrors ``eval_harness.dataset.ValidationFinding``: ``line_no`` is
    1-indexed, ``reason`` is the human-readable string, ``code`` is the
    machine-routable label (see module docstring for the list).
    """

    line_no: int
    reason: str
    code: str

    def to_dict(self) -> dict[str, Any]:
        return {"line_no": self.line_no, "reason": self.reason, "code": self.code}


@dataclass(frozen=True)
class ValidationReport:
    """Result of walking a corpus JSONL in collecting mode.

    ``ok`` is true iff the file contained at least one valid row AND
    there are zero findings. An empty file is a finding shape (``empty``
    code, ``line_no=0``) and yields ``ok=False`` without an extra
    ``empty=True`` flag — same convention as the harness.
    """

    path: str
    n_rows: int
    n_valid: int
    findings: tuple[ValidationFinding, ...]

    @property
    def ok(self) -> bool:
        return not self.findings and self.n_valid > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "ok": self.ok,
            "n_rows": self.n_rows,
            "n_valid": self.n_valid,
            "findings": [f.to_dict() for f in self.findings],
        }


def validate_corpus(path: str | Path) -> ValidationReport:
    """Walk a corpus JSONL in collecting mode; see module docstring."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    findings: list[ValidationFinding] = []
    seen_ids: dict[str, int] = {}
    n_rows = 0
    n_valid = 0

    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                # Blank lines are silently skipped by `_read_corpus_jsonl`;
                # match that to keep validate-and-then-run agreeing on the
                # row count. They do not count toward `n_rows`.
                continue
            n_rows += 1
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as e:
                findings.append(
                    ValidationFinding(
                        line_no=line_no,
                        reason=f"invalid JSON: {e.msg}",
                        code="malformed_json",
                    )
                )
                continue
            if not isinstance(obj, dict):
                findings.append(
                    ValidationFinding(
                        line_no=line_no,
                        reason=(
                            f"row must be a JSON object with chunk_id and text fields, "
                            f"got {type(obj).__name__}"
                        ),
                        code="not_an_object",
                    )
                )
                continue

            row_findings = _validate_row(obj, line_no)
            findings.extend(row_findings)

            # The duplicate-chunk_id check is independent of the other field
            # checks: a duplicate is a real, separate finding even when the row
            # also has (say) an empty text, and collecting mode must surface
            # every finding in one pass. Run it whenever the chunk_id field is
            # itself valid (present, string, non-empty) — a missing/empty/
            # non-string chunk_id is already reported by `_validate_row`, so the
            # guard avoids a KeyError and a junk `seen_ids` entry. A valid
            # chunk_id is recorded even when the row has other errors, so a later
            # row reusing it is still flagged.
            row_has_duplicate = False
            chunk_id = obj.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id != "":
                if chunk_id in seen_ids:
                    row_has_duplicate = True
                    findings.append(
                        ValidationFinding(
                            line_no=line_no,
                            reason=(
                                f"duplicate chunk_id {chunk_id!r}; first seen at line "
                                f"{seen_ids[chunk_id]}; chunk_id must be unique within a corpus"
                            ),
                            code="duplicate_chunk_id",
                        )
                    )
                else:
                    seen_ids[chunk_id] = line_no

            # Only a fully clean row counts as valid.
            if row_findings or row_has_duplicate:
                continue
            n_valid += 1

    if n_rows == 0 and not findings:
        findings.append(
            ValidationFinding(
                line_no=0,
                reason=f"corpus file {path} contains no rows",
                code="empty",
            )
        )

    return ValidationReport(
        path=str(path),
        n_rows=n_rows,
        n_valid=n_valid,
        findings=tuple(findings),
    )


def _validate_row(obj: dict[str, Any], line_no: int) -> list[ValidationFinding]:
    """Per-row schema checks for the required (chunk_id, text) pair.

    All findings on the same row are collected in one call so multiple
    independent problems (e.g., missing chunk_id AND empty text) surface
    together rather than one-per-revalidate.
    """
    findings: list[ValidationFinding] = []
    for field in ("chunk_id", "text"):
        if field not in obj:
            findings.append(
                ValidationFinding(
                    line_no=line_no,
                    reason=f"missing required field {field!r}",
                    code=f"missing_{field}",
                )
            )
            continue
        value = obj[field]
        if not isinstance(value, str):
            findings.append(
                ValidationFinding(
                    line_no=line_no,
                    reason=(f"field {field!r} must be a string, got {type(value).__name__}"),
                    code=f"non_string_{field}",
                )
            )
            continue
        if value == "":
            findings.append(
                ValidationFinding(
                    line_no=line_no,
                    reason=f"field {field!r} must not be empty",
                    code=f"empty_{field}",
                )
            )
    return findings
