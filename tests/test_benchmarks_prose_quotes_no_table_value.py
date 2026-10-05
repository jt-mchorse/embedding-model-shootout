"""The prose `sweep aggregate` preserves quotes no number only the table can keep current (#180).

`docs/benchmarks.md` is regenerated between the `emb-shootout:table` markers
and its surrounding prose is carried through unchanged (#145). That prose said
"recall@5 of **0.52**", so a regeneration from any other result -- a hunt
agent's `--queries 7 --seed 3` gave 0.429; #115 records 0.580 on Python 3.11
-- left the old number under the new table.
"""

from __future__ import annotations

import re
from pathlib import Path

_DOC = Path(__file__).resolve().parents[1] / "docs" / "benchmarks.md"
_BEGIN, _END = "<!-- emb-shootout:table:begin -->", "<!-- emb-shootout:table:end -->"


def _split(text: str) -> tuple[str, str]:
    head, rest = text.split(_BEGIN, 1)
    table, tail = rest.split(_END, 1)
    return table, head + tail


def _quoted_table_values(table: str, prose: str) -> list[str]:
    quality = set()
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 8 and cells[1].isdigit():
            quality.update(cells[4:8])  # recall@1/5/10, NDCG@10
    hits = []
    for value in sorted(quality):
        forms = {value, value.rstrip("0").rstrip(".")}
        for form in forms:
            if re.search(rf"(?<![\d.]){re.escape(form)}(?![\d])", prose):
                hits.append(form)
    return hits


def test_the_preserved_prose_quotes_no_table_value() -> None:
    table, prose = _split(_DOC.read_text(encoding="utf-8"))
    assert _quoted_table_values(table, prose) == []


def test_the_check_sees_the_sentence_it_was_written_for() -> None:
    table, _ = _split(_DOC.read_text(encoding="utf-8"))
    old = "`hash-embedder-128d-ngram2`'s recall@5 of **0.52** isn't the takeaway"
    assert _quoted_table_values(table, old) == ["0.52"]
