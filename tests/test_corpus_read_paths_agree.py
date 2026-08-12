"""`validate_corpus` and `_read_corpus_jsonl` must agree on what a usable corpus is (#117).

The repo has two readers for `data/corpus.jsonl`: the collecting-mode
`validate_corpus` behind `corpus validate`, and the fail-fast
`_read_corpus_jsonl` that `sweep run` uses. #75 unified their *row-level*
checks, and `_read_corpus_jsonl`'s own comment states the goal — "so the two
loaders agree on a valid row".

`duplicate_chunk_id` was the finding code left behind, because uniqueness is a
*file-level* property and a row-by-row parity pass can't see it. That gap let
`corpus validate` fail a file while `sweep run` benchmarked it and wrote an
artifact at exit 0 — and retrieval scoring is id-equality, so a duplicated id
credits the wrong text for a hit.

`test_read_paths_agree_on_every_case` is the durable half: fixing the duplicate
case alone closes today's instance, while a differential table fails the *next*
divergence instead of waiting for someone to notice it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emb_shootout.cli import _read_corpus_jsonl
from emb_shootout.providers import PROVIDER_REGISTRY
from emb_shootout.queries import build_queries
from emb_shootout.sweep import CorpusChunk, run_sweep
from emb_shootout.validate import validate_corpus


def _row(chunk_id: str, text: str) -> str:
    return json.dumps({"chunk_id": chunk_id, "text": text})


def _write(tmp_path: Path, lines: list[str], name: str = "c.jsonl") -> Path:
    p = tmp_path / name
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return p


# (label, lines, both_accept). `both_accept` is the agreed verdict — the point
# of the test is that the two readers reach it together, not what it is.
_CASES: list[tuple[str, list[str], bool]] = [
    ("ok baseline", [_row("a", "hello world")], True),
    ("two distinct rows", [_row("a", "x"), _row("b", "y")], True),
    ("blank line skipped", [_row("a", "x"), "", _row("b", "y")], True),
    ("whitespace-only line skipped", [_row("a", "x"), "   ", _row("b", "y")], True),
    ("extra unknown field tolerated", ['{"chunk_id":"a","text":"x","zzz":1}'], True),
    ("whitespace-only text tolerated", [_row("a", "   ")], True),
    ("missing chunk_id", ['{"text":"x"}'], False),
    ("missing text", ['{"chunk_id":"a"}'], False),
    ("non-string text", ['{"chunk_id":"a","text":123}'], False),
    ("bool text", ['{"chunk_id":"a","text":true}'], False),
    ("null text", ['{"chunk_id":"a","text":null}'], False),
    ("non-string chunk_id", ['{"chunk_id":1,"text":"x"}'], False),
    ("empty text", [_row("a", "")], False),
    ("empty chunk_id", [_row("", "x")], False),
    ("bare array row", ['["a","b"]'], False),
    ("malformed json", ["{not json"], False),
    ("duplicate chunk_id", [_row("a", "x"), _row("a", "y")], False),
    ("duplicate not adjacent", [_row("a", "x"), _row("b", "y"), _row("a", "z")], False),
]


def _loader_accepts(path: Path) -> bool:
    try:
        _read_corpus_jsonl(path)
    except ValueError:
        return False
    return True


@pytest.mark.parametrize(
    ("label", "lines", "expected_accept"),
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_read_paths_agree_on_every_case(
    label: str, lines: list[str], expected_accept: bool, tmp_path: Path
) -> None:
    path = _write(tmp_path, lines)

    validator_accepts = validate_corpus(path).ok
    loader_accepts = _loader_accepts(path)

    assert validator_accepts == loader_accepts, (
        f"{label}: the two corpus readers disagree — validate_corpus "
        f"{'accepts' if validator_accepts else 'rejects'} but _read_corpus_jsonl "
        f"{'accepts' if loader_accepts else 'rejects'}. `corpus validate` and "
        "`sweep run` must not differ on whether a corpus is usable."
    )
    assert validator_accepts == expected_accept, (
        f"{label}: expected both readers to "
        f"{'accept' if expected_accept else 'reject'}; both "
        f"{'accepted' if validator_accepts else 'rejected'}"
    )


def test_empty_file_divergence_is_covered_downstream(tmp_path: Path) -> None:
    """The one accepted difference, pinned rather than left implicit.

    `validate_corpus` reports an `empty` finding for a zero-row file while
    `_read_corpus_jsonl` returns `[]` without complaint. That does *not* leak a
    usable-but-invalid corpus into a benchmark, because `build_queries` refuses
    an empty corpus and the CLI translates that to exit 2. Pinning it here
    means the difference stays deliberate: if `build_queries` ever stops
    guarding, this fails instead of the gap re-opening silently.
    """
    path = _write(tmp_path, [])

    assert not validate_corpus(path).ok, "an empty corpus is a validator finding"
    assert _read_corpus_jsonl(path) == [], "the fail-fast loader returns no rows"

    with pytest.raises(ValueError, match="corpus must be non-empty"):
        build_queries([], n=5, seed=1)


def test_duplicate_chunk_id_message_names_both_lines(tmp_path: Path) -> None:
    """Mirrors `validate_corpus`'s finding text so an operator can locate the
    collision from either reader."""
    path = _write(tmp_path, [_row("a", "x"), _row("b", "y"), _row("a", "z")])

    with pytest.raises(ValueError, match="duplicate chunk_id") as exc:
        _read_corpus_jsonl(path)

    msg = str(exc.value)
    assert "duplicate chunk_id 'a'" in msg, msg
    assert "first seen at line 1" in msg, msg
    assert f"{path}:3" in msg, msg


def test_duplicate_id_inflated_recall_at_1_and_is_now_rejected(tmp_path: Path) -> None:
    """Anchor on the corruption, not the exception type.

    This is the corpus from #117: two lexically competing chunks plus filler.
    With unique ids recall@1 is 0.9600; duplicating one id raised it to 0.9667,
    because queries drawn from the second chunk that retrieve the first are
    credited for carrying the twin's id. The unique-id half is asserted too, so
    if the retrieval maths ever changes this test tells us the baseline moved
    rather than silently comparing against nothing.
    """
    texts = [
        "postgres connection pooling work_mem tuning analytic workloads vacuum autovacuum settings",
        "postgres connection pooling work_mem tuning analytic workloads checkpoint wal settings",
        *[
            f"unrelated filler document number {i} about kites and weather balloons drifting"
            for i in range(12)
        ],
    ]

    unique = [CorpusChunk(chunk_id=f"c{i}", text=t) for i, t in enumerate(texts)]
    queries = build_queries(unique, n=300, seed=5)
    baseline = run_sweep(unique, queries, embedder=PROVIDER_REGISTRY["hash"](), k_values=(1, 5, 10))
    assert baseline.recall_at_k[1] == pytest.approx(0.9600, abs=1e-4), (
        "the unique-id baseline moved; the inflation figure below is measured "
        f"against it. got recall@1={baseline.recall_at_k[1]}"
    )

    # The same corpus with c1's id duplicated onto c0's value — the file that
    # `corpus validate` has always rejected and `sweep run` used to benchmark.
    dup_lines = [_row("c0" if i == 1 else f"c{i}", t) for i, t in enumerate(texts)]
    path = _write(tmp_path, dup_lines, name="dup.jsonl")

    assert not validate_corpus(path).ok
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        _read_corpus_jsonl(path)


def test_committed_corpus_has_no_duplicate_ids() -> None:
    """The pinned corpus must stay loadable — this fix must move no benchmark
    number. If `data/corpus.jsonl` ever gains a duplicate, that is a corpus
    bug and this names it directly rather than surfacing as a sweep failure."""
    corpus_path = Path(__file__).resolve().parents[1] / "data" / "corpus.jsonl"
    if not corpus_path.exists():  # pragma: no cover - corpus is committed
        pytest.skip(f"{corpus_path} not present")

    chunks = _read_corpus_jsonl(corpus_path)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "the committed corpus must have unique chunk_ids"
