"""A result records its query set's identity, not only its size (#156, D-015).

#155 (D-014) made the aggregators refuse rows whose `n_queries` or `n_corpus`
disagree, and said what that left open: the seed was not recorded, so two rows
run with equal counts and different `--seed`s still shared a table. Measured
through the CLI on the committed corpus, before this change:

    sweep run --provider hash --queries 50 --seed 42   recall@5 0.520  NDCG@10 0.449
    sweep run --provider hash --queries 50 --seed 7    recall@5 0.620  NDCG@10 0.542
    sweep aggregate                                    exit 0, both rows in one table

The same embedder, ten points apart, published side by side under "apples-to-
apples by construction". A result now records `corpus_fingerprint` and
`query_fingerprint` -- computed by `run_sweep` from what it scored -- and the
seed as provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from emb_shootout.cli import main
from emb_shootout.providers.hash_embedder import HashEmbedderProvider
from emb_shootout.queries import build_queries
from emb_shootout.sweep import (
    CorpusChunk,
    Query,
    SweepResult,
    aggregate_json,
    aggregate_markdown,
    fingerprint_corpus,
    fingerprint_queries,
    require_comparable,
    run_sweep,
)

_ROOT = Path(__file__).resolve().parents[1]
_FP_A = "a" * 64
_FP_B = "b" * 64


def _corpus(n: int = 40) -> list[CorpusChunk]:
    return [
        CorpusChunk(f"mod.f{i}", f"function number {i} handles case {i % 7} of the parser")
        for i in range(n)
    ]


def _result(name: str, **overrides: Any) -> SweepResult:
    kwargs: dict[str, Any] = {
        "embedder_name": name,
        "embedder_dim": 8,
        "cost_per_million_tokens": 0.0,
        "n_corpus": 40,
        "n_queries": 10,
        "recall_at_k": {1: 0.5, 5: 0.6, 10: 0.7},
        "ndcg_at_10": 0.5,
        "embed_latency_ms": {"corpus_total": 1.0, "query_p50": 0.1, "query_p95": 0.2},
        "notes": [],
    }
    kwargs.update(overrides)
    return SweepResult(**kwargs)


def _write_corpus(path: Path, corpus: list[CorpusChunk]) -> None:
    path.write_text(
        "".join(json.dumps({"chunk_id": c.chunk_id, "text": c.text}) + "\n" for c in corpus),
        encoding="utf-8",
    )


# ----------------------------------------------------------------------
# The issue's scenario, through the CLI
# ----------------------------------------------------------------------


def _sweep(tmp_path: Path, corpus_path: Path, seed: int, name: str) -> Path:
    out = tmp_path / "results" / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    rc = main(
        [
            "sweep",
            "run",
            "--provider",
            "hash",
            "--corpus",
            str(corpus_path),
            "--queries",
            "10",
            "--seed",
            str(seed),
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    # Two runs of one provider: rename so the table has two distinct rows,
    # which is what two providers would produce.
    data = json.loads(out.read_text(encoding="utf-8"))
    data["embedder_name"] = name
    out.write_text(json.dumps(data), encoding="utf-8")
    return out


def test_equal_counts_with_different_seeds_are_refused_at_aggregate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    _write_corpus(corpus_path, _corpus())
    _sweep(tmp_path, corpus_path, 42, "seed42")
    _sweep(tmp_path, corpus_path, 7, "seed7")
    _ = capsys.readouterr()
    out = tmp_path / "benchmarks.md"
    out.write_text("untouched\n", encoding="utf-8")
    rc = main(["sweep", "aggregate", "--results-dir", str(tmp_path / "results"), "--out", str(out)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "different query sets" in err
    assert "query_fingerprint=" in err
    # The seeds are named, so the operator knows which run to repeat.
    assert "seed42 (seed 42)" in err
    assert "seed7 (seed 7)" in err
    assert out.read_text(encoding="utf-8") == "untouched\n"


def test_the_same_seed_twice_still_aggregates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    _write_corpus(corpus_path, _corpus())
    _sweep(tmp_path, corpus_path, 42, "first")
    _sweep(tmp_path, corpus_path, 42, "second")
    out = tmp_path / "benchmarks.md"
    rc = main(["sweep", "aggregate", "--results-dir", str(tmp_path / "results"), "--out", str(out)])
    _ = capsys.readouterr()
    assert rc == 0


def test_sweep_run_records_the_seed_and_both_fingerprints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus()
    corpus_path = tmp_path / "corpus.jsonl"
    _write_corpus(corpus_path, corpus)
    data = json.loads(_sweep(tmp_path, corpus_path, 42, "hash").read_text(encoding="utf-8"))
    _ = capsys.readouterr()
    assert data["query_seed"] == 42
    assert data["corpus_fingerprint"] == fingerprint_corpus(corpus)
    assert data["query_fingerprint"] == fingerprint_queries(build_queries(corpus, n=10, seed=42))


# ----------------------------------------------------------------------
# What a fingerprint names
# ----------------------------------------------------------------------


def test_a_corpus_of_the_same_size_with_different_text_is_a_different_corpus() -> None:
    """The #115 case in miniature: two builds agreeing on size, not on content."""
    corpus = _corpus()
    edited = [*corpus[:-1], CorpusChunk(corpus[-1].chunk_id, corpus[-1].text + " (3.14)")]
    assert len(edited) == len(corpus)
    assert fingerprint_corpus(edited) != fingerprint_corpus(corpus)


def test_order_does_not_change_a_fingerprint() -> None:
    """Retrieval breaks cosine ties on `chunk_id`, never on position, so a
    reordered corpus scores identically and must not read as a different set."""
    corpus = _corpus()
    assert fingerprint_corpus(list(reversed(corpus))) == fingerprint_corpus(corpus)
    queries = build_queries(corpus, n=10, seed=42)
    assert fingerprint_queries(list(reversed(queries))) == fingerprint_queries(queries)


def test_a_query_fingerprint_covers_every_field_of_a_query() -> None:
    q = Query("q1", "what parses case 3", "mod.f3")
    base = fingerprint_queries([q])
    assert fingerprint_queries([Query("q2", q.text, q.expected_chunk_id)]) != base
    assert fingerprint_queries([Query(q.query_id, "other", q.expected_chunk_id)]) != base
    assert fingerprint_queries([Query(q.query_id, q.text, "mod.f4")]) != base


def test_no_text_can_forge_a_field_boundary() -> None:
    """JSON-encoded, not delimiter-joined: moving a boundary changes the hash."""
    a = [CorpusChunk("a", "b c")]
    b = [CorpusChunk("a b", "c")]
    assert fingerprint_corpus(a) != fingerprint_corpus(b)


def test_run_sweep_fingerprints_what_it_scored() -> None:
    corpus = _corpus()
    queries = build_queries(corpus, n=10, seed=3)
    result = run_sweep(corpus, queries, embedder=HashEmbedderProvider(dim=16))
    assert result.corpus_fingerprint == fingerprint_corpus(corpus)
    assert result.query_fingerprint == fingerprint_queries(queries)
    # The seed is the caller's to give; `run_sweep` never saw it.
    assert result.query_seed is None


# ----------------------------------------------------------------------
# require_comparable: compared among the rows that carry it
# ----------------------------------------------------------------------


@pytest.mark.parametrize("aggregate", [aggregate_markdown, aggregate_json])
@pytest.mark.parametrize("field", ["corpus_fingerprint", "query_fingerprint"])
def test_rows_disagreeing_on_a_fingerprint_are_refused(aggregate: Any, field: str) -> None:
    with pytest.raises(ValueError, match="different query sets") as excinfo:
        aggregate([_result("hash", **{field: _FP_A}), _result("openai", **{field: _FP_B})])
    assert f"{field}=aaaaaaaaaaaa: hash" in str(excinfo.value)
    assert f"{field}=bbbbbbbbbbbb: openai" in str(excinfo.value)


def test_a_row_without_fingerprints_is_held_to_the_counts_only() -> None:
    """A result written before #156 -- the committed baseline is one -- compares
    by counts against anything, and fingerprinted rows still compare among
    themselves."""
    legacy = _result("legacy")
    require_comparable([legacy, _result("a", query_fingerprint=_FP_A)])
    with pytest.raises(ValueError, match="different query sets"):
        require_comparable(
            [legacy, _result("a", query_fingerprint=_FP_A), _result("b", query_fingerprint=_FP_B)]
        )


def test_the_seed_alone_is_not_compared() -> None:
    """Provenance, not identity: equal fingerprints with different recorded
    seeds are the same query set."""
    require_comparable(
        [
            _result("a", query_seed=1, query_fingerprint=_FP_A),
            _result("b", query_seed=2, query_fingerprint=_FP_A),
        ]
    )


def test_counts_are_still_checked_first() -> None:
    with pytest.raises(ValueError, match="n_queries=10") as excinfo:
        require_comparable(
            [
                _result("a", query_fingerprint=_FP_A),
                _result("b", n_queries=11, query_fingerprint=_FP_A),
            ]
        )
    assert "fingerprint" not in str(excinfo.value)


# ----------------------------------------------------------------------
# The field contract
# ----------------------------------------------------------------------


def test_the_three_fields_round_trip() -> None:
    r = _result("hash", query_seed=42, corpus_fingerprint=_FP_A, query_fingerprint=_FP_B)
    d = r.to_dict()
    assert (d["query_seed"], d["corpus_fingerprint"], d["query_fingerprint"]) == (42, _FP_A, _FP_B)
    assert SweepResult.from_dict(json.loads(json.dumps(d))) == r


def test_an_unrecorded_field_is_omitted_so_old_files_round_trip_to_the_same_bytes() -> None:
    """Written only when recorded, so the committed baseline -- which predates
    #156 -- reloads and re-serialises without gaining three nulls."""
    path = _ROOT / "results" / "hash.json"
    raw = path.read_text(encoding="utf-8")
    loaded = SweepResult.from_dict(json.loads(raw))
    assert (loaded.query_seed, loaded.corpus_fingerprint, loaded.query_fingerprint) == (
        None,
        None,
        None,
    )
    assert json.dumps(loaded.to_dict(), indent=2, sort_keys=True) + "\n" == raw


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query_seed", True),
        ("query_seed", 42.0),
        ("query_seed", "42"),
        ("corpus_fingerprint", "A" * 64),
        ("corpus_fingerprint", "a" * 63),
        ("corpus_fingerprint", "a" * 64 + "\n"),
        ("query_fingerprint", 7),
        ("query_fingerprint", ""),
    ],
)
def test_a_malformed_identity_field_is_refused_on_both_paths(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=field) as built:
        _result("hash", **{field: value})
    d = _result("hash").to_dict()
    d[field] = value
    with pytest.raises(ValueError, match=field) as loaded:
        SweepResult.from_dict(d)
    assert str(built.value) == str(loaded.value)


def test_an_explicit_null_reads_as_unrecorded() -> None:
    d = _result("hash").to_dict()
    d.update(query_seed=None, corpus_fingerprint=None, query_fingerprint=None)
    assert SweepResult.from_dict(d) == _result("hash")
