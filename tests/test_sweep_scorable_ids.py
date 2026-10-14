"""Lock tests for #210: `run_sweep` refuses an id set it cannot score honestly.

Scoring is id-equality, so two id populations decide what recall@k and
NDCG@10 mean. #117 refused a duplicate `chunk_id` in the CLI's corpus reader
and left `run_sweep` -- the public scorer -- open; a query whose answer chunk
is absent from the corpus was never checked anywhere. Both returned a valid
`SweepResult` with a wrong number.

`test_duplicate_id_inflated_recall_before_the_guard` anchors the first guard
to the corruption (0.5 -> 1.0) rather than to an exception type.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emb_shootout import CorpusChunk, run_sweep
from emb_shootout.cli import main
from emb_shootout.providers import HashEmbedderProvider
from emb_shootout.sweep import Query

# q2's answer is chunk "a", but its text matches chunk "b": an honest retriever
# misses it at k=1, so recall@1 is 0.5.
_QUERIES = [Query("q1", "alpha beta gamma", "a"), Query("q2", "delta epsilon zeta", "a")]
_UNIQUE = [CorpusChunk("a", "alpha beta gamma"), CorpusChunk("b", "delta epsilon zeta")]
_DUPLICATE = [CorpusChunk("a", "alpha beta gamma"), CorpusChunk("a", "delta epsilon zeta")]


class _CountingEmbedder(HashEmbedderProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def embed(self, texts):  # noqa: ANN001, ANN201 - Protocol shape
        self.calls += 1
        return super().embed(texts)


def test_unique_ids_baseline_is_half() -> None:
    # Without this the duplicate test below compares against nothing.
    result = run_sweep(_UNIQUE, _QUERIES, embedder=HashEmbedderProvider(), k_values=(1,))
    assert result.recall_at_k == {1: 0.5}


def test_duplicate_id_inflated_recall_before_the_guard() -> None:
    # Pre-fix this returned {1: 1.0}: the second "a" -- the wrong text -- was
    # credited as q2's answer.
    with pytest.raises(ValueError, match=r"duplicate chunk_id 'a' at corpus\[0\] and corpus\[1\]"):
        run_sweep(_DUPLICATE, _QUERIES, embedder=HashEmbedderProvider(), k_values=(1,))


def test_query_whose_answer_is_absent_is_refused() -> None:
    queries = [Query("q1", "alpha beta gamma", "a"), Query("q2", "delta epsilon zeta", "missing")]
    with pytest.raises(ValueError, match=r"query 'q2' expects chunk_id 'missing'"):
        run_sweep(_UNIQUE, queries, embedder=HashEmbedderProvider(), k_values=(1,))


@pytest.mark.parametrize(
    ("corpus", "queries"),
    [
        pytest.param(_DUPLICATE, _QUERIES, id="duplicate-id"),
        pytest.param(_UNIQUE, [Query("q1", "alpha", "missing")], id="absent-answer"),
    ],
)
def test_refused_before_any_embed_call(corpus, queries) -> None:  # noqa: ANN001
    embedder = _CountingEmbedder()
    with pytest.raises(ValueError, match="duplicate chunk_id|not in the corpus"):
        run_sweep(corpus, queries, embedder=embedder, k_values=(1,))
    assert embedder.calls == 0, "a paid provider would have been billed for an unscorable sweep"


def test_sweep_run_maps_an_absent_answer_to_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "\n".join(json.dumps({"chunk_id": c.chunk_id, "text": c.text}) for c in _UNIQUE) + "\n",
        encoding="utf-8",
    )
    # `build_queries` always draws answers from the corpus, so stand in a query
    # set that does not -- what a BYO query builder would hand `run_sweep`.
    monkeypatch.setattr(
        "emb_shootout.queries.build_queries",
        lambda corpus, n, seed: [Query("q1", "alpha", "missing")],  # noqa: ARG005
    )
    out = tmp_path / "out.json"

    rc = main(
        ["sweep", "run", "--provider", "hash", "--corpus", str(corpus_path), "--output", str(out)]
    )

    err = capsys.readouterr().err
    assert rc == 2
    assert err.startswith("error:")
    assert "'missing'" in err
    assert not out.exists()
