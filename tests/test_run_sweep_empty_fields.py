"""`run_sweep` refuses an empty id or text before embedding anything (#238).

#236 refused a non-`str` field and left an empty one through. Both file
readers refuse it (`cli._read_corpus_jsonl` and `validate._validate_row`'s
``empty_chunk_id`` / ``empty_text``). Measured on `main` 1d9062b: an empty
query "retrieved" an empty chunk at recall 1.0, and an empty chunk id and
query id scored 1.0 with no error. With a fake embedder that refuses ``""``
the way OpenAI does, 1000 chunks and 50 queries with the last one empty billed
1049 texts and then failed with a `RuntimeError`.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from emb_shootout import CorpusChunk, run_sweep
from emb_shootout.providers import HashEmbedderProvider
from emb_shootout.sweep import Query


class _Counting:
    """An `Embedder` that counts the texts it is asked to embed.

    With ``refuse_empty`` it raises on ``""`` the way a paid provider does.
    """

    def __init__(self, *, refuse_empty: bool = False) -> None:
        self._inner = HashEmbedderProvider()
        self.name = "counting"
        self.dim = self._inner.dim
        self.cost_per_million_tokens = 0.0
        self.refuse_empty = refuse_empty
        self.texts = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self.refuse_empty and "" in texts:
            raise RuntimeError("provider refused an empty input")
        self.texts += len(texts)
        return self._inner.embed(texts)


_A = "alpha beta gamma"
_B = "delta epsilon zeta"


_CASES = [
    # (id, corpus, queries, expected message)
    (
        "corpus-chunk_id",
        [CorpusChunk("", _A), CorpusChunk("b", _B)],
        [Query("q1", "alpha beta", "")],
        r"^corpus\[0\]\.chunk_id must not be empty$",
    ),
    (
        "corpus-text",
        [CorpusChunk("a", _A), CorpusChunk("b", "")],
        [Query("q1", "alpha beta", "a")],
        r"^corpus\[1\]\.text must not be empty$",
    ),
    (
        "query-query_id",
        [CorpusChunk("a", _A), CorpusChunk("b", _B)],
        [Query("", "alpha beta", "a")],
        r"^queries\[0\]\.query_id must not be empty$",
    ),
    (
        "query-text",
        [CorpusChunk("a", _A), CorpusChunk("b", _B)],
        [Query("q1", "alpha beta", "a"), Query("q2", "", "b")],
        r"^queries\[1\]\.text must not be empty$",
    ),
    (
        # A non-empty corpus cannot contain "", so before the fix this was
        # already refused, by `_require_scorable_ids`, as "not in the corpus".
        # It is listed so the empty rule names the field, as the readers do.
        "query-expected_chunk_id",
        [CorpusChunk("a", _A), CorpusChunk("b", _B)],
        [Query("q1", "alpha beta", "")],
        r"^queries\[0\]\.expected_chunk_id must not be empty$",
    ),
]


@pytest.mark.parametrize(
    ("corpus", "queries", "message"),
    [c[1:] for c in _CASES],
    ids=[c[0] for c in _CASES],
)
def test_refused_by_name_before_any_embed(
    corpus: list[CorpusChunk], queries: list[Query], message: str
) -> None:
    embedder = _Counting()
    with pytest.raises(ValueError, match=message):
        run_sweep(corpus, queries, embedder=embedder, k_values=(1,))
    assert embedder.texts == 0


def test_empty_query_cannot_retrieve_an_empty_chunk() -> None:
    """The issue's first repro: recall 1.0 for an empty query on `main`."""
    corpus = [
        CorpusChunk("json.dumps", "json.dumps serialize obj to a JSON formatted str"),
        CorpusChunk("json.loads", "json.loads deserialize s to a Python object"),
        CorpusChunk("blank", ""),
    ]
    with pytest.raises(ValueError, match=r"corpus\[2\]\.text must not be empty"):
        run_sweep(corpus, [Query("q0", "", "blank")], embedder=HashEmbedderProvider())


def test_last_empty_query_bills_nothing() -> None:
    """The issue's billing repro: the corpus was embedded before the provider failed."""
    corpus = [CorpusChunk(f"c{i}", f"chunk number {i} text") for i in range(1000)]
    queries = [Query(f"q{i}", f"chunk number {i}", f"c{i}") for i in range(49)]
    queries.append(Query("q49", "", "c49"))
    embedder = _Counting(refuse_empty=True)
    with pytest.raises(ValueError, match=r"queries\[49\]\.text must not be empty"):
        run_sweep(corpus, queries, embedder=embedder, k_values=(1,))
    assert embedder.texts == 0


def test_non_empty_inputs_still_sweep() -> None:
    """Control: green before and after the fix, including a whitespace-only id.

    Neither reader refuses whitespace, so neither does `run_sweep`.
    """
    embedder = _Counting(refuse_empty=True)
    result = run_sweep(
        [CorpusChunk(" ", _A), CorpusChunk("b", _B)],
        [Query("q1", "alpha beta", " ")],
        embedder=embedder,
        k_values=(1,),
    )
    assert embedder.texts == 3
    assert result.recall_at_k == {1: 1.0}
