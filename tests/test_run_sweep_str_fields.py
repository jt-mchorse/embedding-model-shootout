"""`run_sweep` refuses a non-`str` id or text before embedding anything (#236).

#212 skipped a non-`str` value in its surrogate check as "a type error ...
which mypy reports". Measured on `main` with a counting wrapper around the hash
provider (2 corpus chunks, 1 query): mixed `int`/`str` chunk ids embedded all 3
texts and then raised a raw `TypeError` in the fingerprint's `sorted`; a
`bytes` query text reached the per-query embed after the corpus had been
embedded; an `int` query id or an all-`int` id set returned a `SweepResult`.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from emb_shootout import CorpusChunk, run_sweep
from emb_shootout.providers import HashEmbedderProvider
from emb_shootout.sweep import Query


class _Counting:
    """An `Embedder` that counts the texts it is asked to embed."""

    def __init__(self) -> None:
        self._inner = HashEmbedderProvider()
        self.name = "counting"
        self.dim = self._inner.dim
        self.cost_per_million_tokens = 0.0
        self.texts = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts += len(texts)
        return self._inner.embed(texts)


_A = "alpha beta gamma"
_B = "delta epsilon zeta"


def _corpus(**a: object) -> list[CorpusChunk]:
    first = CorpusChunk(a.get("chunk_id", "a"), a.get("text", _A))  # type: ignore[arg-type]
    return [first, CorpusChunk("b", _B)]


_CASES = [
    # (id, corpus, queries, expected message)
    (
        "corpus-chunk_id-int-mixed",
        _corpus(chunk_id=1),
        [Query("q1", "alpha beta", 1)],  # type: ignore[arg-type]
        r"corpus\[0\]\.chunk_id must be a string; got int 1",
    ),
    (
        "corpus-chunk_id-all-int",
        [CorpusChunk(1, _A), CorpusChunk(2, _B)],  # type: ignore[arg-type]
        [Query("q1", "alpha beta", 1)],  # type: ignore[arg-type]
        r"corpus\[0\]\.chunk_id must be a string; got int 1",
    ),
    (
        "corpus-chunk_id-unhashable",
        _corpus(chunk_id=["a"]),
        [Query("q1", "alpha beta", "a")],
        r"corpus\[0\]\.chunk_id must be a string; got list",
    ),
    (
        "corpus-text-bytes",
        _corpus(text=_A.encode()),
        [Query("q1", "alpha beta", "a")],
        r"corpus\[0\]\.text must be a string; got bytes",
    ),
    (
        "corpus-text-none",
        _corpus(text=None),
        [Query("q1", "alpha beta", "a")],
        r"corpus\[0\]\.text must be a string; got NoneType",
    ),
    (
        "query-query_id-int",
        _corpus(),
        [Query(1, "alpha beta", "a")],  # type: ignore[arg-type]
        r"queries\[0\]\.query_id must be a string; got int 1",
    ),
    (
        "query-text-bytes",
        _corpus(),
        [Query("q1", b"alpha beta", "a")],  # type: ignore[arg-type]
        r"queries\[0\]\.text must be a string; got bytes",
    ),
    (
        "query-expected_chunk_id-int",
        _corpus(),
        [Query("q1", "alpha beta", 0)],  # type: ignore[arg-type]
        r"queries\[0\]\.expected_chunk_id must be a string; got int 0",
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


def test_clean_inputs_still_sweep() -> None:
    """Control: green before and after the fix."""
    embedder = _Counting()
    result = run_sweep(
        _corpus(), [Query("q1", "alpha beta", "a")], embedder=embedder, k_values=(1,)
    )
    assert embedder.texts == 3
    assert result.n_corpus == 2
