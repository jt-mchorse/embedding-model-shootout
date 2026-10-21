"""`run_sweep` refuses a single-use `corpus` or `queries` before embedding (#243).

`require_sequence` says a generator "is iterable and single-use: validating its
elements would consume it and leave the copy empty", and that `corpus` /
`queries` / `k_values` are "all `Sequence[...]` on purpose" -- but only `notes`
and `k_values` went through it. Measured on `main` bad0a92, with a fake
embedder that counts texts: a generator `queries` embedded both corpus texts
and then raised `TypeError: object of type 'generator' has no len()`; a
generator `corpus` raised "query 'q1' expects chunk_id 'a', which is not in the
corpus", because `_require_str_fields` had already consumed it.
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


_CORPUS = [CorpusChunk("a", "alpha beta gamma"), CorpusChunk("b", "delta epsilon zeta")]
_QUERIES = [Query("q1", "alpha beta", "a"), Query("q2", "delta epsilon", "b")]


@pytest.mark.parametrize(
    ("label", "corpus", "queries", "message"),
    [
        (
            "generator-queries",
            lambda: _CORPUS,
            lambda: (q for q in _QUERIES),
            r"^queries must be an ordered sequence of Query; got generator$",
        ),
        (
            "generator-corpus",
            lambda: (c for c in _CORPUS),
            lambda: _QUERIES,
            r"^corpus must be an ordered sequence of CorpusChunk; got generator$",
        ),
        (
            "set-queries",
            lambda: _CORPUS,
            lambda: set(_QUERIES),
            r"^queries must be an ordered sequence of Query; got set$",
        ),
    ],
)
def test_a_non_sequence_is_refused_before_anything_is_embedded(
    label: str, corpus, queries, message: str
) -> None:
    embedder = _Counting()
    with pytest.raises(ValueError, match=message):
        run_sweep(corpus(), queries(), embedder=embedder)
    assert embedder.texts == 0


@pytest.mark.parametrize("container", [list, tuple], ids=["list", "tuple"])
def test_a_list_or_tuple_still_sweeps(container) -> None:
    """GREEN against the unfixed tree: the controls that reject a rule
    demanding a `list` (this function's own `k_values` default is a tuple)."""
    embedder = _Counting()
    result = run_sweep(container(_CORPUS), container(_QUERIES), embedder=embedder)
    assert result.n_corpus == 2
    assert result.n_queries == 2
    assert embedder.texts == 4
