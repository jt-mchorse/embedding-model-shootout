"""A zero vector is the absence of a measurement, not a bad one (#131).

`run_sweep` guards three things `Embedder.embed` can return that would corrupt a
published score, all at the same seam and all applied to corpus and query vectors
alike: the wrong number of vectors (#112), a vector of the wrong length, and a
non-finite component (#125). A vector of **all zeros** was the fourth, and the
one that cannot be caught downstream -- `cosine` deliberately maps a zero norm to
`0.0`, and `retrieve_top_k` names that as a dependency.

Measured before this change, a 6-chunk corpus with one query per chunk and an
embedder returning all zeros::

    recall@1 = 0.167   recall@3 = 0.500   recall@5 = 0.833   ndcg@10 = 0.551

Exactly `k/N`, and reported as a measurement. Every similarity ties at `0.0`, so
the ranking falls entirely to the `chunk_id` tiebreak -- the same order for every
query -- and query *i* hits iff `i < k`. Partial corruption reads as an ordinary
result: one zeroed chunk out of six gives `recall@1 = 0.833`.

The reproducibility is the sharp part. `#73` added that tiebreak so ranking is a
pure function of `(similarity, chunk_id)`, which is right. But when every
similarity is equal the tiebreak stops breaking ties and becomes the entire
ranking, so a score that would once have jumped around under corpus reordering is
now stably, reproducibly meaningless -- and stability is what a reader takes as
evidence of a real measurement.
"""

from __future__ import annotations

import random

import pytest

from emb_shootout.sweep import (
    CorpusChunk,
    Query,
    cosine,
    retrieve_top_k,
    run_sweep,
)

DIM = 8


class ZeroEmbedder:
    """Returns no information whatsoever."""

    name = "all-zero"
    dim = DIM
    cost_per_million_tokens = 1.0

    def embed(self, texts):
        return [[0.0] * DIM for _ in texts]


class PartialZeroEmbedder:
    """Zeros only the texts it is told to; everything else gets a real vector."""

    name = "partial-zero"
    dim = DIM
    cost_per_million_tokens = 1.0

    def __init__(self, zero_texts: set[str]) -> None:
        self._zero = zero_texts

    def embed(self, texts):
        out = []
        for t in texts:
            if t in self._zero:
                out.append([0.0] * DIM)
            else:
                out.append(_signal(t))
        return out


class GoodEmbedder:
    name = "good"
    dim = DIM
    cost_per_million_tokens = 1.0

    def embed(self, texts):
        return [_signal(t) for t in texts]


def _signal(text: str) -> list[float]:
    """A deterministic non-zero vector, distinct per text."""
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(text))
    rng = random.Random(seed)
    v = [rng.uniform(0.1, 1.0) for _ in range(DIM)]
    assert any(v)
    return v


def _fixture(n: int = 6) -> tuple[list[CorpusChunk], list[Query]]:
    corpus = [CorpusChunk(chunk_id=f"c{i}", text=f"document number {i}") for i in range(n)]
    queries = [
        Query(query_id=f"q{i}", text=f"document number {i}", expected_chunk_id=f"c{i}")
        for i in range(n)
    ]
    return corpus, queries


# --- the premise: why nothing downstream can see it -------------------------


def test_cosine_scores_a_zero_vector_zero_against_everything() -> None:
    """Not a bug in `cosine` -- a deliberate choice `retrieve_top_k` depends on
    for finiteness. It is also why the corruption has to be caught at the seam
    that produces the vector, not the one that consumes it."""
    zero = [0.0] * DIM
    assert cosine(zero, _signal("anything")) == 0.0
    assert cosine(zero, zero) == 0.0


def test_a_zero_vector_ties_with_every_other_and_the_tiebreak_decides() -> None:
    """With all similarities equal, `#73`'s `chunk_id` tiebreak is no longer a
    tiebreak -- it is the whole ranking."""
    corpus_vecs = [[0.0] * DIM for _ in range(4)]
    chunk_ids = [f"c{i}" for i in range(4)]
    top = retrieve_top_k(_signal("q"), corpus_vecs, chunk_ids, 4)
    assert [cid for cid, _score in top] == ["c0", "c1", "c2", "c3"]
    assert {score for _cid, score in top} == {0.0}


# --- the guard ---------------------------------------------------------------


def test_an_all_zero_embedder_can_no_longer_produce_a_result() -> None:
    corpus, queries = _fixture()
    with pytest.raises(ValueError, match="all-zero corpus vector"):
        run_sweep(corpus, queries, embedder=ZeroEmbedder(), k_values=(1, 5))


def test_one_zeroed_corpus_chunk_is_rejected() -> None:
    """The case that read as an ordinary slightly-worse embedder:
    `recall@1 = 0.833`."""
    corpus, queries = _fixture()
    emb = PartialZeroEmbedder({corpus[3].text})
    with pytest.raises(ValueError, match="all-zero corpus vector at index 3"):
        run_sweep(corpus, queries, embedder=emb, k_values=(1, 5))


def test_a_zeroed_query_is_rejected_and_named_as_a_query() -> None:
    """Both call sites, and the message distinguishes them -- an operator needs
    to know which side of the sweep produced it."""
    corpus, queries = _fixture()

    class ZeroQueryOnly:
        name = "zero-query"
        dim = DIM
        cost_per_million_tokens = 1.0

        def __init__(self) -> None:
            self._corpus_done = False

        def embed(self, texts):
            # The corpus is embedded in one batch first; queries follow.
            if not self._corpus_done:
                self._corpus_done = True
                return [_signal(t) for t in texts]
            return [[0.0] * DIM for _ in texts]

    with pytest.raises(ValueError, match="all-zero query vector"):
        run_sweep(corpus, queries, embedder=ZeroQueryOnly(), k_values=(1, 5))


def test_the_message_names_the_embedder_and_the_harm() -> None:
    corpus, queries = _fixture()
    with pytest.raises(ValueError, match="all-zero") as exc:
        run_sweep(corpus, queries, embedder=ZeroEmbedder(), k_values=(1, 5))
    message = str(exc.value)
    assert "all-zero" in message  # the embedder's name
    assert "measures nothing" in message
    assert "k/N" in message


# --- what must NOT be rejected ----------------------------------------------


def test_an_ordinary_embedder_is_unaffected() -> None:
    corpus, queries = _fixture()
    result = run_sweep(corpus, queries, embedder=GoodEmbedder(), k_values=(1, 5))
    assert result.recall_at_k[5] > 0.0


def test_a_zero_component_is_not_a_zero_vector() -> None:
    """A single zero component is a legitimate embedding value. The rule is
    exactly zero-*norm*, and scoping it to the component would reject real
    measurements -- most obviously any sparse-ish embedding."""

    class OneZeroComponent:
        name = "one-zero-component"
        dim = DIM
        cost_per_million_tokens = 1.0

        def embed(self, texts):
            out = []
            for t in texts:
                v = _signal(t)
                v[0] = 0.0
                out.append(v)
            return out

    corpus, queries = _fixture()
    result = run_sweep(corpus, queries, embedder=OneZeroComponent(), k_values=(1, 5))
    assert result.recall_at_k[5] > 0.0


def test_a_tiny_but_nonzero_vector_is_still_a_measurement() -> None:
    """Exactly zero-norm, not "small norm". A threshold would be a number nobody
    measured, which is what this repo refuses to publish."""

    class TinyEmbedder:
        name = "tiny"
        dim = DIM
        cost_per_million_tokens = 1.0

        def embed(self, texts):
            return [[c * 1e-300 for c in _signal(t)] for t in texts]

    corpus, queries = _fixture()
    result = run_sweep(corpus, queries, embedder=TinyEmbedder(), k_values=(1, 5))
    assert result.recall_at_k[5] > 0.0


# --- structural: four guards, two call sites --------------------------------


def test_every_embed_seam_guard_runs_on_both_corpus_and_query_vectors() -> None:
    """The four guards are a set, and a future fifth -- or a third call site --
    must not pick up three of four. Source check because the guards are
    module-private helpers with no runtime registry."""
    import inspect

    from emb_shootout import sweep

    src = inspect.getsource(sweep.run_sweep)
    guards = [
        "_require_one_vector_per_text",
        "_require_declared_dim",
        "_reject_non_finite_vectors",
        "_reject_zero_vectors",
    ]
    assert 'kind="corpus"' in src
    assert 'kind="query"' in src
    # Each guard must appear for both kinds somewhere in the function.
    for guard in guards:
        calls = src.count(f"{guard}(")
        assert calls >= 2, f"{guard} is called {calls} time(s); expected corpus and query"
