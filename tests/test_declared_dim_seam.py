"""`embedder_dim` must describe the vectors, not just the declaration (#119).

`SweepResult.embedder_dim` was copied straight off `embedder.dim` and never
reconciled with what the embedder actually returned, so a provider that
declares one dimension and returns another produced a clean, successful sweep
whose recorded dimension was fiction. Measured pre-fix:

    declared == actual (honest)  -> OK  embedder_dim RECORDED = 8     ndcg=0.66237898...
    declared 1024, actual 8      -> OK  embedder_dim RECORDED = 1024  ndcg=0.66237898...
    declared 8, actual 1024      -> OK  embedder_dim RECORDED = 8     ndcg=0.71030991...

No error, no warning, exit 0. The third row's NDCG *differs* from the first,
which is the point: the quality number genuinely depends on the real dimension
while the reported dimension is the declared one, so `results/*.json` paired a
real measured score with a fabricated dim — and dim is a first-class axis in
the README table, the JSON payload, and the Pareto plot.

The `Embedder` Protocol docstring is one sentence with two clauses: "Return one
float vector per input text. All vectors share `self.dim`." #112 enforced the
first; this enforces the second, at the same seam.

Assertions are anchored to that pre-fix outcome — sweep *succeeded* with a
fictional dim — rather than merely to the post-fix exception type.
"""

from __future__ import annotations

import pytest

from emb_shootout.corpus import Chunk
from emb_shootout.queries import Query
from emb_shootout.sweep import run_sweep


class _StubEmbedder:
    """Declares `dim`; returns vectors of `actual` length."""

    def __init__(self, declared: int, actual: int, name: str = "stub") -> None:
        self.dim = declared
        self.actual = actual
        self.name = name
        self.cost_per_million_tokens = 0.0

    def embed(self, texts):  # noqa: ANN001, ANN201 — duck-typed Protocol
        return [[0.1 + (i % 7) * 0.01] * self.actual for i, _ in enumerate(texts)]


def _corpus(n: int = 6) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"c{i}",
            text=f"document number {i} about postgres tuning",
            module="m",
            qualname="q",
            kind="function",
            source="s",
        )
        for i in range(n)
    ]


def _queries(n: int = 3) -> list[Query]:
    return [
        Query(query_id=f"q{i}", text=f"document number {i}", expected_chunk_id=f"c{i}")
        for i in range(n)
    ]


def test_honest_embedder_is_unaffected() -> None:
    # The guard must not cost anything for a provider telling the truth. This
    # is the row that stays green on both trees.
    result = run_sweep(embedder=_StubEmbedder(8, 8), corpus=_corpus(), queries=_queries())
    assert result.embedder_dim == 8
    assert 0.0 <= result.ndcg_at_10 <= 1.0


def test_declared_larger_than_actual_is_rejected() -> None:
    # Pre-fix: succeeded, recording embedder_dim=1024 for 8-component vectors.
    with pytest.raises(ValueError, match=r"declares dim=1024 but returned a 8-component"):
        run_sweep(embedder=_StubEmbedder(1024, 8), corpus=_corpus(), queries=_queries())


def test_declared_smaller_than_actual_is_rejected() -> None:
    # Pre-fix: succeeded, recording embedder_dim=8 for 1024-component vectors —
    # and with a DIFFERENT ndcg than the honest run, so the fabricated dim sat
    # next to a real, dimension-dependent quality score.
    with pytest.raises(ValueError, match=r"declares dim=8 but returned a 1024-component"):
        run_sweep(embedder=_StubEmbedder(8, 1024), corpus=_corpus(), queries=_queries())


def test_error_names_the_embedder_and_the_seam() -> None:
    # The diagnostic must point at the provider, not at `cosine`. Pre-fix, the
    # only way this class surfaced at all was a mismatch *between* two vectors,
    # raising "vector length mismatch: 768 vs 512" from inside the retrieval
    # loop, naming neither the embedder nor the seam.
    with pytest.raises(ValueError, match=r"declares dim=512") as exc:
        run_sweep(
            embedder=_StubEmbedder(512, 4, name="acme/embed-v9"),
            corpus=_corpus(),
            queries=_queries(),
        )
    msg = str(exc.value)
    assert "acme/embed-v9" in msg
    assert "corpus" in msg or "query" in msg
    assert "512" in msg
    assert "4-component" in msg


def test_the_query_seam_is_covered_too() -> None:
    # #112 established that the batch corpus call and the per-query call are
    # two separate expressions of one contract, and that guarding only one is
    # how the arity gap survived. An embedder honest on the corpus pass and
    # wrong on the query pass must still be caught.
    class _HonestThenWrong(_StubEmbedder):
        def __init__(self) -> None:
            super().__init__(8, 8, name="flaky")
            self._calls = 0

        def embed(self, texts):  # noqa: ANN001, ANN201
            self._calls += 1
            width = 8 if self._calls == 1 else 5
            return [[0.1] * width for _ in texts]

    with pytest.raises(ValueError, match=r"declares dim=8 but returned a 5-component query vector"):
        run_sweep(embedder=_HonestThenWrong(), corpus=_corpus(), queries=_queries())


def test_cli_exit_code_contract_still_holds_for_this_class() -> None:
    # `cli sweep run` catches ValueError and exits 2. Raising a ValueError (not
    # a bespoke exception type) is what keeps this class inside the documented
    # exit-code contract rather than escaping as a traceback at exit 1.
    with pytest.raises(ValueError, match=r"declares dim=1024"):
        run_sweep(embedder=_StubEmbedder(1024, 8), corpus=_corpus(), queries=_queries())
