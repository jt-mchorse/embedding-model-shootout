"""Lock tests for #112: both embed seams enforce one vector per input text.

`run_sweep` checked the corpus call's row count inline and left the per-query
call nine lines below it — same duck-typed `Embedder` Protocol (D-004) —
unguarded. `_reject_non_finite_vectors`'s docstring even said the length check
"already lives" at the embedder-output seam; that was true of one of the two.

Both ways of breaking the contract mattered:

* **Too few.** `embedder.embed([q.text])[0]` on an empty list raised
  `IndexError`, which `cli`'s `sweep run` doesn't catch (it catches
  `ValueError`), so it escaped as a traceback at exit 1.
* **Too many.** `[0]` took the first row and asked no questions, producing a
  plausible and entirely wrong benchmark number with no error at all.

`test_extra_vector_scored_the_wrong_recall_before_the_guard` is the important
one: it anchors the guard to the *corruption* rather than to an exception type,
by demonstrating the wrong number the unguarded expression produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emb_shootout import CorpusChunk, run_sweep
from emb_shootout.cli import main
from emb_shootout.providers import PROVIDER_REGISTRY
from emb_shootout.sweep import Query

_CORPUS = [CorpusChunk("c1", "alpha document"), CorpusChunk("c2", "beta document")]
_QUERIES = [Query("q1", "alpha document", "c1")]


def _honest_vectors(texts) -> list[list[float]]:
    """Orthogonal unit vectors: 'alpha' → [1, 0], anything else → [0, 1]."""
    return [[1.0, 0.0] if "alpha" in t else [0.0, 1.0] for t in texts]


class _FakeEmbedder:
    dim = 2
    cost_per_million_tokens = 1.0

    def __init__(self, name: str) -> None:
        self.name = name

    def embed(self, texts):  # noqa: ANN001, ANN201 - Protocol shape
        return _honest_vectors(texts)


class _EmptyOnQuery(_FakeEmbedder):
    """Returns nothing for a single-text batch — what a truncated response looks like."""

    def embed(self, texts):  # noqa: ANN001, ANN201
        return [] if len(texts) == 1 else _honest_vectors(texts)


class _ExtraOnQuery(_FakeEmbedder):
    """Prepends a stale vector to a single-text batch, so `[0]` is the wrong one."""

    def embed(self, texts):  # noqa: ANN001, ANN201
        vecs = _honest_vectors(texts)
        return ([[0.0, 1.0]] + vecs) if len(texts) == 1 else vecs


class _ShortOnCorpus(_FakeEmbedder):
    """Drops the last row of a multi-text batch — the corpus seam's original case."""

    def embed(self, texts):  # noqa: ANN001, ANN201
        return _honest_vectors(texts)[:-1] if len(texts) > 1 else _honest_vectors(texts)


# --- the query seam, both directions ----------------------------------------


@pytest.mark.parametrize(
    ("embedder_cls", "returned"),
    [(_EmptyOnQuery, 0), (_ExtraOnQuery, 2)],
)
def test_query_seam_arity_mismatch_raises_value_error(embedder_cls, returned: int) -> None:
    with pytest.raises(ValueError, match=rf"returned {returned} vectors for 1 query text\b"):
        run_sweep(_CORPUS, _QUERIES, embedder=embedder_cls("fake/q"))


def test_query_seam_too_few_is_not_an_index_error() -> None:
    # The specific regression: `[0]` on an empty list. `IndexError` is not a
    # `ValueError`, so `sweep run`'s catch missed it entirely.
    with pytest.raises(ValueError, match="returned 0 vectors") as exc:
        run_sweep(_CORPUS, _QUERIES, embedder=_EmptyOnQuery("fake/q"))
    assert not isinstance(exc.value, IndexError)


def test_extra_vector_scored_the_wrong_recall_before_the_guard() -> None:
    """Anchor the guard to the corruption, not to the exception type.

    Pre-fix, `_ExtraOnQuery` scored recall@1 = 0.0 where the honest embedder
    scores 1.0 — a plausible, entirely wrong benchmark number with no error
    raised. Reproduce that outcome here by doing exactly what the unguarded
    line did, so the guard can't later be "fixed" by widening a catch while
    the wrong-number path quietly returns.
    """
    honest = run_sweep(_CORPUS, _QUERIES, embedder=_FakeEmbedder("fake/honest"))
    assert honest.recall_at_k[1] == 1.0

    lying = _ExtraOnQuery("fake/extra")
    corpus_vectors = lying.embed([c.text for c in _CORPUS])
    query_vector = lying.embed([_QUERIES[0].text])[0]  # the pre-fix expression
    from emb_shootout.sweep import retrieve_top_k

    top = retrieve_top_k(query_vector, corpus_vectors, [c.chunk_id for c in _CORPUS], 1)
    assert [cid for cid, _ in top] != [_QUERIES[0].expected_chunk_id]


# --- the corpus seam is unchanged -------------------------------------------


def test_corpus_seam_still_rejects_a_short_batch() -> None:
    with pytest.raises(ValueError, match=r"returned 1 vectors for 2 corpus texts\b"):
        run_sweep(_CORPUS, _QUERIES, embedder=_ShortOnCorpus("fake/c"))


def test_honest_embedder_is_undisturbed() -> None:
    result = run_sweep(_CORPUS, _QUERIES, embedder=_FakeEmbedder("fake/honest"))
    assert result.recall_at_k[1] == 1.0
    assert result.n_queries == 1
    assert result.n_corpus == 2


# --- end to end through the CLI ---------------------------------------------


def test_sweep_run_exits_two_on_an_arity_mismatch(tmp_path: Path, capsys, monkeypatch) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    # `build_queries` skips chunks under six words, and a corpus of short ones
    # exits 2 with "no corpus chunks have >= 6 words" — the same exit code this
    # test asserts, for an entirely unrelated reason. The stderr assertion below
    # pins the message so the test can't pass without reaching the sweep.
    corpus_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "chunk_id": f"c{i}",
                    "text": f"alpha document number {i} with several additional words here",
                }
            )
            for i in range(6)
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(PROVIDER_REGISTRY, "fake", lambda: _EmptyOnQuery("fake/q"))

    rc = main(
        [
            "sweep",
            "run",
            "--provider",
            "fake",
            "--corpus",
            str(corpus_path),
            "--queries",
            "2",
            "--output",
            str(tmp_path / "out.json"),
        ]
    )

    err = capsys.readouterr().err
    assert rc == 2
    assert err.startswith("error:")
    assert "vectors for 1 query text" in err, f"exited 2 for the wrong reason: {err!r}"
    assert "Traceback" not in err


# --- the shipped providers ---------------------------------------------------


def test_hash_provider_returns_one_vector_per_text() -> None:
    # The one provider with no optional dependency, so it can be exercised for
    # real. The SDK-backed four are covered by the seam guard rather than by a
    # network call — see the issue for why the check belongs at the seam.
    provider = PROVIDER_REGISTRY["hash"]()
    for texts in ([], ["one"], ["one", "two", "three"]):
        assert len(provider.embed(texts)) == len(texts)


# --- lock: a future embed seam can't skip the guard --------------------------


def test_every_embed_call_site_is_guarded() -> None:
    """One `_require_one_vector_per_text` per `embedder.embed(...)` call.

    Asserting "the corpus and query seams are guarded" would go stale the
    moment a third seam is added — which is how this gap appeared in the first
    place: the corpus call grew an inline check and the query call, nine lines
    below it, never did. Count the shape instead.
    """
    import ast

    import emb_shootout.sweep as sweep_module

    # Parse rather than grep: the guard's own docstring quotes the expression
    # it replaced, and a text scan counts that prose as a call site.
    tree = ast.parse(Path(sweep_module.__file__).read_text(encoding="utf-8"))
    embed_calls = 0
    guards = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "embed"
            and isinstance(func.value, ast.Name)
            and func.value.id == "embedder"
        ):
            embed_calls += 1
        elif isinstance(func, ast.Name) and func.id == "_require_one_vector_per_text":
            guards += 1

    assert embed_calls >= 2, "expected the corpus and query seams to still exist"
    assert guards == embed_calls, (
        f"{embed_calls} `embedder.embed(...)` call sites but {guards} arity guards — "
        "every embed seam needs `_require_one_vector_per_text` before it indexes the "
        "result, or #112 comes back as an IndexError or a silently wrong recall."
    )
