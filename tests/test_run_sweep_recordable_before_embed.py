"""`run_sweep` refuses an unrecordable `notes`, `query_seed` or embedder
attribute before embedding anything (#218).

#210 and #212 made `run_sweep` the pre-embed choke point for the corpus and
query inputs. Its own `notes` / `query_seed` and the embedder's recorded
`name` / `dim` / `cost_per_million_tokens` were still checked only by the
`SweepResult(...)` in its `return`. Measured on `main`: every one of these was
refused, correctly, after all 3 texts (2 corpus + 1 query) had been embedded.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from emb_shootout import CorpusChunk, run_sweep
from emb_shootout.providers import HashEmbedderProvider
from emb_shootout.sweep import Query


class _Counting:
    """A minimal `Embedder` Protocol implementer, with no constructor-side
    validation of its own, that counts the texts it is asked to embed."""

    def __init__(self, *, name: object = "counting", dim: object = None, cost: object = 0.0):
        self._inner = HashEmbedderProvider()
        self.name = name
        self.dim = self._inner.dim if dim is None else dim
        self.cost_per_million_tokens = cost
        self.texts = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts += len(texts)
        return self._inner.embed(texts)


def _inputs() -> tuple[list[CorpusChunk], list[Query]]:
    corpus = [CorpusChunk("a", "alpha beta gamma"), CorpusChunk("b", "delta epsilon zeta")]
    return corpus, [Query(query_id="q1", text="alpha beta", expected_chunk_id="a")]


_SWEEP_ARG_CASES = [
    ("notes-str", {"notes": "recall looks low"}, r"notes must be a sequence of strings"),
    ("notes-int-element", {"notes": [1]}, r"notes\[0\] must be a string"),
    ("query_seed-str", {"query_seed": "7"}, r"query_seed must be an int or absent"),
    ("query_seed-bool", {"query_seed": True}, r"query_seed must be an int or absent"),
]

_EMBEDDER_CASES = [
    ("name-empty", {"name": ""}, r"embedder_name must be a non-empty string"),
    ("name-blank", {"name": " "}, r"embedder_name must not be blank"),
    ("name-surrogate", {"name": f"x{chr(0xD800)}"}, r"embedder_name"),
    ("cost-negative", {"cost": -1.0}, r"cost_per_million_tokens must be a finite number"),
    ("cost-nan", {"cost": float("nan")}, r"cost_per_million_tokens must be a finite number"),
    ("cost-bool", {"cost": True}, r"cost_per_million_tokens must be a finite number"),
    ("dim-zero", {"dim": 0}, r"embedder_dim must be >= 1"),
    ("dim-float", {"dim": 256.0}, r"embedder_dim must be an int"),
]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [c[1:] for c in _SWEEP_ARG_CASES],
    ids=[c[0] for c in _SWEEP_ARG_CASES],
)
def test_sweep_argument_refused_before_any_embed(kwargs: dict, match: str) -> None:
    corpus, queries = _inputs()
    embedder = _Counting()
    with pytest.raises(ValueError, match=match):
        run_sweep(corpus, queries, embedder=embedder, **kwargs)
    assert embedder.texts == 0


@pytest.mark.parametrize(
    ("attrs", "match"),
    [c[1:] for c in _EMBEDDER_CASES],
    ids=[c[0] for c in _EMBEDDER_CASES],
)
def test_embedder_attribute_refused_before_any_embed(attrs: dict, match: str) -> None:
    corpus, queries = _inputs()
    embedder = _Counting(**attrs)
    with pytest.raises(ValueError, match=match):
        run_sweep(corpus, queries, embedder=embedder)
    assert embedder.texts == 0


def test_clean_inputs_still_embed_and_record_notes_and_seed() -> None:
    """Control: the pre-check passes valid values through to the real result,
    and the stored `notes` is a copy, not the caller's list (#133)."""
    corpus, queries = _inputs()
    embedder = _Counting()
    notes = ["first note"]
    result = run_sweep(corpus, queries, embedder=embedder, notes=notes, query_seed=7)
    assert embedder.texts == 3
    assert result.notes == ["first note"]
    assert result.query_seed == 7
    notes.append("later")
    assert result.notes == ["first note"]
