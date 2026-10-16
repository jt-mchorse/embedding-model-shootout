"""`cosine` is scale-invariant, so a scaled embedding scores like itself (#226).

`sqrt(sum(x*x))` overflowed to inf (inf/inf = NaN) around 1e154 and underflowed
to 0 -- the zero-vector branch -- around 1e-154, so a finite non-zero embedding
scored like the zero vector. Measured on main, `HashEmbedderProvider` output
times a constant over an 8-chunk corpus:

    scale 1:      recall {1: 1.0, 5: 1.0}      ndcg@10 1.000
    scale 1e+200: recall {1: 0.125, 5: 0.625}  ndcg@10 0.494   (= k/N)
    scale 1e-170: recall {1: 0.125, 5: 0.625}  ndcg@10 0.494
"""

from __future__ import annotations

import math
import random

import pytest

from emb_shootout.providers.hash_embedder import HashEmbedderProvider
from emb_shootout.sweep import CorpusChunk, Query, cosine, run_sweep


def _old_cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _vectors(seed: int, n: int = 200, dim: int = 16) -> list[tuple[list[float], list[float]]]:
    rng = random.Random(seed)
    return [
        ([rng.uniform(-1, 1) for _ in range(dim)], [rng.uniform(-1, 1) for _ in range(dim)])
        for _ in range(n)
    ]


def test_ordinary_vectors_are_bit_identical_to_the_unscaled_formula() -> None:
    # The fix must not move any committed number: power-of-two scaling is exact.
    for a, b in _vectors(226):
        assert cosine(a, b) == _old_cosine(a, b)


@pytest.mark.parametrize("exponent", [-1000, -600, -300, 0, 300, 600, 1000])
def test_a_power_of_two_scale_gives_exactly_the_same_similarity(exponent: int) -> None:
    for a, b in _vectors(exponent % 97, n=50):
        scaled = ([math.ldexp(x, exponent) for x in a], [math.ldexp(x, exponent) for x in b])
        assert cosine(*scaled) == cosine(a, b)


@pytest.mark.parametrize("scale", [1e-300, 1e-170, 1e-100, 1e100, 1e200, 1e300])
def test_any_finite_scale_gives_the_same_similarity_to_rounding(scale: float) -> None:
    for a, b in _vectors(7, n=50):
        got = cosine([x * scale for x in a], [x * scale for x in b])
        assert math.isfinite(got)
        assert got == pytest.approx(cosine(a, b), abs=1e-12)


def test_the_zero_vector_still_scores_zero() -> None:
    assert cosine([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0
    assert cosine([0.0, 0.0], [0.0, 0.0]) == 0.0


class _Scaled:
    def __init__(self, scale: float) -> None:
        self._base = HashEmbedderProvider()
        self.scale = scale
        self.name = f"hash-x{scale:g}"
        self.dim = self._base.dim
        self.cost_per_million_tokens = 0.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[c * self.scale for c in v] for v in self._base.embed(texts)]


WORDS = [
    "apple pie recipe",
    "binary search tree",
    "tcp handshake",
    "photosynthesis in leaves",
    "french revolution",
    "quantum entanglement",
    "sourdough starter",
    "kubernetes pod",
]


def test_a_scaled_embedder_sweeps_like_the_unscaled_one() -> None:
    corpus = [CorpusChunk(f"c{i}", w) for i, w in enumerate(WORDS)]
    queries = [Query(f"q{i}", w, f"c{i}") for i, w in enumerate(WORDS)]
    results = {
        scale: run_sweep(corpus, queries, embedder=_Scaled(scale), k_values=(1, 5))
        for scale in (1.0, 1e200, 1e-170)
    }
    base = results[1.0]
    assert base.recall_at_k == {1: 1.0, 5: 1.0}  # anti-vacuity: the fixture separates
    for scale, r in results.items():
        assert r.recall_at_k == base.recall_at_k, scale
        assert r.ndcg_at_10 == base.ndcg_at_10, scale
