"""A text shorter than one n-gram no longer embeds to a shared sentinel (#174).

`HashEmbedderProvider` (default `ngram=2`) built bigrams only, so a one-word text
fell through to `vec[0] = 1.0` and every one-word text embedded identically:
measured on `main`, `cos("json", "asyncio") == 1.0`.
"""

from __future__ import annotations

import math

from emb_shootout import build_corpus
from emb_shootout.providers.hash_embedder import HashEmbedderProvider


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )


def test_distinct_one_word_texts_get_distinct_vectors() -> None:
    words = ["json", "asyncio", "csv", "re", "typing", "pathlib"]
    vecs = HashEmbedderProvider().embed(words)
    assert len({tuple(v) for v in vecs}) == len(words)
    assert _cos(vecs[0], vecs[1]) < 1.0


def test_the_same_word_still_matches_itself() -> None:
    a, b = HashEmbedderProvider().embed(["json", "json"])
    assert a == b


def test_the_empty_text_keeps_its_sentinel() -> None:
    (vec,) = HashEmbedderProvider().embed([""])
    assert vec[0] == 1.0
    assert sum(abs(v) for v in vec) == 1.0


def test_corpus_texts_embed_bit_identically() -> None:
    # The corpus is built from the stdlib at run time (D-002; `data/` is not
    # committed). Every chunk `build_corpus` produces has >= 2 tokens, so the new
    # branch never runs for them; the old rule is spelled out here to prove the
    # vectors the sweep embeds are unchanged.
    def old(text: str, dim: int) -> list[float]:
        import hashlib

        tokens = [t for t in text.lower().split() if t]
        grams = [" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
        vec = [0.0] * dim
        if not grams:
            vec[0] = 1.0
            return vec
        for g in grams:
            vec[int.from_bytes(hashlib.sha256(g.encode()).digest()[:4], "big") % dim] += 1.0
        n = math.sqrt(sum(v * v for v in vec))
        return [v / n for v in vec]

    provider = HashEmbedderProvider()
    texts = [c.text for c in build_corpus(["json", "csv", "pathlib", "asyncio"])][:500]
    assert len(texts) > 100
    assert all(len(t.split()) >= 2 for t in texts)
    assert provider.embed(texts) == [old(t, provider.dim) for t in texts]
