"""Dep-free hash embedder for hermetic CI.

Bag-of-token-n-grams projected into a fixed-dim vector via SHA-256 hashing.
Not a real embedder — present so the sweep flow exercises end-to-end without
external services or API keys, and so the harness's metrics computation has
something to score.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence


class HashEmbedderProvider:
    """Sweep-compatible Embedder backed by hash projection."""

    def __init__(self, *, dim: int = 128, ngram: int = 2) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive; got {dim}")
        if ngram < 1:
            raise ValueError(f"ngram must be >= 1; got {ngram}")
        self.dim = dim
        self.ngram = ngram
        self.name = f"hash-embedder-{dim}d-ngram{ngram}"
        # Hash embedder doesn't cost anything; record 0.0 so the SweepResult
        # column is filled in.
        self.cost_per_million_tokens = 0.0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        tokens = [t for t in text.lower().split() if t]
        if self.ngram == 1:
            ngrams = list(tokens)
        else:
            ngrams = [
                " ".join(tokens[i : i + self.ngram]) for i in range(len(tokens) - self.ngram + 1)
            ]
        vec = [0.0] * self.dim
        if not ngrams:
            vec[0] = 1.0
            return vec
        for ng in ngrams:
            h = hashlib.sha256(ng.encode("utf-8")).digest()
            slot = int.from_bytes(h[:4], "big") % self.dim
            vec[slot] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
