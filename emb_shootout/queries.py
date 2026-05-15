"""Query-set construction.

Each query is a verbatim contiguous-word snippet drawn from a corpus chunk.
The query's `expected_chunk_id` is the chunk it came from; a perfect retriever
returns that chunk first. Construction is deterministic given a `seed`.

This shape is deliberate — the alternative is a hand-curated query set that
drifts from the corpus as the corpus evolves. Deriving queries from the
corpus at sweep time means corpus + queries are always perfectly in sync
(D-005).
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence

from .sweep import CorpusChunk, Query

DEFAULT_QUERY_COUNT = 200
DEFAULT_SEED = 42


def build_queries(
    corpus: Sequence[CorpusChunk],
    *,
    n: int = DEFAULT_QUERY_COUNT,
    seed: int = DEFAULT_SEED,
    min_words: int = 6,
    max_words: int = 15,
) -> list[Query]:
    """Build `n` deterministic verbatim-snippet queries from `corpus`.

    Picks chunks at random (with replacement to support n > len(corpus)),
    then picks a contiguous word window of length [min_words, max_words]
    from each picked chunk, and uses that as the query. The chunk's id is
    the `expected_chunk_id`.

    Chunks too short for `min_words` are skipped.
    """
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    if min_words <= 0 or max_words < min_words:
        raise ValueError(f"need 0 < min_words <= max_words; got {min_words}, {max_words}")
    if not corpus:
        raise ValueError("corpus must be non-empty")

    rng = random.Random(seed)
    eligible = [(c, _words_of(c.text)) for c in corpus if len(_words_of(c.text)) >= min_words]
    if not eligible:
        raise ValueError(f"no corpus chunks have >= {min_words} words")

    queries: list[Query] = []
    for i in range(n):
        chunk, words = rng.choice(eligible)
        max_for_chunk = min(max_words, len(words))
        window = rng.randint(min_words, max_for_chunk)
        start = rng.randint(0, len(words) - window)
        snippet = " ".join(words[start : start + window])
        queries.append(
            Query(
                query_id=f"q-{i:05d}",
                text=snippet,
                expected_chunk_id=chunk.chunk_id,
            )
        )
    return queries


_WORD_RE = re.compile(r"\S+")


def _words_of(text: str) -> list[str]:
    return _WORD_RE.findall(text)
