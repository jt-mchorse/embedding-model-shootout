"""Construction-time validation for the five parameters PR #34 explicitly
left as deferred follow-ups.

`HashEmbedderProvider.dim`, `.ngram` (`emb_shootout/providers/hash_embedder.py:23`)
and `build_queries.n`, `.min_words`, `.max_words` (`emb_shootout/queries.py:25`)
all had sign-only checks. Each silently accepted `bool` (Python subclass of
`int`) and whole-float values, and surfaced opaque downstream errors.

Validator shape matches the portfolio positive-int contract:
`not isinstance(int) or isinstance(bool) or <= 0`. See
`runs.list_runs.limit` in `llm-eval-harness#42`, `rag-production-kit#41/#43`,
`llm-cost-optimizer#39`, `llm-eval-harness#45`.
"""

from __future__ import annotations

import math

import pytest

from emb_shootout.providers.hash_embedder import HashEmbedderProvider
from emb_shootout.queries import build_queries
from emb_shootout.sweep import CorpusChunk

# ----------------------------------------------------------------------
# Corpus fixture small enough to make build_queries validators reachable
# without pulling in heavy dataset setup.
# ----------------------------------------------------------------------


_CORPUS_FIXTURE = [
    CorpusChunk(
        chunk_id=f"c-{i}",
        text=(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa "
            "lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
        ),
    )
    for i in range(3)
]


# ======================================================================
# HashEmbedderProvider.dim
# ======================================================================


@pytest.mark.parametrize(
    "bad_dim",
    [
        True,  # bool: silently bound self.dim=True, [0.0]*True returned [0.0].
        False,  # bool: caught by sign-only but masked type bug.
        0,
        -1,
        -128,
        0.5,
        128.0,  # whole float: [0.0]*128.0 raised TypeError later.
        math.nan,
        math.inf,
        -math.inf,
        None,
        "128",
    ],
)
def test_hash_provider_rejects_non_positive_int_dim(bad_dim):
    with pytest.raises(ValueError, match="dim must be a positive integer"):
        HashEmbedderProvider(dim=bad_dim)


@pytest.mark.parametrize("good_dim", [1, 8, 128, 1024, 65536])
def test_hash_provider_accepts_positive_int_dim(good_dim):
    p = HashEmbedderProvider(dim=good_dim)
    assert p.dim == good_dim


# ======================================================================
# HashEmbedderProvider.ngram
# ======================================================================


@pytest.mark.parametrize(
    "bad_ngram",
    [
        True,  # bool: silently treated as unigram; name="...-ngramTrue".
        False,  # bool: would silently bypass sign as 0 == False.
        0,  # original "ngram < 1" caught 0 too, but new shape is uniform.
        -1,
        -2,
        0.5,
        2.0,  # whole float: silently bound, then range() raised later.
        math.nan,
        math.inf,
        -math.inf,
        None,
        "2",
    ],
)
def test_hash_provider_rejects_non_positive_int_ngram(bad_ngram):
    with pytest.raises(ValueError, match="ngram must be a positive integer"):
        HashEmbedderProvider(ngram=bad_ngram)


@pytest.mark.parametrize("good_ngram", [1, 2, 3, 5, 10])
def test_hash_provider_accepts_positive_int_ngram(good_ngram):
    p = HashEmbedderProvider(ngram=good_ngram)
    assert p.ngram == good_ngram


# ======================================================================
# build_queries.n
# ======================================================================


@pytest.mark.parametrize(
    "bad_n",
    [
        True,  # bool: silently produced 1-query set.
        False,
        0,
        -1,
        0.5,
        200.5,
        200.0,  # whole float: bypassed sign-only, range(n) raised TypeError.
        math.nan,
        math.inf,
        -math.inf,
        None,
        "200",
    ],
)
def test_build_queries_rejects_non_positive_int_n(bad_n):
    with pytest.raises(ValueError, match="n must be a positive integer"):
        build_queries(_CORPUS_FIXTURE, n=bad_n)


# ======================================================================
# build_queries.min_words / max_words
# ======================================================================


@pytest.mark.parametrize(
    "bad_min",
    [
        True,
        False,
        0,
        -1,
        0.5,
        6.0,
        math.nan,
        math.inf,
        -math.inf,
        None,
        "6",
    ],
)
def test_build_queries_rejects_non_positive_int_min_words(bad_min):
    with pytest.raises(ValueError, match="min_words must be a positive integer"):
        build_queries(_CORPUS_FIXTURE, n=5, min_words=bad_min, max_words=10)


@pytest.mark.parametrize(
    "bad_max",
    [
        True,
        False,
        0,
        -1,
        0.5,
        15.0,
        math.nan,
        math.inf,
        -math.inf,
        None,
        "15",
    ],
)
def test_build_queries_rejects_non_positive_int_max_words(bad_max):
    with pytest.raises(ValueError, match="max_words must be a positive integer"):
        build_queries(_CORPUS_FIXTURE, n=5, min_words=6, max_words=bad_max)


def test_build_queries_paired_check_runs_after_type_contract():
    # Both are valid positive ints — the paired check still fires.
    with pytest.raises(ValueError, match="need min_words <= max_words"):
        build_queries(_CORPUS_FIXTURE, n=5, min_words=10, max_words=6)


def test_build_queries_boundary_accepts_min_equals_max():
    # Smallest valid pairing: min_words == max_words.
    queries = build_queries(_CORPUS_FIXTURE, n=2, min_words=5, max_words=5, seed=1)
    assert len(queries) == 2
    for q in queries:
        assert len(q.text.split()) == 5


def test_build_queries_smallest_valid_n_produces_one_query():
    queries = build_queries(_CORPUS_FIXTURE, n=1, seed=1)
    assert len(queries) == 1
