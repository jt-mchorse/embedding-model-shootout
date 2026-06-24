"""Sweep harness: same retrieval task, swappable embedder.

`run_sweep(corpus, queries, embedder)` embeds the corpus, embeds the queries,
runs cosine top-k retrieval, and reports `recall@k` for each k plus `NDCG@10`,
plus latency percentiles and the operator-supplied cost-per-million-tokens.

The retrieval task is intentionally simple — no rerankers, no fusion — so the
only variable across runs is the `Embedder`. This is what makes the
cross-provider comparison meaningful.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

# ----------------------------------------------------------------------
# Embedder Protocol
# ----------------------------------------------------------------------


class Embedder(Protocol):
    """Single-method seam every provider implements (D-004)."""

    name: str
    dim: int
    cost_per_million_tokens: float

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one float vector per input text. All vectors share `self.dim`."""


# ----------------------------------------------------------------------
# Data shapes
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusChunk:
    """One indexable chunk."""

    chunk_id: str
    text: str


@dataclass(frozen=True)
class Query:
    """One retrieval query with a known-correct answer chunk."""

    query_id: str
    text: str
    expected_chunk_id: str


@dataclass(frozen=True)
class SweepResult:
    """End-to-end result of one provider's sweep."""

    embedder_name: str
    embedder_dim: int
    cost_per_million_tokens: float
    n_corpus: int
    n_queries: int
    recall_at_k: dict[int, float]  # e.g., {1: 0.62, 5: 0.84, 10: 0.91}
    ndcg_at_10: float
    embed_latency_ms: dict[str, float]  # {"corpus_total": ..., "query_p50": ..., "query_p95": ...}
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # D-006 makes `cost_per_million_tokens` operator-supplied at provider
        # construction. A negative value silently inverts the Pareto-frontier
        # comparator at pareto.py:33-34 (a negative-cost provider dominates
        # every other point), so guard at the central construction site.
        # Embedder Protocol-implementers also benefit from the centralized
        # check without copying the validation per provider.
        # Finiteness guard (#31): a NaN cost_per_million_tokens propagates
        # into the Pareto frontier comparator at pareto.py:33-34 where
        # `a.cost < b.cost` and `a.cost <= b.cost` are both false for NaN —
        # the dominance check silently degrades. Same harm class as #29's
        # sign-only-rejects-negative, one step further along the
        # "operator-supplied numeric silently corrupts comparator" arc.
        if not math.isfinite(self.cost_per_million_tokens) or self.cost_per_million_tokens < 0.0:
            raise ValueError(
                f"cost_per_million_tokens must be a finite number >= 0.0; "
                f"got {self.cost_per_million_tokens}"
            )
        # Integer guard (#31): the three count fields are typed `int` but the
        # runtime can take floats. NaN/Infinity in dim makes `len(vec) == dim`
        # always false; fractional dim silently truncates via int comparison.
        # bool excluded explicitly (Python's bool subclasses int).
        if not isinstance(self.embedder_dim, int) or isinstance(self.embedder_dim, bool):
            raise ValueError(f"embedder_dim must be an int; got {self.embedder_dim!r}")
        if self.embedder_dim < 1:
            raise ValueError(f"embedder_dim must be >= 1; got {self.embedder_dim}")
        if not isinstance(self.n_corpus, int) or isinstance(self.n_corpus, bool):
            raise ValueError(f"n_corpus must be an int; got {self.n_corpus!r}")
        if self.n_corpus < 0:
            raise ValueError(f"n_corpus must be >= 0; got {self.n_corpus}")
        if not isinstance(self.n_queries, int) or isinstance(self.n_queries, bool):
            raise ValueError(f"n_queries must be an int; got {self.n_queries!r}")
        if self.n_queries < 0:
            raise ValueError(f"n_queries must be >= 0; got {self.n_queries}")
        # recall@k and nDCG@10 are proportions in [0, 1]. The guards above
        # cover cost/dim/counts but not these metric values, so a corrupt or
        # hand-edited result (loaded via `from_dict`) carrying recall=1.5 or a
        # NaN would silently win the Pareto-frontier comparison (pareto.py) and
        # render nonsensical points in the plot. Same "numeric silently corrupts
        # comparator" class as the cost guard above (#29/#31), on the metric axis.
        for k, v in self.recall_at_k.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"recall_at_k[{k}] must be a number; got {v!r}")
            if not math.isfinite(v) or not 0.0 <= v <= 1.0:
                raise ValueError(f"recall_at_k[{k}] must be a finite number in [0, 1]; got {v!r}")
        if not isinstance(self.ndcg_at_10, (int, float)) or isinstance(self.ndcg_at_10, bool):
            raise ValueError(f"ndcg_at_10 must be a number; got {self.ndcg_at_10!r}")
        if not math.isfinite(self.ndcg_at_10) or not 0.0 <= self.ndcg_at_10 <= 1.0:
            raise ValueError(
                f"ndcg_at_10 must be a finite number in [0, 1]; got {self.ndcg_at_10!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        # Explicit nine-field contract (#47) — no `asdict(self)`. A
        # future internal-only field on SweepResult can no longer
        # silently leak into the JSON consumed by the sweep + Pareto
        # frontier scripts. `recall_at_k` keys are stringified (JSON
        # has no integer key type); `embed_latency_ms` is already
        # string-keyed. `notes` is copied to a fresh list so caller
        # mutation of the returned dict doesn't bleed back into the
        # frozen dataclass.
        return {
            "embedder_name": self.embedder_name,
            "embedder_dim": self.embedder_dim,
            "cost_per_million_tokens": self.cost_per_million_tokens,
            "n_corpus": self.n_corpus,
            "n_queries": self.n_queries,
            "recall_at_k": {str(k): v for k, v in self.recall_at_k.items()},
            "ndcg_at_10": self.ndcg_at_10,
            "embed_latency_ms": dict(self.embed_latency_ms),
            "notes": list(self.notes),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SweepResult:
        return SweepResult(
            embedder_name=d["embedder_name"],
            embedder_dim=int(d["embedder_dim"]),
            cost_per_million_tokens=float(d["cost_per_million_tokens"]),
            n_corpus=int(d["n_corpus"]),
            n_queries=int(d["n_queries"]),
            recall_at_k={int(k): float(v) for k, v in d["recall_at_k"].items()},
            ndcg_at_10=float(d["ndcg_at_10"]),
            embed_latency_ms={k: float(v) for k, v in d["embed_latency_ms"].items()},
            notes=list(d.get("notes", [])),
        )


# ----------------------------------------------------------------------
# Math
# ----------------------------------------------------------------------


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def ndcg_at_k(relevances: list[int], k: int) -> float:
    """DCG@k / iDCG@k for binary relevance.

    `relevances[i]` is 0 or 1; the list is in ranked order (most relevant
    first). NDCG is bounded in [0, 1] for binary relevance.
    """
    # Integer + positive guard (#31). NaN passes the sign-only `<= 0` check
    # and then surfaces deep inside `relevances[:k]` as a cryptic TypeError;
    # fractional silently truncates via the slicing-int coercion.
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive integer; got {k!r}")
    if not relevances:
        return 0.0

    def _dcg(rs: list[int]) -> float:
        return sum(r / math.log2(i + 2) for i, r in enumerate(rs))

    actual = _dcg(relevances[:k])
    ideal = _dcg(sorted(relevances, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"p must be in [0, 100]; got {p}")
    s = sorted(values)
    rank = (p / 100.0) * (len(s) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return s[lo]
    frac = rank - lo
    return s[lo] * (1 - frac) + s[hi] * frac


# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------


def retrieve_top_k(
    query_vector: list[float],
    corpus_vectors: list[list[float]],
    chunk_ids: list[str],
    k: int,
) -> list[tuple[str, float]]:
    """Return the top-k chunk ids by cosine similarity, descending."""
    # Integer + positive guard (#31). NaN passes sign-only; fractional k
    # silently truncates via list slicing.
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive integer; got {k!r}")
    if len(corpus_vectors) != len(chunk_ids):
        raise ValueError("corpus_vectors and chunk_ids must be the same length")
    sims = [
        (chunk_ids[i], cosine(query_vector, corpus_vectors[i])) for i in range(len(corpus_vectors))
    ]
    sims.sort(key=lambda pair: pair[1], reverse=True)
    return sims[:k]


# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------


def run_sweep(
    corpus: Sequence[CorpusChunk],
    queries: Sequence[Query],
    *,
    embedder: Embedder,
    k_values: Sequence[int] = (1, 5, 10),
    notes: Sequence[str] = (),
) -> SweepResult:
    """End-to-end sweep: embed → retrieve → score.

    Returns a `SweepResult` with recall@k for each requested k, NDCG@10,
    embed-latency p50/p95, and the operator-supplied cost-per-million-tokens.

    The cost number is informational — the harness records it alongside the
    quality numbers so a future Pareto plot has all three axes (recall,
    latency, cost) without needing a separate price lookup.
    """
    if not corpus:
        raise ValueError("corpus must be non-empty")
    if not queries:
        raise ValueError("queries must be non-empty")
    if not k_values:
        raise ValueError("k_values must be non-empty")
    # Non-positive `k` passes through list slicing (`retrieved_ids[:k]`)
    # without raising — `k=0` produces a tautological recall@0=0, `k<0`
    # silently miscounts ("all but the last N" entries). Surface every
    # bad value in one pass so operators don't chase them one at a time.
    bad_k = sorted({k for k in k_values if k <= 0})
    if bad_k:
        raise ValueError(f"every k in k_values must be positive; got {bad_k}")
    # Duplicate `k` silently miscounts: the per-query loop below iterates
    # `k_values` directly and increments `hits_at_k[k]` once per occurrence,
    # while `hits_at_k` (and the output `recall_at_k` dict) carry only one
    # entry per distinct k. A k appearing N times therefore counts each hit N
    # times and yields recall > 1.0 — a mathematically invalid number from a
    # benchmark whose whole job is trustworthy retrieval scores. The output is
    # keyed by k, so a duplicate carries no distinct meaning; reject it loud in
    # the same one-pass style as the non-positive guard rather than silently
    # collapsing operator input.
    dup_k = sorted({k for k in k_values if list(k_values).count(k) > 1})
    if dup_k:
        raise ValueError(f"k_values must not contain duplicates; got duplicate {dup_k}")
    max_k = max(k_values)

    # Embed corpus (single batch).
    corpus_texts = [c.text for c in corpus]
    chunk_ids = [c.chunk_id for c in corpus]
    t0 = time.perf_counter()
    corpus_vectors = embedder.embed(corpus_texts)
    corpus_total_ms = (time.perf_counter() - t0) * 1000.0
    if len(corpus_vectors) != len(corpus_texts):
        raise ValueError(
            f"embedder returned {len(corpus_vectors)} vectors for {len(corpus_texts)} texts"
        )

    # Embed queries one at a time so we can capture per-query latency.
    query_latencies_ms: list[float] = []
    query_vectors: list[list[float]] = []
    for q in queries:
        t0 = time.perf_counter()
        vec = embedder.embed([q.text])[0]
        query_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        query_vectors.append(vec)

    # Compute hits per query at each k, plus NDCG@10.
    hits_at_k: dict[int, int] = dict.fromkeys(k_values, 0)
    ndcg_scores: list[float] = []
    for q, qvec in zip(queries, query_vectors, strict=True):
        top = retrieve_top_k(qvec, corpus_vectors, chunk_ids, max(max_k, 10))
        retrieved_ids = [cid for cid, _ in top]
        for k in k_values:
            if q.expected_chunk_id in retrieved_ids[:k]:
                hits_at_k[k] += 1
        rels = [1 if cid == q.expected_chunk_id else 0 for cid in retrieved_ids[:10]]
        ndcg_scores.append(ndcg_at_k(rels, 10))

    recall = {k: hits_at_k[k] / len(queries) for k in k_values}
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)

    return SweepResult(
        embedder_name=embedder.name,
        embedder_dim=embedder.dim,
        cost_per_million_tokens=embedder.cost_per_million_tokens,
        n_corpus=len(corpus),
        n_queries=len(queries),
        recall_at_k=recall,
        ndcg_at_10=avg_ndcg,
        embed_latency_ms={
            "corpus_total": corpus_total_ms,
            "query_p50": percentile(query_latencies_ms, 50.0),
            "query_p95": percentile(query_latencies_ms, 95.0),
        },
        notes=list(notes),
    )


# ----------------------------------------------------------------------
# Aggregator
# ----------------------------------------------------------------------


def _aggregate_ks(results: Sequence[SweepResult]) -> list[int]:
    """Union of `recall_at_k` keys across results, sorted ascending."""
    k_set: set[int] = set()
    for r in results:
        k_set.update(r.recall_at_k.keys())
    return sorted(k_set)


def aggregate_markdown(results: Sequence[SweepResult]) -> str:
    """Render a markdown comparison table over multiple SweepResults."""
    if not results:
        return "_no results to aggregate_\n"
    ks = _aggregate_ks(results)
    header_recall = " | ".join(f"recall@{k}" for k in ks)
    lines = [
        f"| embedder | dim | n_corpus | n_queries | {header_recall} | NDCG@10 | corpus embed (ms) | query p50 (ms) | query p95 (ms) | $/1M tokens |",
        "|----------|----:|---------:|----------:|"
        + "|".join("---:" for _ in ks)
        + "|--------:|------------------:|---------------:|---------------:|------------:|",
    ]
    for r in sorted(results, key=lambda x: x.embedder_name):
        recalls = " | ".join(f"{r.recall_at_k.get(k, 0.0):.3f}" for k in ks)
        lines.append(
            f"| {r.embedder_name} | {r.embedder_dim} | {r.n_corpus} | {r.n_queries} | {recalls} | "
            f"{r.ndcg_at_10:.3f} | {r.embed_latency_ms.get('corpus_total', 0.0):.0f} | "
            f"{r.embed_latency_ms.get('query_p50', 0.0):.1f} | "
            f"{r.embed_latency_ms.get('query_p95', 0.0):.1f} | "
            f"${r.cost_per_million_tokens:.3f} |"
        )
    return "\n".join(lines) + "\n"


def aggregate_json(results: Sequence[SweepResult]) -> dict:
    """JSON-shaped aggregation parallel to `aggregate_markdown`.

    Returns ``{"results": [<one_dict_per_provider>], "ks": [<int>, ...]}``.
    Rows sorted by ``embedder_name`` to match `aggregate_markdown`'s order,
    so a downstream consumer can cross-check the two formats line-by-line.
    """
    ks = _aggregate_ks(results)
    rows = []
    for r in sorted(results, key=lambda x: x.embedder_name):
        rows.append(
            {
                "embedder": r.embedder_name,
                "dim": r.embedder_dim,
                "n_corpus": r.n_corpus,
                "n_queries": r.n_queries,
                "recall": {str(k): r.recall_at_k.get(k, 0.0) for k in ks},
                "ndcg_at_10": r.ndcg_at_10,
                "corpus_embed_ms": r.embed_latency_ms.get("corpus_total", 0.0),
                "query_p50_ms": r.embed_latency_ms.get("query_p50", 0.0),
                "query_p95_ms": r.embed_latency_ms.get("query_p95", 0.0),
                "cost_per_million_tokens": r.cost_per_million_tokens,
            }
        )
    return {"results": rows, "ks": ks}
