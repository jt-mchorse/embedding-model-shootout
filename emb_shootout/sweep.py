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
import re
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
        #
        # bool excluded explicitly (Python's bool subclasses int, so
        # `math.isfinite(True)` is True and `True < 0.0` is False): this was the
        # one numeric field the sibling bool exclusions (embedder_dim/n_corpus/
        # n_queries/ndcg_at_10/recall_at_k/embed_latency_ms) overlooked, so a
        # provider-supplied boolean cost (`embedder.cost_per_million_tokens`,
        # this is the Protocol-implementer validation choke-point) was stored as
        # a fabricated $1.0/$0.0 point on the Pareto frontier and the committed
        # plot. `from_dict` already rejects a bool cost pre-coercion (#108); this
        # closes its direct-construction sibling.
        if (
            isinstance(self.cost_per_million_tokens, bool)
            or not math.isfinite(self.cost_per_million_tokens)
            or self.cost_per_million_tokens < 0.0
        ):
            raise ValueError(
                f"cost_per_million_tokens must be a finite number >= 0.0; "
                f"got {self.cost_per_million_tokens!r}"
            )
        # Integer guard (#31): the three count fields are typed `int` but the
        # runtime can take floats. NaN/Infinity in dim makes the
        # `len(vec) != declared_dim` seam check in `_require_declared_dim`
        # always fire; a fractional dim silently truncates via int comparison.
        # (That seam check did not exist when this comment was written — it
        # cited a consumer that was never built. Added in #119.)
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
        # `embed_latency_ms` is the one numeric field the guards above overlooked
        # (#65). `from_dict` coerces with `float(v)`, which accepts the JSON
        # tokens "Infinity"/"NaN", so a hand-edited or externally-generated result
        # carries a non-finite latency straight into the markdown table and the
        # JSON aggregate — an invalid-JSON Infinity/NaN token or a fabricated
        # latency number. A latency is physically finite and non-negative; reject
        # like the recall_at_k loop above (#31).
        for k, v in self.embed_latency_ms.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"embed_latency_ms[{k!r}] must be a number; got {v!r}")
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"embed_latency_ms[{k!r}] must be a finite number >= 0; got {v!r}")

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
        # `from_dict` is the single validation choke-point for result files, and
        # hand-editing them is an explicitly invited workflow (#75). A result can
        # be valid JSON yet structurally malformed: not an object, a missing
        # required field, or a `recall_at_k`/`embed_latency_ms` of the wrong
        # container type. Left raw, `d["k"]` / `int(...)` / `.items()` escaped as
        # KeyError/TypeError/AttributeError at exit 1 — bypassing the "name the
        # bad file and exit 2" contract the sweep CLI documents and that the
        # numeric-range guards in `__post_init__` already honor via ValueError.
        # Translate every structural failure to a ValueError naming the offending
        # field, mirroring `_read_corpus_jsonl` in cli.py (#85).
        if not isinstance(d, dict):
            raise ValueError(f"result must be a JSON object; got {type(d).__name__}")
        required = (
            "embedder_name",
            "embedder_dim",
            "cost_per_million_tokens",
            "n_corpus",
            "n_queries",
            "recall_at_k",
            "ndcg_at_10",
            "embed_latency_ms",
        )
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"result missing required field(s): {', '.join(missing)}")
        # `embedder_name` is the one required field with no coercion below, so a
        # present-but-non-string value (int/null from a hand-edited result) slips
        # past the `except TypeError` and only crashes later at exit 1: a raw
        # `AttributeError` in `aggregate_markdown` (`.replace("|", ...)`) and a
        # `TypeError` when sorting a batch of mixed-type names. Guard it here with
        # the isinstance-str + non-empty check, the sibling of the #94 corpus
        # loader `chunk_id`/`text` fix, so the CLI maps it to exit 2.
        if not isinstance(d["embedder_name"], str) or not d["embedder_name"]:
            raise ValueError(
                f"embedder_name must be a non-empty string; got {d['embedder_name']!r} "
                f"({type(d['embedder_name']).__name__})"
            )
        for container_field in ("recall_at_k", "embed_latency_ms"):
            if not isinstance(d[container_field], dict):
                raise ValueError(
                    f"{container_field} must be a JSON object; "
                    f"got {type(d[container_field]).__name__}"
                )
        # bool is an int subclass, so `int(True)`/`float(True)` below silently
        # coerce a JSON `true`/`false` to 1/0 — and the `__post_init__` bool guards
        # (`isinstance(x, bool)`) run AFTER that coercion, so they never fire for
        # this loader path (they protect only direct construction). A hand-edited
        # or externally-generated result with a boolean metric would land a
        # fabricated 1.0 recall/ndcg on the Pareto frontier and the committed JSON.
        # Reject bools on the RAW values here, before coercion — the
        # isinstance-before-coerce sibling of the #94/#95 `embedder_name` guard.
        for scalar_field in (
            "embedder_dim",
            "cost_per_million_tokens",
            "n_corpus",
            "n_queries",
            "ndcg_at_10",
        ):
            if isinstance(d[scalar_field], bool):
                raise ValueError(
                    f"{scalar_field} must be a number, not a bool; got {d[scalar_field]!r}"
                )
        for container_field in ("recall_at_k", "embed_latency_ms"):
            for k, v in d[container_field].items():
                if isinstance(v, bool):
                    raise ValueError(
                        f"{container_field}[{k!r}] must be a number, not a bool; got {v!r}"
                    )
        try:
            return SweepResult(
                embedder_name=d["embedder_name"],
                embedder_dim=int(d["embedder_dim"]),
                cost_per_million_tokens=float(d["cost_per_million_tokens"]),
                n_corpus=int(d["n_corpus"]),
                n_queries=int(d["n_queries"]),
                recall_at_k={k: float(v) for k, v in _coerce_recall_keys(d["recall_at_k"]).items()},
                ndcg_at_10=float(d["ndcg_at_10"]),
                embed_latency_ms={k: float(v) for k, v in d["embed_latency_ms"].items()},
                notes=list(d.get("notes", [])),
            )
        except TypeError as e:
            # A required field present but of a non-coercible type (e.g.
            # embedder_dim=[1] or a null cost) reaches int()/float() as a
            # TypeError. Coercion ValueErrors (int("x")) and the __post_init__
            # range ValueErrors already are ValueError and pass straight through
            # to the CLI's exit-2 handler.
            raise ValueError(f"result has a non-numeric field: {e}") from e


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


def _require_one_vector_per_text(
    vectors: list[list[float]], n_texts: int, *, embedder_name: str, kind: str
) -> None:
    """Fail loud unless the embedder returned exactly one vector per input.

    The `Embedder` Protocol promises "one float vector per input text" (D-004),
    and nothing enforces it — `embed` is a BYO seam, and the five shipped SDK
    providers each assemble their output differently (sorting `response.data`
    by `.index`, reading `response.embeddings.float`, extending from
    `response.embeddings`, mapping over a numpy array). None of them checks the
    row count, so a truncated or partial batch response lands here.

    The corpus call had this check inline; the per-query call nine lines below
    it did not, and both ways of breaking the contract were bad there (#112):

    - **Too few.** `embedder.embed([q.text])[0]` on an empty list raised
      `IndexError`, which `cli`'s `sweep run` doesn't catch (it catches
      `ValueError`), so it escaped as a traceback at exit 1 instead of the
      documented exit-2 contract.
    - **Too many.** `[0]` took the first row and asked no questions, so a
      provider that prepended or misordered one vector produced a *plausible
      and entirely wrong* benchmark number — recall@1 of 0.0 where an honest
      embedder scores 1.0, with no error anywhere. That is the same corruption
      `_reject_non_finite_vectors` below was added to prevent, reached through
      arity instead of value, and it is equally invisible to the `SweepResult`
      range guard (#62).

    One helper rather than two inline checks, because the gap existed exactly
    because the two seams were separate expressions of the same contract.
    """
    if len(vectors) != n_texts:
        raise ValueError(
            f"embedder {embedder_name!r} returned {len(vectors)} vectors for "
            f"{n_texts} {kind} text{'' if n_texts == 1 else 's'}; the Embedder "
            "protocol promises one vector per input text, and a mismatch either "
            "crashes the sweep or silently scores the wrong vector"
        )


def _require_declared_dim(
    vectors: list[list[float]], declared_dim: int, *, embedder_name: str, kind: str
) -> None:
    """Fail loud unless every vector has exactly the embedder's declared `dim`.

    The `Embedder` Protocol docstring is one sentence with two clauses: "Return
    one float vector per input text. All vectors share `self.dim`." #112
    enforced the first clause (`_require_one_vector_per_text`); this enforces
    the second, at the same seam, because it fails the same way for the same
    reason.

    `SweepResult.embedder_dim` is copied straight off `embedder.dim` and was
    never reconciled with what the embedder returned, so a provider that
    declares one dimension and returns another produced a clean, successful
    sweep whose recorded dimension was fiction. Measured: declaring 1024 while
    returning 8-component vectors recorded `embedder_dim: 1024` at exit 0, and
    declaring 8 while returning 1024 recorded `8` — with a *different* NDCG,
    since the quality number genuinely depends on the real dimension. That
    lands a real measured score next to a fabricated dimension in
    `results/*.json`, the README table and the Pareto plot, where dim is a
    first-class axis.

    `cosine`'s length guard cannot cover this. It compares two vectors, so it
    only fires when corpus and query vectors disagree with *each other*. The
    reachable case is uniform — a provider hardcodes `DEFAULT_DIM` and the
    upstream model's output size changes, or an operator passes a wrong `dim=`
    — and uniformly-wrong vectors are perfectly self-consistent. When lengths
    *do* differ, `cosine` raises from inside the retrieval loop naming neither
    the embedder nor the seam; checking here names both.
    """
    for vi, vec in enumerate(vectors):
        if len(vec) != declared_dim:
            raise ValueError(
                f"embedder {embedder_name!r} declares dim={declared_dim} but returned a "
                f"{len(vec)}-component {kind} vector at index {vi}; the Embedder protocol "
                "promises all vectors share `self.dim`, and a mismatch records a "
                "fabricated dimension alongside a real quality score"
            )


def _reject_non_finite_vectors(
    vectors: list[list[float]], *, embedder_name: str, kind: str
) -> None:
    """Fail loud if any component of any vector is non-finite (NaN/Inf).

    A single non-finite component makes `cosine` return NaN, and a NaN
    similarity compares False against everything, so `retrieve_top_k`'s sort
    leaves the ranking scrambled. The resulting `recall = hits / n_queries`
    (and averaged NDCG) are still finite values in [0, 1], so the SweepResult
    finiteness/range guard (#62) cannot catch the corruption — the benchmark
    would report plausible-but-wrong numbers. Reject at the embedder-output
    seam (where the length check already lives), not inside the hot `cosine`
    path, matching prompt-regression-suite's `NonFiniteEmbeddingError` seam.
    """
    for vi, vec in enumerate(vectors):
        for ci, v in enumerate(vec):
            if not math.isfinite(v):
                raise ValueError(
                    f"embedder {embedder_name!r} returned a non-finite component in "
                    f"{kind} vector {vi} at dim {ci}: {v!r}. A NaN/Inf embedding "
                    "scrambles the cosine ranking and yields plausible-but-wrong "
                    "recall/NDCG; fix the embedder or re-run."
                )


def _reject_zero_vectors(vectors: list[list[float]], *, embedder_name: str, kind: str) -> None:
    """Fail loud if any vector is all zeros.

    The fourth thing `embed` can return that corrupts a published score, beside
    the wrong count (#112), the wrong length, and a non-finite component (#125).
    It is the one that cannot be caught downstream: `cosine` deliberately maps a
    zero-norm vector to `0.0`, and `retrieve_top_k`'s comment names that as a
    dependency ("zero-norm -> 0.0 in `cosine`, so negation is safe").

    A zero vector is not a bad measurement, it is the **absence** of one -- it is
    equidistant from every other vector, so it carries no information about which
    chunk answers which query. Scored anyway, it produces a plausible number.
    Measured on a 6-chunk corpus with one query per chunk, an embedder returning
    all zeros::

        recall@1 = 0.167   recall@3 = 0.500   recall@5 = 0.833   ndcg@10 = 0.551

    Exactly `k/N`. Every similarity ties at `0.0`, so the ranking falls entirely
    to the `chunk_id` tiebreak, which is the same order for every query -- query
    *i* hits iff `i < k`. Partial corruption is worse because it looks ordinary:
    one zeroed corpus chunk out of six reads `recall@1 = 0.833`, exactly the
    shape of an honest "slightly worse embedder".

    And #73 removed the only symptom. Its `chunk_id` tiebreak made ranking a pure
    function of `(similarity, chunk_id)` -- correct, and a determinism fix -- but
    when every similarity is equal the tiebreak stops breaking ties and becomes
    the whole ranking. Measured across five corpus shuffles the all-zero result
    is bit-identical, so the score that would once have jumped around under
    reordering is now stably, reproducibly meaningless. Determinism was achieved;
    meaningfulness was not, and nothing downstream tells them apart.

    Exactly zero-norm, deliberately -- not "small norm". A near-zero vector is
    still a measurement, and any threshold would be a number nobody measured,
    which is the thing this repo's handoff forbids. Same rule
    `llm-eval-harness` D-017 settled for its own embed seam: a zero embed vector
    means *uncomparable*, rejected on the authored side. Both inputs here are
    authored fixtures -- a committed corpus and a committed query set -- so the
    authored-side rule is the one that applies (#131).
    """
    for vi, vec in enumerate(vectors):
        if vec and not any(vec):
            raise ValueError(
                f"embedder {embedder_name!r} returned an all-zero {kind} vector at "
                f"index {vi}. A zero vector is equidistant from every other vector, "
                "so `cosine` scores it 0.0 against everything and the ranking falls "
                "to the chunk_id tiebreak -- yielding a plausible recall/NDCG "
                "(exactly k/N when every vector is zero) that measures nothing; "
                "fix the embedder or re-run."
            )


def ndcg_at_k(relevances: list[int], k: int) -> float:
    """DCG@k / iDCG@k. Bounded in [0, 1] for any non-negative relevances.

    `relevances[i]` is a non-negative integer gain — 0/1 for binary relevance,
    which is what `run_sweep` derives, and graded values work too: `[3, 2, 1, 0]`
    scores 1.000, because `ideal` is computed over the *sorted* list and so is a
    genuine maximum. The list is in ranked order (most relevant first).

    The bound is a property of non-negativity, not of binariness. The docstring
    used to say "for binary relevance", which understated the guarantee in one
    direction and overstated it in another: a *negative* gain breaks the bound
    outright (#125).
    """
    # Integer + positive guard (#31). NaN passes the sign-only `<= 0` check
    # and then surfaces deep inside `relevances[:k]` as a cryptic TypeError;
    # fractional silently truncates via the slicing-int coercion.
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive integer; got {k!r}")
    # `relevances` is the other operand of the same expression #31 hardened `k`
    # in, and it was left unchecked. Measured (#125):
    #
    #   relevances        k          ndcg   in [0, 1]?
    #   [1, 1, 0, 0]      4      1.000000   yes
    #   [3, 2, 1, 0]      4      1.000000   yes    graded was always fine
    #   [-10, 3]          1     -3.333333   NO
    #   [-1, 20]          1     -0.050000   NO
    #   [-1, 1]           2     -1.000000   NO
    #   [inf, 1]          2           nan   NO
    #   [nan, 1]          2      0.000000   yes -- and that is the problem
    #   ["1", "0"]        2     raw TypeError, outside this module's contract
    #
    # The NaN row is the quietest and the worst: `_dcg` yields NaN, `ideal > 0`
    # is False for NaN, so the fallback returns a clean 0.000 -- indistinguishable
    # from "nothing relevant was retrieved". Same shape as the extreme-default
    # class #123 closed, where an unmeasured recall scored 0.0 and was DOMINATED
    # on the Pareto frontier rather than excluded from it.
    #
    # Validated rather than documented-around, and this repo has already made
    # that call for this very number: `SweepResult.__post_init__` enforces
    # `0.0 <= ndcg_at_10 <= 1.0` with a ValueError. The consumer treats the range
    # as a hard contract; the producer only promised it in prose. Composing them
    # surfaced the failure as "ndcg_at_10 must be a finite number in [0, 1]" --
    # a message naming the FIELD rather than the relevance list at fault.
    #
    # Scoped to non-negative integers, which is what the annotation and the
    # docstring both say. Graded gains stay accepted; only sign and type are
    # constrained.
    for i, r in enumerate(relevances):
        if not isinstance(r, int) or isinstance(r, bool):
            raise ValueError(f"relevances[{i}] must be a non-negative integer gain; got {r!r}")
        if r < 0:
            raise ValueError(
                f"relevances[{i}] must be a non-negative integer gain; got {r}. "
                f"A negative gain breaks the [0, 1] bound this function documents "
                f"and SweepResult enforces"
            )
    if not relevances:
        return 0.0

    def _dcg(rs: list[int]) -> float:
        return sum(r / math.log2(i + 2) for i, r in enumerate(rs))

    actual = _dcg(relevances[:k])
    ideal = _dcg(sorted(relevances, reverse=True)[:k])
    # After the guard above, the only list that reaches `ideal <= 0` is one of
    # all zeros, where 0.0 is the truthful answer -- so this fallback no longer
    # doubles as a silent catch-all for a corrupt input, which is what it was
    # doing for NaN (#125).
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
    # Break similarity ties on the stable chunk id, not corpus insertion order.
    # `reverse=True` on the score alone left tied chunks in their corpus-read
    # order, so recall@k / NDCG at a top-k boundary depended on how the corpus
    # was passed in — a benchmark whose job is trustworthy scores must rank as a
    # pure function of the (similarity, chunk_id) set (#73). Negate the score for
    # an ascending composite sort (reverse=True would also reverse the id
    # tiebreak). Similarities are guaranteed finite (`_reject_non_finite_vectors`
    # + zero-norm → 0.0 in `cosine`), so negation is safe — same shape as
    # chunking-strategies-lab #68.
    sims.sort(key=lambda pair: (-pair[1], pair[0]))
    return sims[:k]


# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------


def _coerce_recall_keys(raw: dict[str, Any]) -> dict[int, Any]:
    """Coerce `recall_at_k`'s JSON string keys back to `int`, loudly (#129).

    `to_dict` writes `{str(k): v}`. Reading that back with a bare
    `{int(k): ... }` was not the inverse: `int()` accepts leading zeros,
    surrounding whitespace, a leading `+`, `_` digit separators and non-ASCII
    decimal digits, none of which `str(int)` can produce. Measured on the
    unguarded version::

        {"5": 0.9, "05": 0.1}      -> {5: 0.1}    0.9 gone
        {"5": 0.9, " 5": 0.1}      -> {5: 0.1}    0.9 gone
        {"5": 0.9, "+5": 0.1}      -> {5: 0.1}    0.9 gone
        {"5": 0.9, "\u0665": 0.1} -> {5: 0.1}    0.9 gone (Arabic-Indic five)
        {"5_0": 0.9}               -> {50: 0.9}   a typo becomes a different k
        {"0": 0.9} / {"-3": 0.9}   -> loaded

    Requiring the *canonical* spelling makes a collision impossible by
    construction -- two distinct canonical spellings cannot coerce to the same
    int -- so there is no separate collision check to fall out of sync.

    The contrast that made this worth fixing is inside `from_dict` itself: every
    *value* on this read path is checked and named (`recall_at_k[5] must be a
    finite number in [0, 1]; got nan`), and the key axis was `int()`. The `[5]`
    in that message is the only place a key was used, and it was already the
    coerced one.

    The range half delegates to :func:`validate_k_values`, the same rule
    `run_sweep` applies before *producing* these keys -- which rejects `k <= 0`
    because "`k=0` produces a tautological recall@0=0, `k<0` silently
    miscounts". The loader accepted both from a file, and `from_dict`'s own
    comment says hand-editing result files is "an explicitly invited workflow
    (#75)".
    """
    out: dict[int, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            # Unreachable from `json.loads` (JSON names are strings), reachable
            # from a hand-built dict passed straight to `from_dict`.
            raise ValueError(f"recall_at_k key must be a string; got {key!r}")
        try:
            k = int(key)
        except ValueError:
            raise ValueError(
                f"recall_at_k key {key!r} is not an integer; `to_dict` writes these "
                f"keys as `str(k)` for the integer k of recall@k"
            ) from None
        if str(k) != key:
            raise ValueError(
                f"recall_at_k key {key!r} is not the canonical spelling of {k} "
                f"(expected {str(k)!r}); `int()` accepts leading zeros, surrounding "
                f"whitespace, a leading '+', '_' digit separators and non-ASCII "
                f"decimal digits, none of which `to_dict` can write -- and two such "
                f"spellings collide silently, dropping a measurement"
            )
        out[k] = value
    if out:
        # Same rule the write path applies, shared rather than restated. The
        # message is re-raised with the field name because `validate_k_values`
        # says "k_values", which is right for `run_sweep` and wrong for a loader
        # whose other errors all name the field they came from.
        try:
            validate_k_values(sorted(out))
        except ValueError as e:
            raise ValueError(f"recall_at_k keys: {e}") from None
    return out


def validate_k_values(k_values: Sequence[int]) -> None:
    """Reject a `k_values` this module cannot honour. Raises ``ValueError``.

    Extracted from ``run_sweep``'s body (#129) so ``SweepResult.from_dict``
    can apply the *same* rule to the ``recall_at_k`` keys it reads back,
    instead of a second copy. `#121`'s note below already observes that the
    complete rule "lives TWICE in this module"; a fourth inline copy in the
    loader would have been the worst available answer. Mirrors the
    ``validate_ks`` extraction in chunking-strategies-lab (#149), whose own
    loader became its third caller in csl#169.

    The duplicate check is vacuous for a caller passing a dict's keys --
    they are unique by construction -- and is kept here rather than split
    out, because the point of the extraction is that there is exactly one
    definition of a valid ``k_values``.
    """
    if not k_values:
        raise ValueError("k_values must be non-empty")
    # Type, before sign (#121). The complete rule already lives TWICE in this
    # module — `ndcg_at_k` and `retrieve_top_k` both carry
    # `not isinstance(k, int) or isinstance(k, bool) or k <= 0` from #31 — and
    # neither can be reached with the operator's actual value, because
    # `retrieve_top_k` is called with `max(max_k, 10)` and `max(True, 10)` is
    # `10`, `max(2.5, 10)` is `10`. The floor launders the bad type away before
    # the only type guard in the file can see it.
    #
    # The measured consequence is that the SAME defect gets a different
    # diagnostic depending on whether the typo happens to exceed 10:
    #
    #   k_values=(2.5,)   -> TypeError: slice indices must be integers ...
    #                        (raw, from `retrieved_ids[:k]`, names nothing)
    #   k_values=(20.5,)  -> ValueError: k must be a positive integer; got 20.5
    #   k_values=(True,)  -> no error at all
    #
    # And the default is `(1, 5, 10)`, so an operator staying near the defaults
    # sits entirely inside the range with the worst diagnostic.
    #
    # `(True,)` was the loudest case: the sweep COMPLETED, `retrieved_ids[:True]`
    # took one element, and `to_dict` emitted `{"True": 0.5}` — which
    # `SweepResult.from_dict` then rejects with `invalid literal for int() with
    # base 10: 'True'`. The writer produced a result file its own reader refuses,
    # and `_aggregate_ks` would have unioned that key straight into a
    # `recall@True` column in the committed comparison table.
    #
    # Ordering is load-bearing, not cosmetic: `k <= 0` raises `TypeError` on a
    # `str`/`None` element, so the sign check below cannot run at all until the
    # non-numerics are gone. Collected in one pass, like both checks after it.
    #
    # An integral float (`3.0`) is rejected rather than coerced, so this gate
    # agrees with the two downstream guards that already reject it. It is also
    # the reachable float: `json.loads("3.0")` is `3.0`, so a `k_values` list
    # from a config file or notebook cell carries floats with no decimal point
    # ever typed.
    bad_type = [k for k in k_values if isinstance(k, bool) or not isinstance(k, int)]
    if bad_type:
        raise ValueError(
            f"every k in k_values must be an int (bool excluded); got {bad_type!r} — "
            "coerce with int(k) if these came from JSON"
        )
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
    validate_k_values(k_values)
    max_k = max(k_values)

    # Embed corpus (single batch).
    corpus_texts = [c.text for c in corpus]
    chunk_ids = [c.chunk_id for c in corpus]
    t0 = time.perf_counter()
    corpus_vectors = embedder.embed(corpus_texts)
    corpus_total_ms = (time.perf_counter() - t0) * 1000.0
    _require_one_vector_per_text(
        corpus_vectors, len(corpus_texts), embedder_name=embedder.name, kind="corpus"
    )
    _require_declared_dim(corpus_vectors, embedder.dim, embedder_name=embedder.name, kind="corpus")
    _reject_non_finite_vectors(corpus_vectors, embedder_name=embedder.name, kind="corpus")
    _reject_zero_vectors(corpus_vectors, embedder_name=embedder.name, kind="corpus")

    # Embed queries one at a time so we can capture per-query latency.
    query_latencies_ms: list[float] = []
    query_vectors: list[list[float]] = []
    for q in queries:
        t0 = time.perf_counter()
        vecs = embedder.embed([q.text])
        query_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        # Same contract as the corpus call above. Indexing `[0]` straight off
        # the result raised `IndexError` on an empty return and silently scored
        # the wrong vector on an over-long one (#112).
        _require_one_vector_per_text(vecs, 1, embedder_name=embedder.name, kind="query")
        _require_declared_dim(vecs, embedder.dim, embedder_name=embedder.name, kind="query")
        query_vectors.append(vecs[0])
    _reject_non_finite_vectors(query_vectors, embedder_name=embedder.name, kind="query")
    _reject_zero_vectors(query_vectors, embedder_name=embedder.name, kind="query")

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


#: Rendered where a result has no measurement for some column (#123, #127).
#: An em dash rather than a blank so the column stays visible in the GFM table,
#: and rather than `0.000` so it can never be read as a measured value.
#:
#: Named `..._RECALL_...` when #123 introduced it for `recall_at_k` alone; it now
#: covers the `embed_latency_ms` columns too, and the name is kept because it is
#: public surface (`test_public_surface.py`).
ABSENT_RECALL_CELL = "—"


def _recall_cell(recall_at_k: dict[int, float], k: int) -> str:
    """One recall cell, distinguishing "measured zero" from "not measured"."""
    if k not in recall_at_k:
        return f" {ABSENT_RECALL_CELL} |"
    return f" {recall_at_k[k]:.3f} |"


def _latency_cell(embed_latency_ms: dict[str, float], key: str, *, places: int) -> str:
    """One latency cell, by the same rule `_recall_cell` applies to recall (#127).

    `#123` fixed `recall_at_k`'s `.get(k, 0.0)` because it "PUBLISHED A NUMBER FOR
    A MEASUREMENT NEVER TAKEN". The three `embed_latency_ms` reads on the same row
    had the identical shape and were not touched — and for latency the fabricated
    default is worse than merely wrong, because `0.0` is the *best possible value*.
    A provider that reported no timings won any "which is fastest" read of the
    published benchmark:

        | b-no-latency | ... | 0.850 |    0 | 0.0 |  0.0 | $0.100 |
        | full-model   | ... | 0.720 | 1234 | 8.1 | 19.4 | $0.130 |

    A default landing at an extreme of a comparison does not abstain, it ranks —
    the same class as `#123` here and `llm-cost-optimizer#190`, reached through a
    different dict.

    Reachable by the path `#123` cites: `SweepResult.from_dict` on an external
    result file, which is what `emb-shootout sweep aggregate` does to every
    `results/*.json`. Verified that `from_dict` accepts both an empty and a
    partial `embed_latency_ms`.
    """
    if key not in embed_latency_ms:
        return f" {ABSENT_RECALL_CELL} |"
    return f" {embed_latency_ms[key]:.{places}f} |"


def _absent_or(value: dict, key: object) -> float | None:
    """`value[key]` or `None` — the JSON spelling of `_recall_cell`'s em dash (#127).

    `aggregate_json` says it is "JSON-shaped aggregation parallel to
    `aggregate_markdown`" and that its rows are sorted alike "so a downstream
    consumer can cross-check the two formats line-by-line". After `#123` fixed the
    markdown side only, that cross-check failed on exactly the cell `#123` was
    about: for a result swept without `k=5`, markdown said `—` and JSON said
    `0.0`. The JSON is the format a CI consumer actually parses.

    `null`, not the em dash: JSON has a spelling for absent and markdown does not,
    and `null` keeps the field numeric-or-null for a typed consumer instead of
    turning a number column into a string column.

    Not "omit the key", either — a missing key and a null key are different
    contracts. A consumer reading `row["recall"]["5"]` would get a `KeyError` from
    omission but a readable `None` from null, and the column set is a property of
    the aggregate (the union of every result's `k`), not of the individual row.
    `test_aggregate_absent_cell_parity.py` pins that every row carries every key.
    """
    v = value.get(key)
    return None if v is None else float(v)


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
    # Splice the per-k recall columns as ONE segment that contributes nothing —
    # to the header, the separator, AND every data row — when `ks` is empty
    # (every result carries an empty `recall_at_k`, reachable via `from_dict` on
    # an external result file; the CLI pipeline validates `k_values` non-empty).
    # The old `| {header_recall} |` wrapper always emitted a cell while the
    # separator's `"|".join(... for _ in ks)` emitted nothing, so an empty `ks`
    # left the header with a phantom column the separator lacked and produced a
    # malformed GFM table (header 10 cols vs separator 9). Building each segment
    # with a trailing `|` per column keeps the three rows column-aligned for any
    # `ks`, and is byte-identical to the prior output when `ks` is non-empty (#83).
    header_recall = "".join(f" recall@{k} |" for k in ks)
    sep_recall = "".join("---:|" for _ in ks)
    lines = [
        f"| embedder | dim | n_corpus | n_queries |{header_recall} NDCG@10 | corpus embed (ms) | query p50 (ms) | query p95 (ms) | $/1M tokens |",
        "|----------|----:|---------:|----------:|"
        + sep_recall
        + "--------:|------------------:|---------------:|---------------:|------------:|",
    ]
    for r in sorted(results, key=lambda x: x.embedder_name):
        # `.get(k, 0.0)` PUBLISHED A NUMBER FOR A MEASUREMENT NEVER TAKEN
        # (#123). `_aggregate_ks` above correctly takes the UNION of k across
        # results, so the column set is right — but a result swept at
        # `k_values=(1, 3, 10)` then reported `0.000` under recall@5, which is
        # indistinguishable from a real measurement of zero. Measured, two
        # results with different k sets:
        #
        #   | mid         | ... | 0.400 | 0.000 | 0.550 | 0.600 |
        #   | strong-no-5 | ... | 0.950 | 0.970 | 0.000 | 0.990 |
        #
        # `mid` was never measured at 3 and `strong-no-5` never at 5; both cells
        # read as measured zeros. Handoff §10: do not invent benchmark numbers.
        # An em dash instead — visible in the column, impossible to read as a
        # value. Sibling of chunking-strategies-lab#160.
        #
        # The CLI hardcodes `k_values=(1, 5, 10)` for every provider, so a
        # normal aggregate never takes this branch and its output is unchanged.
        recalls = "".join(_recall_cell(r.recall_at_k, k) for k in ks)
        # `embedder_name` is the one free-form cell (every other is a formatted
        # number). It reaches here with an arbitrary `|` via `from_dict` on an
        # external/hand-edited result file, or a BYO `Embedder` whose `name`
        # carries one. GFM splits table cells on unescaped pipes, so an unescaped
        # `|` injects a spurious column and corrupts the whole benchmark table's
        # alignment. Escape `|` -> `\|` (GitHub renders `\|` as a literal pipe,
        # contributing zero column delimiters) — same fix as `comment._row_to_md`
        # (rag-kit #130) and `calibration.render_report` (llm-eval-harness #134).
        #
        # A `\n`/`\r` in the same cell is the sibling corruption: a GFM row is a
        # single physical line, so an embedded newline splits one result across
        # two lines and breaks every row after it. The pipe-escape (#79/#80)
        # closed one delimiter class at this site but left the row-delimiter class
        # open; collapse `[\r\n]+` -> a single space, matching the portfolio
        # `md_table_cell` pattern. Same external-input reachability as the pipe
        # (`from_dict` invited hand-edit #75 / BYO embedder name).
        embedder_name = r.embedder_name.replace("|", "\\|")
        embedder_name = re.sub(r"[\r\n]+", " ", embedder_name)
        lines.append(
            f"| {embedder_name} | {r.embedder_dim} | {r.n_corpus} | {r.n_queries} |{recalls} "
            f"{r.ndcg_at_10:.3f} |"
            f"{_latency_cell(r.embed_latency_ms, 'corpus_total', places=0)}"
            f"{_latency_cell(r.embed_latency_ms, 'query_p50', places=1)}"
            f"{_latency_cell(r.embed_latency_ms, 'query_p95', places=1)}"
            f" ${r.cost_per_million_tokens:.3f} |"
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
                # `null`, not `0.0`, for a measurement never taken (#127). See
                # `_absent_or`; the markdown sibling renders the same cells as an
                # em dash, and `test_aggregate_absent_cell_parity.py` asserts the
                # two formats agree on which cells are absent.
                "recall": {str(k): _absent_or(r.recall_at_k, k) for k in ks},
                "ndcg_at_10": r.ndcg_at_10,
                "corpus_embed_ms": _absent_or(r.embed_latency_ms, "corpus_total"),
                "query_p50_ms": _absent_or(r.embed_latency_ms, "query_p50"),
                "query_p95_ms": _absent_or(r.embed_latency_ms, "query_p95"),
                "cost_per_million_tokens": r.cost_per_million_tokens,
            }
        )
    return {"results": rows, "ks": ks}
