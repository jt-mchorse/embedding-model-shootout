"""`ndcg_at_k` validates `relevances`, the operand #31 left unchecked (#125).

`ndcg_at_k` is exported in `__all__` and documents a range. It validated `k`
(#31: `not isinstance(k, int) or isinstance(k, bool) or k <= 0`) and not
`relevances` — the other operand of the same expression. Measured on `main`:

    relevances        k          ndcg   in [0, 1]?
    [1, 1, 0, 0]      4      1.000000   yes
    [0, 0, 1, 1]      4      0.570642   yes
    [0, 0, 0, 0]      4      0.000000   yes
    [3, 2, 1, 0]      4      1.000000   yes    graded was always fine
    [-10, 3]          1     -3.333333   NO
    [-1, 20]          1     -0.050000   NO
    [-1, 1]           2     -1.000000   NO
    [inf, 1]          2           nan   NO
    [nan, 1]          2      0.000000   yes -- and that is the problem
    ["1", "0"]        2     raw TypeError, outside the module's ValueError contract

The NaN row is the quietest and the worst: `_dcg` yields NaN, `ideal > 0` is
`False` for NaN, and the fallback returns a clean `0.000` — indistinguishable
from "nothing relevant was retrieved". Same shape as the extreme-default class
#123 closed, where an unmeasured recall scored `0.0` and was *dominated* on the
Pareto frontier rather than excluded from it.

The decisive argument for validating rather than widening the docstring is that
this repo has already made the call for this exact number:
`SweepResult.__post_init__` enforces `0.0 <= ndcg_at_10 <= 1.0` with a
`ValueError`. The consumer treats the range as a hard contract; the producer
only promised it in prose. Composing the two surfaced the failure as
`ndcg_at_10 must be a finite number in [0, 1]` — a message naming the *field*
rather than the relevance list at fault. `test_the_consumer_contract_is_why`
below pins that relationship.
"""

from __future__ import annotations

import math

import pytest

from emb_shootout.sweep import SweepResult, ndcg_at_k

# ----------------------------------------------------------------------
# Everything that used to leave the documented range
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("relevances", "k", "was"),
    [
        ([-10, 3], 1, "-3.333333"),
        ([-1, 20], 1, "-0.050000"),
        ([-1, 1], 2, "-1.000000"),
        ([-1, 0], 2, "0.000000 via the ideal>0 fallback — a quieter wrong answer"),
        ([-1, -2], 2, "0.000000 via the ideal>0 fallback"),
        ([0, 0, -1], 3, "0.000000 via the ideal>0 fallback"),
    ],
)
def test_a_negative_gain_is_rejected(relevances: list[int], k: int, was: str) -> None:
    with pytest.raises(ValueError, match=r"relevances\[\d+\] must be a non-negative integer gain"):
        ndcg_at_k(list(relevances), k)


def test_the_message_says_why_a_negative_gain_is_refused() -> None:
    """Naming the bound makes the refusal actionable: a caller who passes -1 is
    modelling "worse than irrelevant", and needs to know the metric has no such
    coordinate rather than that "-1 is invalid"."""
    with pytest.raises(ValueError, match=r"relevances\[0\]") as exc:
        ndcg_at_k([-1, 1], 2)
    message = str(exc.value)
    assert "relevances[0]" in message
    assert "[0, 1] bound" in message
    assert "SweepResult" in message


@pytest.mark.parametrize(
    ("relevances", "why"),
    [
        ([float("nan"), 1], "returned a clean 0.000, indistinguishable from 'nothing relevant'"),
        ([float("inf"), 1], "returned nan"),
        ([0.5, 0.5], "a float gain, where the annotation says list[int]"),
        ([True, False], "bool is an int subclass and must not pose as a gain"),
        (["1", "0"], "raw TypeError from the division, outside the ValueError contract"),
        ([None, 1], "raw TypeError from the division"),
    ],
)
def test_a_non_integer_gain_is_rejected(relevances: list, why: str) -> None:
    with pytest.raises(ValueError, match=r"relevances\[\d+\] must be a non-negative integer gain"):
        ndcg_at_k(list(relevances), 2)


def test_the_index_of_the_offending_element_is_reported() -> None:
    """A relevance list is 10 long by default in `run_sweep`; "one of them is
    bad" is not a usable diagnostic."""
    with pytest.raises(ValueError, match=r"relevances\[3\]"):
        ndcg_at_k([1, 0, 1, -1, 0], 5)


# ----------------------------------------------------------------------
# Everything that worked must keep working
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("relevances", "k", "expected"),
    [
        ([1, 1, 0, 0], 4, 1.0),
        ([0, 0, 0, 0], 4, 0.0),
        ([3, 2, 1, 0], 4, 1.0),  # the acceptance criterion: graded still works
        ([1, 0], 10, 1.0),  # k past the end of the list
        ([], 3, 0.0),
    ],
)
def test_valid_input_is_unchanged(relevances: list[int], k: int, expected: float) -> None:
    assert ndcg_at_k(list(relevances), k) == pytest.approx(expected)


def test_the_worst_binary_ordering_still_scores_what_it_did() -> None:
    assert ndcg_at_k([0, 0, 1, 1], 4) == pytest.approx(0.5706419, rel=1e-6)


@pytest.mark.parametrize(
    "relevances",
    [[0], [1], [0, 0, 0], [1, 1, 1], [3, 2, 1, 0], [0, 1, 2, 3], [5, 0, 5, 0], [2, 2, 2, 2]],
)
def test_every_accepted_input_lands_inside_the_documented_range(relevances: list[int]) -> None:
    """The property the docstring claims, asserted over the domain the guard now
    admits — rather than over a couple of hand-picked lists."""
    for k in range(1, len(relevances) + 3):
        score = ndcg_at_k(list(relevances), k)
        assert math.isfinite(score)
        assert 0.0 <= score <= 1.0, f"{relevances!r} at k={k} scored {score}"


def test_graded_relevance_is_bounded_because_ideal_is_a_real_maximum() -> None:
    """`ideal` is computed over the *sorted* list, so a graded ranking cannot
    exceed it. This is why the bound is a property of non-negativity rather than
    of binariness — the fact the old docstring understated."""
    assert ndcg_at_k([3, 2, 1, 0], 4) == pytest.approx(1.0)
    assert ndcg_at_k([0, 1, 2, 3], 4) < 1.0
    assert ndcg_at_k([0, 1, 2, 3], 4) > 0.0


# ----------------------------------------------------------------------
# Why validation, and not a wider docstring
# ----------------------------------------------------------------------


def test_the_consumer_contract_is_why() -> None:
    """`SweepResult` already refuses what `ndcg_at_k` used to be able to produce.

    This pins the read-vs-write asymmetry that motivated the fix: the consumer
    enforces `[0, 1]` with a `ValueError` while the producer only promised it in
    prose, so a caller composing them got a message naming the *field* rather
    than the relevance list at fault.
    """
    for bad in (-3.333333, float("nan"), float("inf"), 1.5):
        with pytest.raises(ValueError, match=r"ndcg_at_10"):
            SweepResult(
                embedder_name="e",
                embedder_dim=4,
                cost_per_million_tokens=0.0,
                n_corpus=1,
                n_queries=1,
                recall_at_k={1: 1.0},
                ndcg_at_10=bad,
                embed_latency_ms={"corpus_total": 1.0},
            )


def test_the_producer_can_no_longer_emit_what_the_consumer_refuses() -> None:
    """The two ends now agree, which is the whole point of the change."""
    for relevances in ([1, 1, 0, 0], [0, 0, 1, 1], [3, 2, 1, 0], [0, 0, 0, 0]):
        score = ndcg_at_k(list(relevances), 10)
        SweepResult(
            embedder_name="e",
            embedder_dim=4,
            cost_per_million_tokens=0.0,
            n_corpus=1,
            n_queries=1,
            recall_at_k={1: 1.0},
            ndcg_at_10=score,
            embed_latency_ms={"corpus_total": 1.0},
        )


def test_run_sweeps_own_relevances_still_pass_the_guard() -> None:
    """`run_sweep` derives relevance by membership, so it is binary by
    construction — but "the internal caller is fine" is exactly the kind of
    claim worth executing rather than asserting in prose."""
    retrieved_ids = ["c3", "c1", "c7", "c2"]
    expected = "c7"
    rels = [1 if cid == expected else 0 for cid in retrieved_ids[:10]]
    assert ndcg_at_k(rels, 10) == pytest.approx(1 / math.log2(4))
