"""`run_sweep`'s `k_values` gate checked sign and duplicates, not type (#121).

The complete rule already lives **twice** in `emb_shootout/sweep.py` —
`ndcg_at_k` and `retrieve_top_k` both carry
`not isinstance(k, int) or isinstance(k, bool) or k <= 0` from #31. Neither
could be reached with the operator's actual value, because `retrieve_top_k` is
called with `max(max_k, 10)`, and `max(True, 10)` is `10`, `max(2.5, 10)` is
`10`. The floor launders the bad type away before the only type guard in the
file can see it, and the failure re-emerges from `retrieved_ids[:k]` in the
inner loop as a raw `TypeError` naming nothing.

Measured on `main` @ 280fa6b:

    k_values=(1,5,10)  OK  {1: 0.5, 5: 1.0, 10: 1.0}  round-trip OK
    k_values=(True,)   OK  {True: 0.5} -> json {"True": 0.5}
                           from_json ValueError: invalid literal for int() with base 10: 'True'
    k_values=(1,True)  ValueError: k_values must not contain duplicates  <- by accident
    k_values=(2.5,)    TypeError: slice indices must be integers or None ...
    k_values=(20.5,)   ValueError: k must be a positive integer; got 20.5
    k_values=(3.0,)    TypeError: slice indices must be integers or None ...
    k_values=(30.0,)   ValueError: k must be a positive integer; got 30.0
    k_values=('3',)    TypeError: '<=' not supported between 'str' and 'int'
    k_values=(None,)   TypeError: '<=' not supported between 'NoneType' and 'int'

The `(2.5,)` vs `(20.5,)` pair is the sharpest row: the same defect, and the
diagnostic differs only by which side of 10 the typo falls on. The default
`k_values` is `(1, 5, 10)`, so an operator staying near the defaults sits
entirely inside the range with the worse one.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from emb_shootout import HashEmbedderProvider
from emb_shootout.sweep import CorpusChunk, Query, SweepResult, run_sweep

NAN = float("nan")
INF = float("inf")


def _corpus() -> list[CorpusChunk]:
    return [
        CorpusChunk(chunk_id=f"c{i}", text=f"alpha beta gamma text number {i} lorem ipsum dolor")
        for i in range(6)
    ]


def _queries() -> list[Query]:
    return [
        Query(query_id="q1", text="alpha beta", expected_chunk_id="c0"),
        Query(query_id="q2", text="number 3", expected_chunk_id="c3"),
    ]


def _sweep(k_values: Any) -> SweepResult:
    return run_sweep(_corpus(), _queries(), embedder=HashEmbedderProvider(), k_values=k_values)


# ----------------------------------------------------------------------
# The writer must not be able to produce a result file its own reader rejects
# ----------------------------------------------------------------------


def test_bool_k_no_longer_produces_a_payload_from_dict_rejects() -> None:
    # Pre-fix the sweep COMPLETED. `retrieved_ids[:True]` takes one element, so
    # it was a mislabelled recall@1, written under the JSON key "True" — which
    # `SweepResult.from_dict` then refuses with `invalid literal for int() with
    # base 10`. `_aggregate_ks` unions `recall_at_k` keys straight into the
    # markdown header, so the committed comparison table would have grown a
    # `recall@True` column no consumer can read back.
    with pytest.raises(ValueError, match=r"must be an int \(bool excluded\)"):
        _sweep((True,))


def test_the_default_k_values_still_round_trip_which_is_the_property_protected() -> None:
    # The invariant stated positively: whatever `run_sweep` writes,
    # `from_dict` reads. Asserted on the KEYS after a real JSON encode/decode
    # rather than on the recall values, which depend on HashEmbedderProvider's
    # scores and would make this fail for an unrelated reason if the fixture
    # changed.
    result = _sweep((1, 5, 10))
    payload = json.loads(json.dumps(result.to_dict()))
    assert sorted(payload["recall_at_k"]) == ["1", "10", "5"]
    restored = SweepResult.from_dict(payload)
    assert restored.recall_at_k == result.recall_at_k


# ----------------------------------------------------------------------
# The max(max_k, 10) laundering — the sharpest part
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("below", "above"), [(2.5, 20.5), (3.0, 30.0), (1.5, 11.5)])
def test_the_same_bad_value_now_fails_the_same_way_on_both_sides_of_ten(
    below: float, above: float
) -> None:
    # This is the defect, not a symptom. `retrieve_top_k(..., max(max_k, 10))`
    # meant a bad `k` at or below 10 never reached the module's own type guard,
    # so it surfaced as `TypeError: slice indices must be integers` from
    # `retrieved_ids[:k]`, while the identical mistake above 10 got a clean
    # `ValueError: k must be a positive integer`. Both must now report
    # identically, from the gate.
    with pytest.raises(ValueError, match=r"must be an int \(bool excluded\)") as low:
        _sweep((below,))
    with pytest.raises(ValueError, match=r"must be an int \(bool excluded\)") as high:
        _sweep((above,))
    assert "k_values" in str(low.value)
    assert "k_values" in str(high.value)
    assert repr(below) in str(low.value)
    assert repr(above) in str(high.value)


def test_3_point_0_is_rejected_because_it_is_the_json_shape_of_an_integer() -> None:
    # The reachable float. `json.loads("3.0")` is `3.0`, so a `k_values` list
    # from a config file or notebook cell carries floats without anyone typing a
    # decimal point. Rejected rather than coerced, so this gate agrees with the
    # two downstream guards (`ndcg_at_k`, `retrieve_top_k`) that already reject
    # it; the message names the coercion.
    with pytest.raises(ValueError, match=r"coerce with int\(k\)"):
        _sweep(tuple(json.loads("[3.0]")))


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_non_finite_k_now_reports_from_the_gate_naming_k_values(bad: float) -> None:
    # Pre-fix these WERE caught — but downstream, in `ndcg_at_k` /
    # `retrieve_top_k`, whose message names `k` rather than `k_values`, because a
    # non-finite value clears the `max(..., 10)` floor. Correct outcome, wrong
    # frame and wrong noun.
    with pytest.raises(ValueError, match=r"k_values must be an int|k in k_values must be an int"):
        _sweep((bad,))


@pytest.mark.parametrize("bad", ["3", None, [], {}])
def test_non_numeric_k_raises_ValueError_not_TypeError(bad: object) -> None:
    # Pre-fix these escaped from the `k <= 0` comparison itself as
    # `TypeError: '<=' not supported between instances of 'str' and 'int'` — a
    # different exception class from every other rejection this gate makes.
    with pytest.raises(ValueError, match=r"must be an int \(bool excluded\)"):
        _sweep((bad,))


def test_every_offender_is_reported_in_one_pass() -> None:
    # Matching the two checks after it, both of which say "surface every bad
    # value in one pass so operators don't chase them one at a time".
    with pytest.raises(ValueError, match=r"must be an int") as excinfo:
        _sweep((1, "x", None, 2.5))
    msg = str(excinfo.value)
    assert "'x'" in msg
    assert "None" in msg
    assert "2.5" in msg


def test_the_type_check_runs_before_the_sign_check() -> None:
    # Ordering is load-bearing: `k <= 0` raises TypeError on a str/None element,
    # so the sign check cannot run until the non-numerics are gone. A list with
    # both problems must report the type.
    with pytest.raises(ValueError, match=r"must be an int \(bool excluded\)"):
        _sweep((0, None))


def test_1_and_True_was_previously_caught_only_by_the_duplicate_check() -> None:
    # Recorded so nobody reads that as type coverage. `list((1, True)).count(1)`
    # is 2 because `True == 1`, so the duplicate guard — written for an entirely
    # different purpose (recall > 1.0 from double-counted hits, #?) — happened to
    # reject this one input. It never covered `(True,)` alone or `(2, True)`.
    with pytest.raises(ValueError, match=r"must be an int \(bool excluded\)"):
        _sweep((1, True))
    with pytest.raises(ValueError, match=r"must be an int \(bool excluded\)"):
        _sweep((2, True))
    assert [1, True].count(1) == 2, "the accident this test exists to document"


# ----------------------------------------------------------------------
# What must not change
# ----------------------------------------------------------------------


def test_the_existing_empty_sign_and_duplicate_guards_are_untouched() -> None:
    with pytest.raises(ValueError, match="k_values must be non-empty"):
        _sweep(())
    with pytest.raises(ValueError, match=r"every k in k_values must be positive; got \[0\]"):
        _sweep((0,))
    with pytest.raises(ValueError, match=r"every k in k_values must be positive; got \[-1\]"):
        _sweep((-1,))
    with pytest.raises(
        ValueError, match=r"k_values must not contain duplicates; got duplicate \[1\]"
    ):
        _sweep((1, 1))


def test_ordinary_int_k_values_still_sweep() -> None:
    for ks in [(1,), (1, 5, 10), (10, 2, 7), (1, 2, 3, 4, 5)]:
        result = _sweep(ks)
        assert sorted(result.recall_at_k) == sorted(ks)
        assert all(0.0 <= v <= 1.0 for v in result.recall_at_k.values())


def test_a_k_above_the_corpus_size_is_still_allowed() -> None:
    # Not the same class: `k` larger than the corpus is a legitimate request
    # (recall@100 on a 6-chunk corpus is 1.0 if the answer is anywhere), and
    # the slice handles it. Only the *type* domain was widened.
    result = _sweep((100,))
    assert result.recall_at_k[100] == 1.0
