"""`SweepResult` copies its mutable fields in, so `frozen=True` means something (#133).

`frozen=True` refuses `r.recall_at_k = {}` and does nothing about the caller's
own reference to the dict it passed. All three mutable fields were aliased, so
`recall[5] = 999.0` after construction put an out-of-range value into a "frozen"
result and `to_dict` carried it into `results/*.json`, the README table and the
Pareto plot.

That is what makes this more than a style point: `__post_init__` runs five
validation loops over exactly these containers, added across #29/#31/#65, each
with its own written argument about a corrupt number reaching the Pareto
comparator or the published table. Every one of them was a one-time check on a
container someone else could still edit.

`to_dict` on the same class already defended the *opposite* direction and said
so — "`notes` is copied to a fresh list so caller mutation of the returned dict
doesn't bleed back into the frozen dataclass". Out was copied; in was not.

Sibling of `vector-search-at-scale#135` and `python-async-llm-pipelines`
#100/#102; found by triaging `portfolio-ops#71`'s worklist, which is explicit
that a `GAP` is a candidate until reachability is checked.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from emb_shootout import SweepResult

BASE = {
    "embedder_name": "probe",
    "embedder_dim": 4,
    "cost_per_million_tokens": 1.0,
    "n_corpus": 2,
    "n_queries": 1,
    "ndcg_at_10": 0.5,
}


def _result(**overrides: object) -> SweepResult:
    kwargs: dict[str, object] = {
        **BASE,
        "recall_at_k": {1: 0.5},
        "embed_latency_ms": {"corpus_total": 1.0},
        "notes": ["measured"],
        **overrides,
    }
    return SweepResult(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The copy
# ---------------------------------------------------------------------------


def test_frozen_still_refuses_attribute_assignment() -> None:
    """The half that already worked, pinned so the fix isn't credited for it."""
    result = _result()
    with pytest.raises(FrozenInstanceError):
        result.recall_at_k = {}  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "caller_value", "mutate"),
    [
        ("recall_at_k", {1: 0.5}, lambda c: c.__setitem__(5, 0.9)),
        ("embed_latency_ms", {"corpus_total": 1.0}, lambda c: c.__setitem__("query_p50", 2.0)),
        ("notes", ["measured"], lambda c: c.append("fabricated")),
    ],
)
def test_mutating_the_callers_container_does_not_reach_the_instance(
    field: str, caller_value: object, mutate: object
) -> None:
    """One arm per field. A fix that copied two of three would pass a test that
    only checked the field named in the issue."""
    result = _result(**{field: caller_value})
    before = (
        dict(getattr(result, field))
        if isinstance(caller_value, dict)
        else list(getattr(result, field))
    )
    mutate(caller_value)  # type: ignore[operator]
    assert getattr(result, field) is not caller_value
    after = (
        dict(getattr(result, field))
        if isinstance(caller_value, dict)
        else list(getattr(result, field))
    )
    assert after == before


# ---------------------------------------------------------------------------
# What the copy is actually protecting: the guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bad_recall", "why"),
    [(999.0, "the [0, 1] range guard"), (math.nan, "the finiteness guard")],
)
def test_a_post_construction_mutation_cannot_bypass_the_recall_guards(
    bad_recall: float, why: str
) -> None:
    """The guards are one-time checks; aliasing made them advisory.

    Asserted through `to_dict`, not only on the attribute, because `to_dict` is
    what reaches `results/*.json`, the README table and the Pareto plot — the
    place a corrupt number actually costs something.
    """
    recall = {1: 0.5}
    result = _result(recall_at_k=recall)
    recall[1] = bad_recall
    assert result.recall_at_k[1] == 0.5, why
    assert result.to_dict()["recall_at_k"]["1"] == 0.5


def test_a_post_construction_mutation_cannot_bypass_the_latency_guard() -> None:
    latency = {"corpus_total": 1.0}
    result = _result(embed_latency_ms=latency)
    latency["corpus_total"] = -1.0
    assert result.embed_latency_ms["corpus_total"] == 1.0
    assert result.to_dict()["embed_latency_ms"]["corpus_total"] == 1.0


def test_the_guards_still_reject_a_bad_value_at_construction() -> None:
    """Anti-vacuous: copying the container must not have moved the validation
    off the values that reach it."""
    with pytest.raises(ValueError, match=r"recall_at_k\[1\] must be a finite number"):
        _result(recall_at_k={1: 999.0})
    with pytest.raises(ValueError, match="embed_latency_ms"):
        _result(embed_latency_ms={"corpus_total": -1.0})


# ---------------------------------------------------------------------------
# `notes` elements, which are what make the shallow copy complete
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [1, ["nested"], {"k": "v"}, None, 2.5, True])
def test_notes_elements_must_be_strings(bad: object) -> None:
    """`notes: list[str]` was unenforced, and `from_dict` did not add a check —
    measured, `[1, ["nested"], {"k": "v"}]` was accepted and `to_dict`
    round-tripped it into the markdown table."""
    with pytest.raises(ValueError, match=r"notes\[0\] must be a string"):
        _result(notes=[bad])


def test_the_notes_error_names_the_offending_index() -> None:
    with pytest.raises(ValueError, match=r"notes\[2\] must be a string"):
        _result(notes=["a", "b", 3])


def test_a_nested_note_cannot_survive_the_copy() -> None:
    """The case a shallow copy alone would not stop.

    Before the element check, `notes=[inner]` then `inner.append(...)` still
    mutated the instance — one field of a "frozen" result left mutable while the
    other two were fixed. Rejecting non-strings is what makes the copy complete
    by construction rather than by assumption.
    """
    inner = ["mutable"]
    with pytest.raises(ValueError, match=r"notes\[0\] must be a string"):
        _result(notes=[inner])


def test_from_dict_also_rejects_non_string_notes() -> None:
    """The reachable path: `from_dict` is the loader for hand-edited and
    externally-generated results, which is who the other guards exist for."""
    payload = {
        **BASE,
        "recall_at_k": {"1": 0.5},
        "embed_latency_ms": {"corpus_total": 1.0},
        "notes": [1, ["nested"]],
    }
    with pytest.raises(ValueError, match=r"notes\[0\] must be a string"):
        SweepResult.from_dict(payload)


# ---------------------------------------------------------------------------
# Valid input is untouched
# ---------------------------------------------------------------------------


def test_valid_input_round_trips_unchanged() -> None:
    """The contract the fix must not move: `to_dict` output for a valid result
    is what it always was, so no published artifact shifts."""
    payload = {
        **BASE,
        "recall_at_k": {"1": 0.5, "5": 0.8},
        "embed_latency_ms": {"corpus_total": 10.0, "query_p50": 1.5},
        "notes": ["one", "two"],
    }
    result = SweepResult.from_dict(payload)
    assert result.to_dict() == payload
    assert result.notes == ["one", "two"]
    assert result.recall_at_k == {1: 0.5, 5: 0.8}


def test_an_omitted_notes_field_still_defaults_to_empty() -> None:
    result = SweepResult(
        **BASE,  # type: ignore[arg-type]
        recall_at_k={1: 0.5},
        embed_latency_ms={"corpus_total": 1.0},
    )
    assert result.notes == []
    # And the default is per-instance, not shared — `field(default_factory=list)`
    # already guarantees this, but the copy must not have reintroduced sharing.
    other = SweepResult(
        **BASE,  # type: ignore[arg-type]
        recall_at_k={1: 0.5},
        embed_latency_ms={"corpus_total": 1.0},
    )
    assert result.notes is not other.notes
