"""`from_dict` hardens every value and left the key axis at `int(k)` (#129).

Sibling of `chunking-strategies-lab#169`, found by grepping the portfolio for
the `{str(k): v}` write / `{int(k): v}` read pair. Here the contrast is sharper,
because it is inside one function: every *value* on this read path is checked
and named, and the key axis was a bare coercion.

    recall value "NaN"      -> ValueError: recall_at_k[5] must be a finite number in [0, 1]; got nan
    recall value "Infinity" -> ValueError: recall_at_k[5] must be a finite number in [0, 1]; got inf
    latency "inf"           -> ValueError: embed_latency_ms['p50'] must be a finite number >= 0; got inf
    ndcg "nan"              -> ValueError: ndcg_at_10 must be a finite number in [0, 1]; got nan

The `[5]` in those messages is the only place a key was used, and it was already
the coerced one. Measured on `main` before this change::

    recall_at_k in the payload   from_dict     loaded recall_at_k
    CONTROL {"1":.., "5":..}     loaded        {1: 0.5, 5: 0.9}
    {"5": 0.9, "05": 0.1}        loaded        {5: 0.1}     <- 0.9 GONE
    {"5": 0.9, " 5": 0.1}        loaded        {5: 0.1}     <- 0.9 GONE
    {"5": 0.9, "+5": 0.1}        loaded        {5: 0.1}     <- 0.9 GONE
    {"5": 0.9, <Arabic-Indic 5>} loaded        {5: 0.1}     <- 0.9 GONE
    {"5_0": 0.9}                 loaded        {50: 0.9}    <- a typo becomes another k
    {"0": 0.9} / {"-3": 0.9}     loaded

`run_sweep` validates `k_values` in four stages hardened across `#31` and
`#121` — non-empty, `int` with `bool` excluded, positive, no duplicates — and
`#121`'s comment explains that `k=0` "produces a tautological recall@0=0" and
`k<0` "silently miscounts". `from_dict` accepted both from a file, and its own
comment says hand-editing result files is "an explicitly invited workflow (#75)".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from emb_shootout.sweep import SweepResult, validate_k_values

# U+0665 ARABIC-INDIC DIGIT FIVE. `int()` reads it as 5; `str(5)` cannot write it.
ARABIC_FIVE = chr(0x0665)

_BASE: dict[str, Any] = {
    "embedder_name": "hash-64",
    "embedder_dim": 64,
    "cost_per_million_tokens": 0.0,
    "n_corpus": 10,
    "n_queries": 5,
    "recall_at_k": {"1": 0.5, "5": 0.9},
    "ndcg_at_10": 0.7,
    "embed_latency_ms": {"p50": 1.0, "p95": 2.0},
    "notes": [],
}


def _load(recall: dict[str, Any]) -> SweepResult:
    payload = dict(_BASE)
    payload["recall_at_k"] = recall
    return SweepResult.from_dict(json.loads(json.dumps(payload)))


NON_CANONICAL = [
    ("leading zero", "05"),
    ("many leading zeros", "0005"),
    ("leading space", " 5"),
    ("trailing space", "5 "),
    ("leading plus", "+5"),
    ("underscore digit separator", "5_0"),
    ("Arabic-Indic decimal digit", ARABIC_FIVE),
    ("negative zero", "-0"),
]


@pytest.mark.parametrize(("label", "key"), NON_CANONICAL, ids=[r[0] for r in NON_CANONICAL])
def test_a_non_canonical_key_is_rejected_by_field_name(label: str, key: str) -> None:
    assert isinstance(int(key), int)  # precondition: `int()` accepts it
    with pytest.raises(ValueError, match=r"recall_at_k key .* is not the canonical spelling"):
        _load({key: 0.9})


@pytest.mark.parametrize(
    ("label", "colliding_key"),
    [
        ("leading zero", "05"),
        ("leading space", " 5"),
        ("leading plus", "+5"),
        ("Arabic-Indic five", ARABIC_FIVE),
    ],
)
def test_a_measurement_can_no_longer_be_silently_overwritten(
    label: str, colliding_key: str
) -> None:
    """Before #129 this loaded and `recall_at_k` was `{5: 0.1}` — the 0.9 gone
    with no diagnostic anywhere."""
    assert int(colliding_key) == 5
    with pytest.raises(ValueError, match="canonical"):
        _load({"5": 0.9, colliding_key: 0.1})


@pytest.mark.parametrize(
    ("label", "key"), [("non-numeric", "best"), ("float-looking", "5.0"), ("empty", "")]
)
def test_a_non_integer_key_names_the_field_rather_than_int_builtin(label: str, key: str) -> None:
    """These already raised, but with `invalid literal for int() with base 10:
    'best'` — naming neither the field nor what the writer actually produces."""
    with pytest.raises(ValueError, match=r"recall_at_k key .* is not an integer") as exc:
        _load({key: 0.9})
    assert "to_dict" in str(exc.value)


@pytest.mark.parametrize(("label", "key"), [("zero", "0"), ("negative", "-3"), ("negative", "-1")])
def test_a_k_the_write_path_refuses_to_produce_cannot_be_read_back(label: str, key: str) -> None:
    with pytest.raises(ValueError, match=r"recall_at_k keys: .*positive"):
        _load({key: 0.9})


def test_the_range_rule_is_shared_not_restated() -> None:
    """The loader must call `validate_k_values`, the same function `run_sweep`
    calls — not a second copy. Asserted by behaviour: patching it to a no-op
    must let a bad k through the *reader* too."""
    import emb_shootout.sweep as sweep

    original = sweep.validate_k_values
    try:
        sweep.validate_k_values = lambda ks: None  # type: ignore[assignment]
        result = _load({"0": 0.9})
    finally:
        sweep.validate_k_values = original  # type: ignore[assignment]
    assert dict(result.recall_at_k) == {0: 0.9}, (
        "the reader did not go through validate_k_values; the rule is restated somewhere"
    )
    with pytest.raises(ValueError, match="positive"):
        _load({"0": 0.9})


def test_run_sweep_still_uses_the_same_extracted_rule() -> None:
    """Both directions of the shared rule, so the extraction cannot rot into a
    loader-only helper."""
    for bad, pattern in [
        ((), "non-empty"),
        ((0,), "positive"),
        ((-1,), "positive"),
        ((True,), "bool excluded"),
        ((2.5,), "bool excluded"),
        ((1, 1), "duplicates"),
    ]:
        with pytest.raises(ValueError, match=pattern):
            validate_k_values(bad)
    validate_k_values((1, 5, 10))


# ----------------------------------------------------------------------
# Controls — keys only; the value axis and the happy path must not move
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "recall"),
    [
        ("the default k_values", {"1": 0.5, "5": 0.7, "10": 0.9}),
        ("a single k", {"1": 1.0}),
        ("a large k", {"1000": 0.0}),
        ("unordered keys", {"10": 0.9, "1": 0.5}),
        ("boundary values", {"1": 0.0, "5": 1.0}),
    ],
)
def test_canonical_payloads_load_unchanged(label: str, recall: dict[str, Any]) -> None:
    result = _load(recall)
    assert dict(result.recall_at_k) == {int(k): float(v) for k, v in recall.items()}


@pytest.mark.parametrize(
    ("label", "value", "pattern"),
    [
        ("NaN string", "NaN", r"recall_at_k\[5\] must be a finite number"),
        ("Infinity string", "Infinity", r"recall_at_k\[5\] must be a finite number"),
        ("out of range", 1.5, r"recall_at_k\[5\] must be a finite number"),
    ],
)
def test_the_value_axis_is_byte_identical(label: str, value: Any, pattern: str) -> None:
    """This change touches keys only. The value guards — which were already the
    thorough half — must fire exactly as before, message included."""
    with pytest.raises(ValueError, match=pattern):
        _load({"5": value})


def test_the_round_trip_is_still_exact() -> None:
    result = _load({"1": 0.5, "5": 0.7, "10": 0.9})
    once = json.dumps(result.to_dict(), sort_keys=True)
    twice = json.dumps(SweepResult.from_dict(json.loads(once)).to_dict(), sort_keys=True)
    assert once == twice


def test_every_committed_result_artifact_still_loads() -> None:
    """The guard must not reject anything this repo ships."""
    loaded = 0
    for path in sorted(Path().rglob("*.json")):
        if ".venv" in path.parts or "node_modules" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if isinstance(row, dict) and "recall_at_k" in row and "embedder_name" in row:
                result = SweepResult.from_dict(row)
                assert all(k >= 1 for k in result.recall_at_k), path
                loaded += 1
    assert loaded > 0, "no committed SweepResult artifacts found — this test proves nothing"


def test_the_table_is_not_vacuous() -> None:
    """Every non-canonical row must be a key `int()` genuinely accepts, or it
    would be testing the pre-existing `invalid literal` path instead."""
    assert len(NON_CANONICAL) >= 8
    for label, key in NON_CANONICAL:
        int(key)
        assert str(int(key)) != key, f"{label}: {key!r} IS canonical; the row proves nothing"
