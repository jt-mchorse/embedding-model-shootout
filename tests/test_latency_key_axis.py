"""`embed_latency_ms` keys are refused unless they are `str` (#222).

#143 guarded the `recall_at_k` key axis and deferred this one on the ground that
"the field is already `dict[str, float]`, so there is no write/read asymmetry".
The annotation was never enforced, and `json.dumps` is the stringification.
Measured on `main` before this change::

    embed_latency_ms     to_dict -> json -> from_dict
    {"query_p50": 1.0}   identical                      (control)
    {1: 1.0}             reloads {'1': 1.0}     NOT identical, no error
    {None: 1.0}          reloads {'null': 1.0}  NOT identical, no error
    {True: 1.0}          reloads {'true': 1.0}  NOT identical, no error
    {(1,): 1.0}          TypeError from json.dumps -- the writer, not the guard

The round-trip is asserted by VALUE, not by the absence of an exception: three of
the four bad rows round-trip without raising and are still wrong.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from emb_shootout.sweep import SweepResult

_BASE: dict[str, Any] = {
    "embedder_name": "hash-64",
    "embedder_dim": 64,
    "cost_per_million_tokens": 0.0,
    "n_corpus": 10,
    "n_queries": 5,
    "recall_at_k": {1: 0.5, 5: 0.9},
    "ndcg_at_10": 0.7,
}


def _build(latency: dict[Any, float]) -> SweepResult:
    return SweepResult(**_BASE, embed_latency_ms=latency)


def _round_trip(r: SweepResult) -> SweepResult:
    return SweepResult.from_dict(json.loads(json.dumps(r.to_dict())))


REFUSED = [
    ("int key", {1: 1.0}),
    ("None key", {None: 1.0}),
    ("bool key", {True: 1.0}),
    ("float key", {1.5: 1.0}),
    ("tuple key", {(1,): 1.0}),
    ("one bad key among good ones", {"corpus_total": 3.0, "query_p50": 1.0, 0: 2.0}),
]


@pytest.mark.parametrize(("label", "latency"), REFUSED, ids=[r[0] for r in REFUSED])
def test_a_non_str_key_is_refused_by_field_name(label: str, latency: dict[Any, float]) -> None:
    with pytest.raises(ValueError, match=r"embed_latency_ms key must be a str"):
        _build(latency)


ACCEPTED = [
    ("the three run_sweep keys", {"corpus_total": 3.0, "query_p50": 1.0, "query_p95": 2.0}),
    ("a partial dict (absent cells, #127)", {"query_p50": 1.0}),
    ("empty", {}),
    ("other str names stay accepted", {"p50": 1.0, "p95": 2.0}),
]


@pytest.mark.parametrize(("label", "latency"), ACCEPTED, ids=[r[0] for r in ACCEPTED])
def test_every_accepted_latency_dict_round_trips_identically(
    label: str, latency: dict[str, float]
) -> None:
    r = _build(latency)
    reloaded = _round_trip(r)
    assert reloaded == r
    assert reloaded.embed_latency_ms == latency
