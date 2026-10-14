"""A JSON integer too large for a float is refused, not an OverflowError (#208).

#206 made `from_dict` type-check each result number before `float()`, and its
repro refused the float spellings `Infinity` / `1e999`. The integer spelling of
the same out-of-range value -- `1` followed by 400 zeros, which `json.loads`
returns as an `int` -- passed `_json_number`'s type check, and `float()` raised
`OverflowError`: `sweep aggregate` / `sweep plot` exited 1 with a traceback.
`math.isfinite` converts an int the same way, so direct construction raised it
too, from `__post_init__` and from the providers' shared `is_non_negative_finite`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from emb_shootout._argcheck import is_finite_number, is_non_negative_finite
from emb_shootout.cli import main
from emb_shootout.sweep import SweepResult

#: Past the largest double (~1.8e308) but far inside CPython's 4300-digit
#: int-parse limit, so `json.loads` reads it as an ordinary `int`.
_HUGE = 10**400
#: The largest int that still converts: the control the guard must not refuse.
_FITS = int(sys.float_info.max)


def _base() -> dict:
    return {
        "embedder_name": "h",
        "embedder_dim": 8,
        "cost_per_million_tokens": 0.0,
        "n_corpus": 10,
        "n_queries": 5,
        "recall_at_k": {"5": 0.5},
        "ndcg_at_10": 0.4,
        "embed_latency_ms": {"p50": 1.0},
    }


def _set(d: dict, field: str, value: object) -> dict:
    if field == "recall_at_k":
        d["recall_at_k"] = {"5": value}
    elif field == "embed_latency_ms":
        d["embed_latency_ms"] = {"p50": value}
    else:
        d[field] = value
    return d


_FLOAT_FIELDS = ["cost_per_million_tokens", "ndcg_at_10", "recall_at_k", "embed_latency_ms"]


@pytest.mark.parametrize("field", _FLOAT_FIELDS)
def test_from_dict_refuses_by_name(field: str) -> None:
    d = json.loads(json.dumps(_set(_base(), field, "HUGE")).replace('"HUGE"', str(_HUGE)))
    with pytest.raises(ValueError, match=field.replace("[", r"\[")):
        SweepResult.from_dict(d)


@pytest.mark.parametrize("command", ["aggregate", "plot"])
@pytest.mark.parametrize("field", _FLOAT_FIELDS)
def test_cli_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str, command: str
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    text = json.dumps(_set(_base(), field, "HUGE")).replace('"HUGE"', str(_HUGE))
    (results / "bad.json").write_text(text, encoding="utf-8")
    if command == "aggregate":
        argv = [
            "sweep",
            "aggregate",
            "--results-dir",
            str(results),
            "--out",
            str(tmp_path / "a.md"),
        ]
    else:
        argv = [
            "sweep",
            "plot",
            "--results-dir",
            str(results),
            "--out-svg",
            str(tmp_path / "p.svg"),
        ]
    rc = main(argv)
    err = capsys.readouterr().err
    assert rc == 2, err
    assert "bad.json" in err
    assert "Traceback" not in err


def _direct(**overrides: object) -> SweepResult:
    kwargs: dict = {
        "embedder_name": "h",
        "embedder_dim": 8,
        "cost_per_million_tokens": 0.0,
        "n_corpus": 10,
        "n_queries": 5,
        "recall_at_k": {5: 0.5},
        "ndcg_at_10": 0.4,
        "embed_latency_ms": {"p50": 1.0},
    }
    kwargs.update(overrides)
    return SweepResult(**kwargs)


@pytest.mark.parametrize(
    "overrides",
    [
        {"cost_per_million_tokens": _HUGE},
        {"ndcg_at_10": _HUGE},
        {"recall_at_k": {5: _HUGE}},
        {"embed_latency_ms": {"p50": _HUGE}},
    ],
    ids=_FLOAT_FIELDS,
)
def test_direct_construction_raises_value_error(overrides: dict) -> None:
    with pytest.raises(ValueError, match="finite"):
        _direct(**overrides)


def test_predicates_answer_instead_of_raising() -> None:
    assert is_finite_number(_HUGE) is False
    assert is_finite_number(-_HUGE) is False
    assert is_non_negative_finite(_HUGE) is False


def test_an_int_that_fits_a_float_still_loads() -> None:
    # Control: the guard is "does not convert", not "is large".
    assert is_finite_number(_FITS) is True
    assert is_non_negative_finite(_FITS) is True
    d = _set(_base(), "embed_latency_ms", _FITS)
    assert SweepResult.from_dict(d).embed_latency_ms["p50"] == float(_FITS)
    assert _direct(cost_per_million_tokens=_FITS).cost_per_million_tokens == _FITS
