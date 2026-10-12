"""`from_dict` reads a result's numbers as JSON numbers, not via bare int()/float() (#206).

`int()` and `float()` accept far more than a number. On main: a JSON
`Infinity` in `n_corpus` was an `OverflowError` no handler caught (`sweep
aggregate` exit 1), and `49.99`, `"5_0"`, `" 12\\n"` and Arabic-Indic digits
loaded as 49, 50, 12 and 42, so `sweep aggregate` published altered values at
exit 0. #129 gave the recall keys this strictness and #156 gave `query_seed`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emb_shootout.cli import main
from emb_shootout.sweep import SweepResult

_RESULTS = Path(__file__).resolve().parent.parent / "results"


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


_REFUSED = [
    ("n_corpus", float("inf")),
    ("n_corpus", 49.99),
    ("n_corpus", "42"),
    ("n_corpus", chr(0x0664) + chr(0x0662)),  # Arabic-Indic 42
    ("embedder_dim", "5_0"),
    ("n_queries", " 12\n"),
    ("cost_per_million_tokens", "1_0"),
    ("ndcg_at_10", "0.4"),
]


@pytest.mark.parametrize(("field", "value"), _REFUSED)
def test_a_non_number_is_refused_by_name(field: str, value: object) -> None:
    d = _base()
    d[field] = value
    with pytest.raises(ValueError, match=field):
        SweepResult.from_dict(d)


def test_a_latency_string_is_refused() -> None:
    d = _base()
    d["embed_latency_ms"] = {"p50": "1_0"}
    with pytest.raises(ValueError, match=r"embed_latency_ms\['p50'\]"):
        SweepResult.from_dict(d)


def test_numbers_still_load() -> None:
    # Control: an int in a float field is a JSON number.
    d = _base()
    d["cost_per_million_tokens"] = 2
    assert SweepResult.from_dict(d).cost_per_million_tokens == 2.0


def test_committed_results_load_unchanged() -> None:
    files = sorted(_RESULTS.glob("*.json"))
    assert files
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert SweepResult.from_dict(data).to_dict() == SweepResult.from_dict(data).to_dict()


@pytest.mark.parametrize("raw", ['"n_corpus": Infinity', '"n_corpus": 1e999', '"n_corpus": 49.99'])
def test_sweep_aggregate_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], raw: str
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    text = json.dumps(_base()).replace('"n_corpus": 10', raw)
    assert raw in text
    (results / "bad.json").write_text(text, encoding="utf-8")
    rc = main(
        ["sweep", "aggregate", "--results-dir", str(results), "--out", str(tmp_path / "a.md")]
    )
    err = capsys.readouterr().err
    assert rc == 2, err
    assert "Traceback" not in err
