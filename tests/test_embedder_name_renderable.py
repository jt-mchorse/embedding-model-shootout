"""`embedder_name` refuses the rest of the set `''` belongs to (#191).

#143 refused `''` at both `SweepResult` seams because it "renders an EMPTY
embedder column into the published README table". Measured on `main`:

- `"   "` was accepted and `sweep aggregate` published `|     | 128 | ...`;
- `"bad\\ud800name"` (what `json.loads` makes of the legal escape) was accepted,
  then `sweep aggregate` died with a raw `UnicodeEncodeError` and `sweep plot`
  with a raw matplotlib `TypeError`, both at exit 1 instead of the documented 2.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emb_shootout.cli import main
from emb_shootout.sweep import SweepResult

BAD = {
    "blank": ("   ", "embedder_name must not be blank"),
    "blank-newline": (" \n\t", "embedder_name must not be blank"),
    "lone-high": ("bad\ud800name", r"embedder_name contains the lone surrogate U\+D800 at index 3"),
    "lone-low": ("\udcff", r"embedder_name contains the lone surrogate U\+DCFF at index 0"),
}
# A surrogate PAIR is not a lone surrogate: in a Python str it is one astral
# codepoint, which encodes fine. Edge/inner spaces are a name, not a blank.
GOOD = ["emoji-\U0001f600", " padded ", "bge small"]


def _kwargs(name: str) -> dict:
    return dict(
        embedder_name=name,
        embedder_dim=128,
        cost_per_million_tokens=0.02,
        n_corpus=10,
        n_queries=5,
        recall_at_k={1: 0.4, 5: 0.8, 10: 0.9},
        ndcg_at_10=0.7,
        embed_latency_ms={"corpus_total": 1.0, "query_p50": 0.1, "query_p95": 0.2},
    )


def _serialized(name: str) -> dict:
    d = SweepResult(**_kwargs("placeholder")).to_dict()
    d["embedder_name"] = name
    return json.loads(json.dumps(d))


@pytest.mark.parametrize("key", sorted(BAD))
def test_direct_construction_refuses(key: str) -> None:
    name, msg = BAD[key]
    with pytest.raises(ValueError, match=msg):
        SweepResult(**_kwargs(name))


@pytest.mark.parametrize("key", sorted(BAD))
def test_from_dict_refuses(key: str) -> None:
    name, msg = BAD[key]
    with pytest.raises(ValueError, match=msg):
        SweepResult.from_dict(_serialized(name))


@pytest.mark.parametrize("name", GOOD)
def test_renderable_names_still_round_trip(name: str) -> None:
    r = SweepResult.from_dict(_serialized(name))
    assert r.embedder_name == name
    assert SweepResult(**_kwargs(name)).embedder_name == name


def _results_dir(tmp_path: Path, name: str) -> Path:
    d = tmp_path / "results"
    d.mkdir()
    # Written exactly as `sweep run` writes a result (json.dumps escapes the
    # surrogate as `\ud800`, which is legal JSON and loads back as a lone one).
    (d / "r.json").write_text(json.dumps(_serialized(name), indent=2), encoding="utf-8")
    return d


@pytest.mark.parametrize("key", ["blank", "lone-high"])
@pytest.mark.parametrize("command", ["aggregate", "plot"])
def test_cli_exits_2_naming_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], key: str, command: str
) -> None:
    results = _results_dir(tmp_path, BAD[key][0])
    if command == "aggregate":
        argv = ["sweep", "aggregate", "--results-dir", str(results)]
        argv += ["--out", str(tmp_path / "bench.md")]
    else:
        argv = ["sweep", "plot", "--results-dir", str(results)]
        argv += ["--out-svg", str(tmp_path / "p.svg")]
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert f"malformed result JSON in {results / 'r.json'}" in err
    assert "embedder_name" in err
    assert not (tmp_path / "bench.md").exists()
