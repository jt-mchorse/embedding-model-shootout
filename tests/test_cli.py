"""CLI smoke tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from emb_shootout.cli import main


def test_cli_corpus_build_subset(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    out = tmp_path / "corpus.jsonl"
    rc = main(
        [
            "corpus",
            "build",
            "--out",
            str(out),
            "--module",
            "json",
            "--module",
            "math",
        ]
    )
    assert rc == 0
    assert out.exists()
    captured = capsys.readouterr().out
    summary = json.loads(captured)
    assert summary["chunk_count"] > 0
    assert summary["out"] == str(out)
    # Every line of the output JSONL must parse.
    for line in out.read_text(encoding="utf-8").splitlines():
        json.loads(line)


def test_cli_invoked_as_module(tmp_path: Path):
    """`python -m emb_shootout.cli ...` works (entry point for the README)."""
    out = tmp_path / "corpus.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "emb_shootout.cli",
            "corpus",
            "build",
            "--out",
            str(out),
            "--module",
            "math",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    assert summary["chunk_count"] > 0


# ---------------------------------------------------------------------------
# Issue #75: CLI paths must translate malformed input / write failure into a
# clean `error:` + exit 2, not a raw traceback at exit 1. Each case is
# hermetic and deterministic. Confirmed failing pre-fix.
# ---------------------------------------------------------------------------


def _valid_result_dict(name: str = "hash") -> dict:
    return {
        "embedder_name": name,
        "embedder_dim": 8,
        "cost_per_million_tokens": 0.0,
        "n_corpus": 2,
        "n_queries": 1,
        "recall_at_k": {"1": 1.0, "5": 1.0, "10": 1.0},
        "ndcg_at_10": 1.0,
        "embed_latency_ms": {"mean": 1.0},
        "notes": [],
    }


def test_sweep_run_missing_field_corpus_row_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    # A row that is valid JSON but missing `chunk_id` previously raised a raw
    # KeyError in _read_corpus_jsonl (the sweep run path skips validate_corpus).
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text": "hi"}\n', encoding="utf-8")
    rc = main(
        [
            "sweep",
            "run",
            "--provider",
            "hash",
            "--corpus",
            str(corpus),
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "chunk_id" in err
    assert f"{corpus}:1" in err
    assert "Traceback" not in err


def test_sweep_run_invalid_json_corpus_row_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("{not valid json\n", encoding="utf-8")
    rc = main(
        [
            "sweep",
            "run",
            "--provider",
            "hash",
            "--corpus",
            str(corpus),
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "invalid JSON" in err
    assert "Traceback" not in err


@pytest.mark.parametrize("bad_n", ["0", "-5"])
def test_sweep_run_nonpositive_queries_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], bad_n: str
):
    # #81: `build_queries`/`run_sweep` sit outside the corpus-read try/except,
    # so a degenerate `--queries 0`/negative used to escape as a raw ValueError
    # traceback at exit 1. It must translate to a clean `error:` + exit 2 like
    # every other sweep CLI seam (#75/#77).
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"chunk_id": "c-0", "text": "alpha beta gamma delta epsilon"}\n'
        '{"chunk_id": "c-1", "text": "zeta eta theta iota kappa lambda"}\n',
        encoding="utf-8",
    )
    rc = main(
        [
            "sweep",
            "run",
            "--provider",
            "hash",
            "--corpus",
            str(corpus),
            "--queries",
            bad_n,
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "positive integer" in err
    assert "Traceback" not in err
    assert not (tmp_path / "out.json").exists(), "no output should be written on usage error"


def test_sweep_aggregate_malformed_result_json_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    results = tmp_path / "results"
    results.mkdir()
    (results / "good.json").write_text(json.dumps(_valid_result_dict("bge")), encoding="utf-8")
    (results / "bad.json").write_text("{truncated", encoding="utf-8")
    rc = main(
        ["sweep", "aggregate", "--results-dir", str(results), "--out", str(tmp_path / "agg.md")]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "bad.json" in err
    assert "Traceback" not in err


def test_sweep_plot_malformed_result_json_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    # #77: the plot path parsed results with a bare comprehension, so a
    # malformed result file raised a raw JSONDecodeError at exit 1 — sibling
    # gap to the #75 aggregate fix. The malformed check fires before
    # render_pareto, so this is hermetic regardless of whether matplotlib is
    # installed. Confirmed failing pre-fix.
    results = tmp_path / "results"
    results.mkdir()
    (results / "good.json").write_text(json.dumps(_valid_result_dict("bge")), encoding="utf-8")
    (results / "bad.json").write_text("{truncated", encoding="utf-8")
    rc = main(
        ["sweep", "plot", "--results-dir", str(results), "--out-png", str(tmp_path / "p.png")]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "bad.json" in err
    assert "Traceback" not in err


def test_sweep_plot_unreadable_result_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    # The OSError arm of the same guard: a result path that cannot be read
    # exits 2 with `failed to read`, not a raw traceback. Use a directory named
    # like a result file so read_text raises OSError (IsADirectoryError).
    results = tmp_path / "results"
    results.mkdir()
    (results / "adir.json").mkdir()
    rc = main(
        ["sweep", "plot", "--results-dir", str(results), "--out-png", str(tmp_path / "p.png")]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "failed to read" in err
    assert "Traceback" not in err


def test_sweep_aggregate_unwritable_out_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    results = tmp_path / "results"
    results.mkdir()
    (results / "good.json").write_text(json.dumps(_valid_result_dict("bge")), encoding="utf-8")
    # --out is an existing directory: os.replace onto a dir raises OSError.
    unwritable = tmp_path / "outdir"
    unwritable.mkdir()
    rc = main(["sweep", "aggregate", "--results-dir", str(results), "--out", str(unwritable)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "failed to write" in err
    assert "Traceback" not in err


def test_corpus_validate_unwritable_out_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"chunk_id": "c1", "text": "hello"}) + "\n", encoding="utf-8")
    unwritable = tmp_path / "outdir"
    unwritable.mkdir()
    rc = main(["corpus", "validate", str(corpus), "--out", str(unwritable)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "failed to write" in err
    assert "Traceback" not in err


def test_sweep_aggregate_happy_path_unaffected(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    # Over-rejection guard: a clean aggregate still exits 0 and writes the file.
    results = tmp_path / "results"
    results.mkdir()
    (results / "a.json").write_text(json.dumps(_valid_result_dict("bge")), encoding="utf-8")
    (results / "b.json").write_text(json.dumps(_valid_result_dict("hash")), encoding="utf-8")
    out = tmp_path / "agg.md"
    rc = main(["sweep", "aggregate", "--results-dir", str(results), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert "Traceback" not in capsys.readouterr().err
