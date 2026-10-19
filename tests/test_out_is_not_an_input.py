"""An `--out` that is one of the command's own inputs is refused (#230).

Measured on main (aa06cd9) with a copy of the committed `results/`:

    sweep aggregate --format json --out res/aggregate.json -> rc 0; the next
        aggregate / plot over res/ exits 2 ("malformed result JSON")
    sweep aggregate --out res/hash.json -> rc 0; the measured result is now
        a markdown table
    sweep run --corpus c.jsonl --out c.jsonl -> rc 0; the corpus is gone
    corpus validate c.jsonl --out c.jsonl -> the summary replaces the corpus
    corpus validate c.jsonl --out '' -> report on stdout, rc 0
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from emb_shootout.cli import main

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def res(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shutil.copytree(REPO / "results", tmp_path / "res")
    (tmp_path / "sub").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path / "res"


def _snapshot(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


# --- sweep aggregate ---------------------------------------------------------


@pytest.mark.parametrize(
    "out",
    [
        "res/hash.json",
        "./res/hash.json",
        "sub/../res/hash.json",
        "res/aggregate.json",  # not there yet: the next glob would read it
        "sub/../res/new.json",
    ],
)
@pytest.mark.parametrize("fmt", ["markdown", "json"])
def test_aggregate_refuses_an_out_in_the_result_glob(
    res: Path, out: str, fmt: str, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _snapshot(res)
    assert main(["sweep", "aggregate", "--results-dir", "res", "--format", fmt, "--out", out]) == 2
    assert _snapshot(res) == before
    assert "refusing to write --out" in capsys.readouterr().err
    # The directory is still aggregatable.
    assert main(["sweep", "aggregate", "--results-dir", "res", "--out", "t.md"]) == 0


def test_aggregate_refuses_a_symlink_to_a_result(res: Path) -> None:
    Path("link.json").symlink_to(res / "hash.json")
    before = _snapshot(res)
    assert main(["sweep", "aggregate", "--results-dir", "res", "--out", "link.json"]) == 2
    assert _snapshot(res) == before


def test_aggregate_refuses_a_hard_link_to_a_result(res: Path) -> None:
    os.link(res / "hash.json", "hard.json")
    before = _snapshot(res)
    assert main(["sweep", "aggregate", "--results-dir", "res", "--out", "hard.json"]) == 2
    assert _snapshot(res) == before


def test_aggregate_refuses_a_symlinked_results_dir(res: Path) -> None:
    Path("alias").symlink_to(res)
    before = _snapshot(res)
    out = "alias/aggregate.json"
    assert (
        main(["sweep", "aggregate", "--results-dir", "res", "--format", "json", "--out", out]) == 2
    )
    assert _snapshot(res) == before


@pytest.mark.parametrize("out", ["agg.json", "sub/agg.json", "res/notes.md"])
def test_aggregate_control_an_out_outside_the_glob_still_writes(res: Path, out: str) -> None:
    fmt = "json" if out.endswith(".json") else "markdown"
    assert main(["sweep", "aggregate", "--results-dir", "res", "--format", fmt, "--out", out]) == 0
    assert Path(out).is_file()


# --- sweep run ---------------------------------------------------------------


def _corpus(path: Path) -> Path:
    rows = [
        {"chunk_id": f"c{i:03d}", "text": f"document number {i} discusses topic {i % 3}"}
        for i in range(20)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _run(corpus: str, out: str) -> int:
    return main(
        ["sweep", "run", "--provider", "hash", "--corpus", corpus, "--queries", "3", "--out", out]
    )


@pytest.mark.parametrize("out", ["c.jsonl", "./c.jsonl", "sub/../c.jsonl", "hard.jsonl"])
def test_sweep_run_refuses_the_corpus_as_out(
    res: Path, out: str, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(Path("c.jsonl"))
    os.link(corpus, "hard.jsonl")
    before = corpus.read_bytes()
    assert _run("c.jsonl", out) == 2
    assert corpus.read_bytes() == before
    assert "it is the --corpus file" in capsys.readouterr().err


def test_sweep_run_control_overwriting_a_previous_result_is_allowed(res: Path) -> None:
    _corpus(Path("c.jsonl"))
    assert _run("c.jsonl", "res/hash.json") == 0
    assert json.loads((res / "hash.json").read_text(encoding="utf-8"))["n_corpus"] == 20


# --- corpus validate ---------------------------------------------------------


@pytest.mark.parametrize("out", ["c.jsonl", "./c.jsonl", "sub/../c.jsonl"])
def test_validate_refuses_the_corpus_as_out(res: Path, out: str) -> None:
    corpus = _corpus(Path("c.jsonl"))
    before = corpus.read_bytes()
    assert main(["corpus", "validate", "c.jsonl", "--out", out]) == 2
    assert corpus.read_bytes() == before


@pytest.mark.parametrize("out", ["", "  "])
def test_validate_refuses_an_empty_out(
    res: Path, out: str, capsys: pytest.CaptureFixture[str]
) -> None:
    _corpus(Path("c.jsonl"))
    assert main(["corpus", "validate", "c.jsonl", "--out", out]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--out must be non-empty" in captured.err


def test_validate_control_a_separate_out_still_writes(res: Path) -> None:
    _corpus(Path("c.jsonl"))
    assert main(["corpus", "validate", "c.jsonl", "--out", "report.txt"]) == 0
    assert Path("report.txt").read_text(encoding="utf-8").startswith("ok: c.jsonl")
