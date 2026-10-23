"""An `embedder_name` cannot carry the table marker into the region (#254).

`splice_markdown_table` found the region's end with
``existing.find(TABLE_END_MARKER, begin)``, and `aggregate_markdown` wrote the
name verbatim into its cell. A name containing the end marker therefore *was*
the end of the region on the next run. Measured on `main` 32cb88d, with
`results/hash.json` plus a copy named ``evil <!-- emb-shootout:table:end --> x``
and a copy of the 53-line `docs/benchmarks.md`::

    sweep aggregate (1) -> rc=0  54 lines, end markers: 2
    sweep aggregate (2) -> rc=0  56 lines, end markers: 3
    sweep aggregate (3) -> rc=0  58 lines, end markers: 4

and after deleting the evil result file the region still held the fragment
rows and the extra markers. The splice docstring's own reason for owning a
region -- it "has to be able to *shrink*" -- was what broke.

Two halves: the cell escapes `&` and `<` (GFM renders `&lt;` as `<`), and a
destination already holding more than one marker is refused at exit 2 and left
untouched instead of grown.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from emb_shootout.cli import main
from emb_shootout.sweep import (
    TABLE_BEGIN_MARKER,
    TABLE_END_MARKER,
    SweepResult,
    aggregate_markdown,
    splice_markdown_table,
)

REPO = Path(__file__).resolve().parent.parent
EVIL = f"evil {TABLE_END_MARKER} x"


def _workspace(tmp_path: Path, *, extra_name: str | None) -> tuple[Path, Path]:
    """Scratch `results/` (the committed hash run, optionally a renamed copy) and output."""
    results = tmp_path / "results"
    shutil.copytree(REPO / "results", results)
    if extra_name is not None:
        data = json.loads((results / "hash.json").read_text(encoding="utf-8"))
        data["embedder_name"] = extra_name
        (results / "extra.json").write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "benchmarks.md"
    shutil.copy(REPO / "docs" / "benchmarks.md", out)
    return results, out


def _aggregate(results: Path, out: Path) -> int:
    return main(["sweep", "aggregate", "--results-dir", str(results), "--out", str(out)])


def _hash_result(name: str) -> SweepResult:
    data = json.loads((REPO / "results" / "hash.json").read_text(encoding="utf-8"))
    data["embedder_name"] = name
    return SweepResult.from_dict(data)


def _row(name: str) -> str:
    """The data row `aggregate_markdown` renders for one result named *name*."""
    return aggregate_markdown([_hash_result(name)]).splitlines()[2]


# --- red against the unfixed code -------------------------------------------


def test_a_marker_name_gives_the_same_file_on_every_run(tmp_path: Path) -> None:
    results, out = _workspace(tmp_path, extra_name=EVIL)
    texts = []
    for _ in range(3):
        assert _aggregate(results, out) == 0
        texts.append(out.read_text(encoding="utf-8"))
    assert texts[0] == texts[1] == texts[2]
    assert texts[0].count(TABLE_BEGIN_MARKER) == 1
    assert texts[0].count(TABLE_END_MARKER) == 1


def test_removing_the_result_file_shrinks_the_region_back(tmp_path: Path) -> None:
    results, out = _workspace(tmp_path, extra_name=EVIL)
    for _ in range(2):
        assert _aggregate(results, out) == 0
    (results / "extra.json").unlink()
    assert _aggregate(results, out) == 0

    clean_results, clean_out = _workspace(tmp_path / "clean", extra_name=None)
    assert _aggregate(clean_results, clean_out) == 0
    assert out.read_text(encoding="utf-8") == clean_out.read_text(encoding="utf-8")


def test_the_cell_escapes_lt_and_amp() -> None:
    row = _row("a<b & c")
    assert row.startswith("| a&lt;b &amp; c |"), row
    assert "<" not in _row(EVIL)


def test_a_polluted_destination_is_refused_and_left_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results, out = _workspace(tmp_path, extra_name=None)
    text = out.read_text(encoding="utf-8")
    polluted = text.replace(TABLE_END_MARKER, f"{TABLE_END_MARKER}\nstale row\n{TABLE_END_MARKER}")
    out.write_text(polluted, encoding="utf-8")

    assert _aggregate(results, out) == 2
    assert out.read_text(encoding="utf-8") == polluted
    err = capsys.readouterr().err
    assert "the generated region is malformed" in err
    assert str(out) in err


@pytest.mark.parametrize(
    "existing",
    [
        f"H\n{TABLE_BEGIN_MARKER}\nold\n{TABLE_END_MARKER}\nx\n{TABLE_END_MARKER}\nT\n",
        f"H\n{TABLE_BEGIN_MARKER}\nold\n{TABLE_BEGIN_MARKER}\nx\n{TABLE_END_MARKER}\nT\n",
    ],
    ids=["two-end-markers", "two-begin-markers"],
)
def test_splice_refuses_more_than_one_marker(existing: str) -> None:
    with pytest.raises(ValueError, match="the generated region is malformed"):
        splice_markdown_table(existing, "TABLE\n")


# --- controls: green before and after ---------------------------------------


def test_an_ordinary_name_renders_unchanged() -> None:
    assert _row("hash-embedder-128d-ngram2").startswith("| hash-embedder-128d-ngram2 |")


def test_the_pipe_and_newline_escapes_are_unchanged() -> None:
    assert _row("a|b").startswith("| a\\|b |")
    assert _row("a\r\nb").startswith("| a b |")


def test_a_single_region_still_splices() -> None:
    existing = f"H\n{TABLE_BEGIN_MARKER}\nold\n{TABLE_END_MARKER}\nT\n"
    assert splice_markdown_table(existing, "new\n") == (
        f"H\n{TABLE_BEGIN_MARKER}\nnew\n{TABLE_END_MARKER}\nT\n"
    )
