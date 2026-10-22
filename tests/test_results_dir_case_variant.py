"""A case-variant spelling of `--results-dir` is still the results dir (#250).

`_is_result_input` compared the output's parent to `--results-dir` as a
resolved string, and `resolve()` keeps the operator's case. Measured on main
(faa58cf), macOS APFS:

    sweep plot --results-dir res --out-png RES/.json
    -> rc 0, the PNG lands at res/.json; the next aggregate -> rc 2
    sweep aggregate --results-dir res --format json --out RES/agg.json
    -> rc 0; the next aggregate -> "result missing required field(s)", rc 2

The case-variant tests only mean something on a case-insensitive filesystem
(default macOS, Windows) and skip elsewhere; the exact-spelling controls run
everywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from emb_shootout.cli import main
from tests.test_plot_output_format import _write_results
from tests.test_plot_output_preflight import _fake_writing_matplotlib


def _case_insensitive(d: Path) -> bool:
    probe = d / "case-probe"
    probe.write_text("", encoding="utf-8")
    try:
        return (d / "CASE-PROBE").exists()
    finally:
        probe.unlink()


@pytest.fixture
def results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = _write_results(tmp_path)  # tmp_path / "results"
    monkeypatch.chdir(tmp_path)
    return d


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Paths the chart writer saved to; the fake really writes, so a clobber shows."""
    return _fake_writing_matplotlib(monkeypatch)


def _snapshot(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


needs_case_insensitive_fs = pytest.mark.skipif(
    not _case_insensitive(Path(__file__).resolve().parent),
    reason="a case variant only names the same directory on a case-insensitive filesystem",
)


def _plot(flag: str, out: str) -> int:
    return main(["sweep", "plot", "--results-dir", "results", flag, out])


def _aggregate(fmt: str, out: str) -> int:
    return main(["sweep", "aggregate", "--results-dir", "results", "--format", fmt, "--out", out])


# --- case variant (case-insensitive filesystems only) ------------------------


@needs_case_insensitive_fs
@pytest.mark.parametrize("out", ["RESULTS/.json", "Results/new.json"])
@pytest.mark.parametrize("flag", ["--out-png", "--out-svg"])
def test_plot_refuses_a_case_variant_results_dir(
    results: Path,
    saved: list[Path],
    capsys: pytest.CaptureFixture[str],
    flag: str,
    out: str,
) -> None:
    before = _snapshot(results)
    assert _plot(flag, out) == 2
    assert "is (or would be read as) a result file" in capsys.readouterr().err
    assert saved == []
    assert _snapshot(results) == before


@needs_case_insensitive_fs
@pytest.mark.parametrize("out", ["RESULTS/.json", "Results/agg.json"])
@pytest.mark.parametrize("fmt", ["markdown", "json"])
def test_aggregate_refuses_a_case_variant_results_dir(
    results: Path, capsys: pytest.CaptureFixture[str], fmt: str, out: str
) -> None:
    before = _snapshot(results)
    assert _aggregate(fmt, out) == 2
    assert "refusing to write --out" in capsys.readouterr().err
    assert _snapshot(results) == before
    # The directory is still aggregatable.
    assert _aggregate("markdown", "t.md") == 0


# --- controls (every filesystem) ---------------------------------------------


@pytest.mark.parametrize("flag", ["--out-png", "--out-svg"])
def test_control_plot_still_refuses_the_exact_spelling(
    results: Path, saved: list[Path], flag: str
) -> None:
    before = _snapshot(results)
    assert _plot(flag, "results/.json") == 2
    assert saved == []
    assert _snapshot(results) == before


@pytest.mark.parametrize("fmt", ["markdown", "json"])
def test_control_aggregate_still_refuses_the_exact_spelling(results: Path, fmt: str) -> None:
    before = _snapshot(results)
    assert _aggregate(fmt, "results/agg.json") == 2
    assert _snapshot(results) == before


def test_control_an_output_in_a_missing_directory_is_written(
    results: Path, saved: list[Path]
) -> None:
    # The identity compare must not trip on a parent that does not exist yet.
    assert _aggregate("json", "new/agg.json") == 0
    assert Path("new/agg.json").is_file()
    assert _plot("--out-svg", "charts/pareto.svg") == 0
    assert saved == [Path("charts/pareto.svg")]


def test_control_a_sibling_dir_is_not_the_results_dir(results: Path) -> None:
    Path("other").mkdir()
    assert _aggregate("json", "other/agg.json") == 0
    assert Path("other/agg.json").is_file()
