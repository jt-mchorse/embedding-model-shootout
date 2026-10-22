"""`sweep plot` refuses an output that is one of its own inputs (#242).

#230 gave `sweep aggregate` this check; `sweep plot` reads the same
`results/*.json` glob and never compared its outputs against it. Measured on
main (d83ebea) with matplotlib 3.10:

    sweep plot --results-dir res --out-png res/.json
    -> rc 0; the next `sweep aggregate` -> "malformed result JSON in res/.json", rc 2
    ln -s hash.json res/latest; sweep plot --results-dir res --out-svg res/latest
    -> rc 0; res/hash.json now begins `<?xml`
"""

from __future__ import annotations

from pathlib import Path

import pytest

from emb_shootout.cli import main
from tests.test_plot_output_format import _write_results
from tests.test_plot_output_preflight import _fake_writing_matplotlib


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Paths the chart writer saved to; the fake really writes, so a clobber shows."""
    return _fake_writing_matplotlib(monkeypatch)


def _plot(d: Path, flag: str, out: Path) -> int:
    return main(["sweep", "plot", "--results-dir", str(d), flag, str(out)])


def _refused(rc: int, capsys: pytest.CaptureFixture[str], saved: list[Path]) -> None:
    assert rc == 2
    assert "is (or would be read as) a result file" in capsys.readouterr().err
    assert saved == []


@pytest.mark.parametrize("flag", ["--out-png", "--out-svg"])
def test_a_dotfile_the_glob_reads_next_run_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], saved: list[Path], flag: str
) -> None:
    d = _write_results(tmp_path)
    _refused(_plot(d, flag, d / ".json"), capsys, saved)
    assert not (d / ".json").exists()


@pytest.mark.parametrize("flag", ["--out-png", "--out-svg"])
def test_a_symlink_to_a_result_is_refused_and_the_result_survives(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], saved: list[Path], flag: str
) -> None:
    d = _write_results(tmp_path)
    before = (d / "m.json").read_bytes()
    (d / "latest").symlink_to("m.json")
    _refused(_plot(d, flag, d / "latest"), capsys, saved)
    assert (d / "m.json").read_bytes() == before


def test_a_symlink_from_outside_the_results_dir_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], saved: list[Path]
) -> None:
    d = _write_results(tmp_path)
    before = (d / "m.json").read_bytes()
    (tmp_path / "chart").symlink_to(d / "m.json")
    _refused(_plot(d, "--out-svg", tmp_path / "chart"), capsys, saved)
    assert (d / "m.json").read_bytes() == before


def test_a_direct_result_path_is_refused_by_the_input_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], saved: list[Path]
) -> None:
    # #220's suffix check already refuses `.json` for a PNG; the input check
    # runs first, so the operator is told the real problem.
    d = _write_results(tmp_path)
    before = (d / "m.json").read_bytes()
    _refused(_plot(d, "--out-png", d / "m.json"), capsys, saved)
    assert (d / "m.json").read_bytes() == before


def test_control_an_output_outside_the_results_dir_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], saved: list[Path]
) -> None:
    d = _write_results(tmp_path)
    out = tmp_path / "docs" / "pareto.svg"
    assert _plot(d, "--out-svg", out) == 0
    assert saved == [out]


def test_control_a_non_json_name_inside_the_results_dir_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], saved: list[Path]
) -> None:
    # The glob never reads `pareto.png`, so it is not an input; aggregate
    # allows the same (#230).
    d = _write_results(tmp_path)
    assert _plot(d, "--out-png", d / "pareto.png") == 0
    assert saved == [d / "pareto.png"]
