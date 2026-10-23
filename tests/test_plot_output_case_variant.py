"""`sweep plot` refuses two outputs the filesystem folds to one file (#255).

#228 compared the two paths resolved and, when both existed, by identity; its
comment admitted the hole that left. Measured on main (32cb88d), macOS APFS,
matplotlib 3.10, in a directory holding neither file:

    sweep plot --out-png chart --out-svg CHART
    -> rc 0, summary {"png": "chart", "svg": "CHART"}, one file `chart`
       whose first bytes are `<?xml version="1.0"`

The fold tests only mean something where the filesystem folds the names
(default macOS, Windows) and skip elsewhere, as #250's tests do; the controls
run everywhere.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from emb_shootout.cli import main
from emb_shootout.plot import render_pareto
from tests.test_plot_output_format import _result, _write_results
from tests.test_plot_output_preflight import _fake_writing_matplotlib

_NFC = unicodedata.normalize("NFC", "café")
_NFD = unicodedata.normalize("NFD", _NFC)


def _folds(d: Path, name: str, variant: str) -> bool:
    probe = d / name
    probe.write_text("", encoding="utf-8")
    try:
        return (d / variant).exists()
    finally:
        probe.unlink()


_HERE = Path(__file__).resolve().parent
needs_case_insensitive_fs = pytest.mark.skipif(
    not _folds(_HERE, "case-probe", "CASE-PROBE"),
    reason="a case variant only names the same file on a case-insensitive filesystem",
)
needs_normalization_insensitive_fs = pytest.mark.skipif(
    not _folds(_HERE, f"{_NFC}-probe", f"{_NFD}-probe"),
    reason="NFC and NFD spellings only name one file on a normalization-insensitive filesystem",
)


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Paths the chart writer saved to; the fake really writes, so a clobber shows."""
    return _fake_writing_matplotlib(monkeypatch)


# --- folded names (filesystems that fold them only) ---------------------------


@needs_case_insensitive_fs
@pytest.mark.parametrize(
    ("png", "svg"),
    [("chart", "CHART"), ("Chart", "chart"), ("Chart", "cHART")],
)
def test_a_case_variant_pair_is_refused_and_nothing_is_written(
    tmp_path: Path, saved: list[Path], png: str, svg: str
) -> None:
    with pytest.raises(ValueError, match="are the same file"):
        render_pareto([_result()], out_png=tmp_path / png, out_svg=tmp_path / svg)
    assert saved == []
    assert list(tmp_path.iterdir()) == []  # the fold probe removed its file


@needs_case_insensitive_fs
def test_a_case_variant_parent_is_refused(tmp_path: Path, saved: list[Path]) -> None:
    (tmp_path / "sub").mkdir()
    with pytest.raises(ValueError, match="are the same file"):
        render_pareto(
            [_result()], out_png=tmp_path / "sub" / "chart", out_svg=tmp_path / "SUB" / "chart"
        )
    assert saved == []
    assert list((tmp_path / "sub").iterdir()) == []


@needs_normalization_insensitive_fs
def test_an_nfc_nfd_pair_is_refused(tmp_path: Path, saved: list[Path]) -> None:
    # Why the filesystem is asked rather than `casefold()` compared: these two
    # names are casefold-distinct and still one file on APFS.
    assert _NFC.casefold() != _NFD.casefold()
    with pytest.raises(ValueError, match="are the same file"):
        render_pareto([_result()], out_png=tmp_path / _NFC, out_svg=tmp_path / _NFD)
    assert saved == []
    assert list(tmp_path.iterdir()) == []


@needs_case_insensitive_fs
def test_the_cli_exits_2_and_writes_nothing(
    tmp_path: Path,
    saved: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_results(tmp_path)
    monkeypatch.chdir(tmp_path)
    Path("out").mkdir()
    argv = ["sweep", "plot", "--results-dir", "results"]
    assert main([*argv, "--out-png", "out/chart", "--out-svg", "out/CHART"]) == 2
    assert "are the same file" in capsys.readouterr().err
    assert saved == []
    assert list(Path("out").iterdir()) == []


# --- controls (every filesystem) ---------------------------------------------


def test_control_two_distinct_names_write_both(tmp_path: Path, saved: list[Path]) -> None:
    png, svg = tmp_path / "chart", tmp_path / "chart2"
    render_pareto([_result()], out_png=png, out_svg=svg)
    assert saved == [png, svg]
    assert png.read_bytes() == b"new png\n"
    assert svg.read_bytes() == b"new svg\n"


def test_control_distinct_names_in_a_missing_directory_write_both(
    tmp_path: Path, saved: list[Path]
) -> None:
    # The fold check runs after the preflight has created the parent.
    png, svg = tmp_path / "new" / "pareto.png", tmp_path / "new" / "pareto.svg"
    render_pareto([_result()], out_png=png, out_svg=svg)
    assert saved == [png, svg]
    assert sorted(p.name for p in (tmp_path / "new").iterdir()) == ["pareto.png", "pareto.svg"]


def test_control_the_same_name_in_two_directories_writes_both(
    tmp_path: Path, saved: list[Path]
) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    png, svg = tmp_path / "a" / "chart", tmp_path / "b" / "chart"
    render_pareto([_result()], out_png=png, out_svg=svg)
    assert saved == [png, svg]


def test_control_existing_distinct_outputs_are_replaced(tmp_path: Path, saved: list[Path]) -> None:
    # Both exist, so the probe is skipped and must not refuse or touch them early.
    png, svg = tmp_path / "pareto.png", tmp_path / "pareto.svg"
    png.write_bytes(b"old png\n")
    svg.write_bytes(b"old svg\n")
    render_pareto([_result()], out_png=png, out_svg=svg)
    assert png.read_bytes() == b"new png\n"
    assert svg.read_bytes() == b"new svg\n"


def test_control_the_exact_spelling_is_still_refused(tmp_path: Path, saved: list[Path]) -> None:
    with pytest.raises(ValueError, match="are the same file"):
        render_pareto([_result()], out_png=tmp_path / "same", out_svg=tmp_path / "same")
    assert saved == []
    assert list(tmp_path.iterdir()) == []
