"""`sweep plot` refuses two outputs that are one file, and an empty path (#228).

#220 allowed an output path with no extension, and each path was checked on
its own. Measured on main with matplotlib 3.10.9:

    --out-png same --out-svg same  -> rc 0, `same` holds the SVG, summary
                                      reports "png": "same" and "svg": "same"
    --out-png ""                   -> rc 0, nothing written, "png": null
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from emb_shootout.cli import main
from emb_shootout.plot import render_pareto
from tests.test_plot_output_format import _fake_matplotlib, _result, _write_results


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, object]]:
    return _fake_matplotlib(monkeypatch)


@pytest.mark.parametrize(
    "svg_spelling",
    ["same", "./same", "sub/../same"],
)
def test_two_flags_naming_one_file_are_refused_before_writing(
    tmp_path: Path,
    saved: list,
    svg_spelling: str,
) -> None:
    (tmp_path / "sub").mkdir()
    png = tmp_path / "same"
    svg = Path(f"{tmp_path}/{svg_spelling}")
    with pytest.raises(ValueError, match="are the same file"):
        render_pareto([_result()], out_png=png, out_svg=svg)
    assert saved == []


def test_a_symlinked_directory_is_the_same_file(tmp_path: Path, saved: list) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    with pytest.raises(ValueError, match="are the same file"):
        render_pareto([_result()], out_png=real / "chart", out_svg=tmp_path / "link" / "chart")
    assert saved == []


def test_an_existing_hard_link_is_the_same_file(tmp_path: Path, saved: list) -> None:
    first = tmp_path / "a"
    first.write_bytes(b"old")
    os.link(first, tmp_path / "b")
    with pytest.raises(ValueError, match="are the same file"):
        render_pareto([_result()], out_png=first, out_svg=tmp_path / "b")
    assert saved == []


@pytest.mark.parametrize("flag", ["--out-png", "--out-svg"])
@pytest.mark.parametrize("value", ["", "   "], ids=["empty", "blank"])
def test_cli_refuses_an_empty_path_at_exit_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    saved: list,
    flag: str,
    value: str,
) -> None:
    d = _write_results(tmp_path)
    rc = main(["sweep", "plot", "--results-dir", str(d), flag, value])
    assert rc == 2
    assert "output path must be non-empty" in capsys.readouterr().err
    assert saved == []


def test_cli_refuses_the_same_file_at_exit_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    saved: list,
) -> None:
    d = _write_results(tmp_path)
    out = str(tmp_path / "same")
    rc = main(["sweep", "plot", "--results-dir", str(d), "--out-png", out, "--out-svg", out])
    assert rc == 2
    assert "are the same file" in capsys.readouterr().err
    assert saved == []


def test_distinct_paths_still_write_both(tmp_path: Path, saved: list) -> None:
    png, svg = tmp_path / "chart", tmp_path / "pareto"
    render_pareto([_result()], out_png=png, out_svg=svg)
    assert saved == [(png, "png"), (svg, "svg")]
