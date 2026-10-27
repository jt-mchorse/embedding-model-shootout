"""`sweep plot` checks every output is writable before writing any (#234).

`render_pareto` saved the outputs in turn with no preflight. Measured on main
(aa06cd9) with matplotlib 3.10:

    --out-png pareto.png --out-svg blocker/pareto.svg   (blocker is a file)
    -> "failed to write plot: [Errno 17] File exists: 'blocker'", rc 2,
       and pareto.png already replaced by the new chart
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

from emb_shootout import io_utils
from emb_shootout.cli import main
from emb_shootout.plot import render_pareto
from tests.test_plot_output_format import _Ax, _result, _write_results

OLD = b"previous committed chart\n"


class _WritingFig:
    """A fake figure whose `savefig` really writes, so a partial write is visible."""

    def __init__(self, saved: list[Path]) -> None:
        self._saved = saved

    def tight_layout(self, *a: object, **k: object) -> None: ...

    def savefig(self, fname: io.BytesIO, *a: object, **k: object) -> None:
        # Into memory since #245; the `saved` fixture records the real write.
        fname.write(f"new {k.get('format')}\n".encode())


class _WritingPlt:
    def __init__(self, saved: list[Path]) -> None:
        self._saved = saved

    def subplots(self, *a: object, **k: object):
        return _WritingFig(self._saved), _Ax()

    def close(self, *a: object, **k: object) -> None: ...

    def rc_context(self, *a: object, **k: object) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


def _fake_writing_matplotlib(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Install `_WritingPlt`; return the path of each chart actually written."""
    calls: list[Path] = []
    monkeypatch.setattr("emb_shootout.plot._import_matplotlib", lambda: _WritingPlt(calls))

    def _recording_write(items: list[tuple[Path, bytes]]) -> None:
        calls.extend(Path(path) for path, _data in items)
        io_utils.atomic_write_bytes_many(items)

    monkeypatch.setattr("emb_shootout.plot.atomic_write_bytes_many", _recording_write)
    return calls


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    return _fake_writing_matplotlib(monkeypatch)


def _blocked(tmp_path: Path, how: str, name: str) -> Path:
    """An output path the writer cannot write, in one of two ways."""
    if how == "parent-is-a-file":
        (tmp_path / "blocker").write_text("a file\n", encoding="utf-8")
        return tmp_path / "blocker" / name
    (tmp_path / name).mkdir()  # "is-a-directory"
    return tmp_path / name


@pytest.mark.parametrize("how", ["parent-is-a-file", "is-a-directory"])
def test_an_unwritable_svg_leaves_the_png_untouched(
    tmp_path: Path, saved: list[Path], how: str
) -> None:
    png = tmp_path / "pareto.png"
    png.write_bytes(OLD)
    svg = _blocked(tmp_path, how, "pareto.svg")
    with pytest.raises(OSError, match=r"Errno"):
        render_pareto([_result()], out_png=png, out_svg=svg)
    assert saved == []
    assert png.read_bytes() == OLD


@pytest.mark.parametrize("how", ["parent-is-a-file", "is-a-directory"])
def test_an_unwritable_png_writes_no_svg(tmp_path: Path, saved: list[Path], how: str) -> None:
    """The mirror case, GREEN before and after: the first output is the bad one, so even
    the unfixed loop stopped before the SVG. Locks that the preflight keeps it so."""
    png = _blocked(tmp_path, how, "pareto.png")
    svg = tmp_path / "pareto.svg"
    with pytest.raises(OSError, match=r"Errno"):
        render_pareto([_result()], out_png=png, out_svg=svg)
    assert saved == []
    assert not svg.exists()


def test_cli_exits_2_with_the_png_untouched(
    tmp_path: Path, saved: list[Path], capsys: pytest.CaptureFixture[str]
) -> None:
    d = _write_results(tmp_path)
    png = tmp_path / "pareto.png"
    png.write_bytes(OLD)
    svg = _blocked(tmp_path, "parent-is-a-file", "pareto.svg")
    rc = main(
        ["sweep", "plot", "--results-dir", str(d), "--out-png", str(png), "--out-svg", str(svg)]
    )
    assert rc == 2
    assert "failed to write plot" in capsys.readouterr().err
    assert png.read_bytes() == OLD


def test_control_two_writable_outputs_are_both_written(tmp_path: Path, saved: list[Path]) -> None:
    """GREEN before and after: the preflight leaves no temp file and blocks nothing."""
    png = tmp_path / "pareto.png"
    png.write_bytes(OLD)
    svg = tmp_path / "new" / "pareto.svg"  # a parent that does not exist yet
    render_pareto([_result()], out_png=png, out_svg=svg)
    assert saved == [png, svg]
    assert png.read_bytes() == b"new png\n"
    assert svg.read_bytes() == b"new svg\n"
    assert sorted(p.name for p in tmp_path.rglob("*")) == ["new", "pareto.png", "pareto.svg"]
