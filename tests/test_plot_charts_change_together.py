"""`--out-png` and `--out-svg` change together or not at all (#251).

#245 made each chart write atomic, but wrote them one after the other, so a
disk-full on the SVG came after the PNG had been replaced. Measured on main
(faa58cf, macOS) with `--out-svg` on a 2 MB HFS+ image filled to 28 KB free:

    sweep plot --out-png out.png --out-svg mnt/out.svg
    -> failed to write plot: [Errno 28] No space left on device   rc 2
       out.png replaced by the new chart, mnt/out.svg still the old one

and the message named neither file. Here the disk-full is the process file-size
limit (`RLIMIT_FSIZE`, EFBIG on the write, as in #245's test), set in a child so
it cannot touch pytest's own files, with an SVG past the limit and a PNG under it.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from emb_shootout import io_utils
from emb_shootout.cli import main
from emb_shootout.plot import render_pareto
from tests.test_plot_output_format import _Ax, _result, _write_results

REPO = Path(__file__).resolve().parent.parent
OLD_PNG = b"old png\n"
OLD_SVG = b"old svg\n"


class _SizedFig:
    """A small PNG and an SVG past the 4 KB limit below."""

    def tight_layout(self, *a: object, **k: object) -> None: ...

    def savefig(self, fname, *a: object, **k: object) -> None:  # type: ignore[no-untyped-def]
        fname.write(b"P" * 100 if k.get("format") == "png" else b"S" * 20000)


class _SizedPlt:
    def subplots(self, *a: object, **k: object):  # type: ignore[no-untyped-def]
        return _SizedFig(), _Ax()

    def close(self, *a: object, **k: object) -> None: ...

    def rc_context(self, *a: object, **k: object) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


_CHILD = r"""
import resource, signal, sys

signal.signal(signal.SIGXFSZ, signal.SIG_IGN)  # EFBIG instead of a kill
import emb_shootout.plot as plot
from tests.test_plot_charts_change_together import _SizedPlt
from tests.test_plot_output_format import _result

plot._import_matplotlib = lambda: _SizedPlt()
resource.setrlimit(resource.RLIMIT_FSIZE, (4096, resource.RLIM_INFINITY))
try:
    plot.render_pareto([_result()], out_png=sys.argv[1], out_svg=sys.argv[2])
except OSError as e:
    print(e)
    sys.exit(2)
sys.exit(0)
"""


def _plot_under_a_4k_limit(png: Path, svg: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    return subprocess.run(
        [sys.executable, "-B", "-c", _CHILD, str(png), str(svg)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _old_charts(tmp_path: Path) -> tuple[Path, Path]:
    png, svg = tmp_path / "pareto.png", tmp_path / "pareto.svg"
    png.write_bytes(OLD_PNG)
    svg.write_bytes(OLD_SVG)
    return png, svg


@pytest.mark.skipif(sys.platform == "win32", reason="needs RLIMIT_FSIZE")
def test_an_svg_that_fails_partway_leaves_the_png_untouched(tmp_path: Path) -> None:
    png, svg = _old_charts(tmp_path)
    proc = _plot_under_a_4k_limit(png, svg)
    assert proc.returncode == 2, proc.stderr
    assert png.read_bytes() == OLD_PNG
    assert svg.read_bytes() == OLD_SVG
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pareto.png", "pareto.svg"]


@pytest.mark.skipif(sys.platform == "win32", reason="needs RLIMIT_FSIZE")
def test_the_error_names_the_chart_that_failed(tmp_path: Path) -> None:
    png, svg = _old_charts(tmp_path)
    proc = _plot_under_a_4k_limit(png, svg)
    assert proc.returncode == 2, proc.stderr
    assert str(svg) in proc.stdout
    assert str(png) not in proc.stdout


def _enospc_on_second_fsync(monkeypatch: pytest.MonkeyPatch) -> None:
    real_fsync = os.fsync
    calls: list[int] = []

    def _fsync(fd: int) -> None:
        calls.append(fd)
        if len(calls) == 2:
            raise OSError(28, "No space left on device")
        real_fsync(fd)

    monkeypatch.setattr(io_utils.os, "fsync", _fsync)


def test_cli_names_the_failed_chart_and_keeps_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("emb_shootout.plot._import_matplotlib", lambda: _SizedPlt())
    d = _write_results(tmp_path)
    png, svg = _old_charts(tmp_path)
    _enospc_on_second_fsync(monkeypatch)
    rc = main(
        ["sweep", "plot", "--results-dir", str(d), "--out-png", str(png), "--out-svg", str(svg)]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert f"No space left on device: {str(svg)!r}" in err
    assert png.read_bytes() == OLD_PNG
    assert svg.read_bytes() == OLD_SVG


def test_control_both_charts_are_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GREEN before and after: with nothing failing both are replaced, no temp left."""
    monkeypatch.setattr("emb_shootout.plot._import_matplotlib", lambda: _SizedPlt())
    png, svg = _old_charts(tmp_path)
    render_pareto([_result()], out_png=png, out_svg=svg)
    assert png.read_bytes() == b"P" * 100
    assert svg.read_bytes() == b"S" * 20000
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pareto.png", "pareto.svg"]


def test_control_a_png_that_fails_writes_no_svg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GREEN before and after: the first chart failing never reached the second."""
    monkeypatch.setattr("emb_shootout.plot._import_matplotlib", lambda: _SizedPlt())
    png, svg = _old_charts(tmp_path)

    def _enospc(_fd: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(io_utils.os, "fsync", _enospc)
    with pytest.raises(OSError, match="No space"):
        render_pareto([_result()], out_png=png, out_svg=svg)
    assert png.read_bytes() == OLD_PNG
    assert svg.read_bytes() == OLD_SVG
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pareto.png", "pareto.svg"]


def test_a_failed_rename_removes_every_staged_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second rename refused: its temp file goes too, and the error says which."""
    from emb_shootout.io_utils import atomic_write_bytes_many

    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    real_replace = os.replace

    def _replace(src, dst):  # type: ignore[no-untyped-def]
        if Path(dst) == b:
            raise PermissionError(1, "Operation not permitted")
        real_replace(src, dst)

    monkeypatch.setattr(io_utils.os, "replace", _replace)
    with pytest.raises(PermissionError) as exc:
        atomic_write_bytes_many([(a, b"A"), (b, b"B")])
    assert exc.value.filename == str(b)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.bin"]
