"""A failed chart write leaves the previous chart intact (#245).

`render_pareto` called `fig.savefig(path)` straight onto the destination, so a
write that failed partway truncated the committed `docs/pareto.png`. Measured
on main (d83ebea) with the process file-size limit at 4 KB (`RLIMIT_FSIZE`,
writes past it fail with EFBIG, the same mid-write failure as ENOSPC):

    sweep plot --out-png <copy of docs/pareto.png> -> "failed to write plot:
        [Errno 27] File too large", rc 2, and the PNG went from 58336 bytes to 4096

The limit is set in a child process so it cannot touch pytest's own files. The
fake figure there writes to a path or a file object, so these arms run in CI
(no `plot` extra, #111) against either shape of `render_pareto`.
"""

from __future__ import annotations

import contextlib
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from emb_shootout.plot import render_pareto
from tests.test_plot_output_format import _result

REPO = Path(__file__).resolve().parent.parent
OLD = b"previous committed chart " * 3000  # 75 KB, past the 4 KB limit below
NEW = b"N" * 20000  # also past it


class _Fig:
    """Saves to a path (main before #245) or a file object (after), the same bytes."""

    def tight_layout(self, *a: object, **k: object) -> None: ...

    def savefig(self, fname: object, *a: object, **k: object) -> None:
        if isinstance(fname, (str, os.PathLike)):
            with open(fname, "wb") as f:
                f.write(NEW)
        else:
            fname.write(NEW)  # type: ignore[attr-defined]


class _Ax:
    def __getattr__(self, _name: str):
        return lambda *a, **k: None


class _Plt:
    def subplots(self, *a: object, **k: object):
        return _Fig(), _Ax()

    def close(self, *a: object, **k: object) -> None: ...

    def rc_context(self, *a: object, **k: object) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


@pytest.fixture
def fake_plt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("emb_shootout.plot._import_matplotlib", lambda: _Plt())


_CHILD = r"""
import os, resource, signal, sys
from pathlib import Path

signal.signal(signal.SIGXFSZ, signal.SIG_IGN)  # EFBIG instead of a kill
import emb_shootout.plot as plot
from tests.test_plot_atomic_write import _Plt
from tests.test_plot_output_format import _result

if sys.argv[1] == "fake":
    plot._import_matplotlib = lambda: _Plt()
else:
    plot._import_matplotlib()  # import before the limit; font caches write files
resource.setrlimit(resource.RLIMIT_FSIZE, (4096, resource.RLIM_INFINITY))
try:
    plot.render_pareto([_result()], out_png=sys.argv[2])
except OSError as e:
    print(e.errno)
    sys.exit(2)
sys.exit(0)
"""


def _plot_under_a_4k_limit(png: Path, mode: str = "fake") -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    return subprocess.run(
        [sys.executable, "-B", "-c", _CHILD, mode, str(png)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="needs RLIMIT_FSIZE")
@pytest.mark.parametrize("mode", ["fake", "real"])
def test_a_write_that_fails_partway_keeps_the_old_chart(tmp_path: Path, mode: str) -> None:
    if mode == "real":
        pytest.importorskip("matplotlib")
    png = tmp_path / "docs" / "pareto.png"
    png.parent.mkdir()
    png.write_bytes(OLD)
    proc = _plot_under_a_4k_limit(png, mode)
    assert proc.returncode == 2, proc.stderr
    assert png.read_bytes() == OLD
    assert sorted(p.name for p in png.parent.iterdir()) == ["pareto.png"]  # no temp file


@pytest.mark.skipif(sys.platform == "win32", reason="needs RLIMIT_FSIZE")
def test_a_new_chart_that_fails_partway_leaves_nothing(tmp_path: Path) -> None:
    png = tmp_path / "pareto.png"
    proc = _plot_under_a_4k_limit(png)
    assert proc.returncode == 2, proc.stderr
    assert list(tmp_path.iterdir()) == []


def test_control_a_symlinked_chart_is_written_through(tmp_path: Path, fake_plt: None) -> None:
    """GREEN before and after, as `savefig(path)` did: the link stays a link and its
    target gets the chart."""
    target = tmp_path / "pareto.png"
    target.write_bytes(OLD)
    link = tmp_path / "latest.png"
    link.symlink_to(target)
    render_pareto([_result()], out_png=link)
    assert link.is_symlink()
    assert target.read_bytes() == NEW


def test_control_an_overwrite_keeps_the_file_mode(tmp_path: Path, fake_plt: None) -> None:
    """GREEN before and after: the normal write replaces the chart, mode intact (#165)."""
    png = tmp_path / "pareto.png"
    png.write_bytes(OLD)
    png.chmod(0o640)
    render_pareto([_result()], out_png=png)
    assert png.read_bytes() == NEW
    assert stat.S_IMODE(png.stat().st_mode) == 0o640
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pareto.png"]


def test_atomic_write_bytes_writes_exactly_the_bytes(tmp_path: Path) -> None:
    from emb_shootout.io_utils import atomic_write_bytes

    out = tmp_path / "new" / "x.bin"
    data = bytes(range(256)) + b"\r\n\x00"
    atomic_write_bytes(out, data)
    assert out.read_bytes() == data
    assert sorted(p.name for p in out.parent.iterdir()) == ["x.bin"]


def test_atomic_write_bytes_failure_keeps_the_old_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from emb_shootout import io_utils

    out = tmp_path / "x.bin"
    out.write_bytes(OLD)

    def _enospc(_fd: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(io_utils.os, "fsync", _enospc)
    with pytest.raises(OSError, match="No space"):
        io_utils.atomic_write_bytes(out, b"new")
    assert out.read_bytes() == OLD
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.bin"]
