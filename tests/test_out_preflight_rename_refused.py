"""The `--out` preflight refuses a target the final rename is refused onto (#251).

`check_writable` (#200) extends the preflight to the writer's final
`os.replace`, but only refused an existing directory. Measured on main
(faa58cf, macOS):

    echo old > locked.json; chflags uchg locked.json
    sweep run --provider hash --corpus c.jsonl --queries 10 --out locked.json
    -> failed to write locked.json: [Errno 1] Operation not permitted:
       '.locked.json.<rand>.tmp' -> 'locked.json'   rc 2, after the whole sweep

The `chflags` arms need macOS/BSD. The faked-`st_flags` and faked-euid arms run
the same check on Linux CI; the sticky-directory case cannot be built for real
without a second user.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from emb_shootout import io_utils, providers
from emb_shootout.io_utils import check_writable
from emb_shootout.plot import render_pareto
from tests.test_plot_output_format import _result
from tests.test_plot_output_preflight import _WritingPlt
from tests.test_sweep_run_out_preflight import SENT, _CountingHash, _sweep

OLD = b"old\n"
needs_chflags = pytest.mark.skipif(
    not hasattr(os, "chflags") or sys.platform.startswith("linux"),
    reason="needs BSD file flags (chflags)",
)


@pytest.fixture(autouse=True)
def _count(monkeypatch: pytest.MonkeyPatch) -> None:
    SENT.clear()
    monkeypatch.setitem(providers.PROVIDER_REGISTRY, "hash", _CountingHash)


@pytest.fixture
def locked(tmp_path: Path):  # type: ignore[no-untyped-def]
    """An existing file flagged `uchg`, unflagged again so tmp_path can be removed."""
    path = tmp_path / "locked"  # no suffix: also a valid --out-svg (#220)
    path.write_bytes(OLD)
    os.chflags(path, stat.UF_IMMUTABLE)
    yield path
    os.chflags(path, 0)


@needs_chflags
def test_an_immutable_out_is_refused_with_nothing_embedded(
    tmp_path: Path, locked: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _sweep(tmp_path, locked) == 2
    assert SENT == []
    assert f"failed to write {locked}" in capsys.readouterr().err
    assert locked.read_bytes() == OLD


@needs_chflags
def test_an_immutable_svg_leaves_the_png_untouched(
    tmp_path: Path, locked: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`render_pareto` preflights each chart with the same helper, so a second
    chart whose rename would be refused stops the first being replaced."""
    monkeypatch.setattr("emb_shootout.plot._import_matplotlib", lambda: _WritingPlt([]))
    png = tmp_path / "pareto.png"
    png.write_bytes(OLD)
    with pytest.raises(PermissionError):
        render_pareto([_result()], out_png=png, out_svg=locked)
    assert png.read_bytes() == OLD
    assert locked.read_bytes() == OLD


class _Flagged:
    """A stat result with `st_flags` set, otherwise the real one."""

    def __init__(self, real: os.stat_result, flags: int) -> None:
        self._real = real
        self.st_flags = flags

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


@pytest.mark.parametrize("flag", ["UF_IMMUTABLE", "SF_IMMUTABLE", "UF_APPEND", "SF_APPEND"])
def test_each_no_replace_flag_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    """The check itself, on any platform: the flags as `stat` reports them on BSD."""
    target = tmp_path / "r.json"
    target.write_bytes(OLD)
    real_stat = os.stat

    def _stat(p, *a, **k):  # type: ignore[no-untyped-def]
        st = real_stat(p, *a, **k)
        return _Flagged(st, getattr(stat, flag)) if Path(p) == target else st

    monkeypatch.setattr(io_utils.os, "stat", _stat)
    with pytest.raises(PermissionError, match="immutable or append-only"):
        check_writable(target)


def test_control_an_unrelated_flag_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GREEN before and after: `UF_NODUMP` does not stop a rename."""
    target = tmp_path / "r.json"
    target.write_bytes(OLD)
    real_stat = os.stat

    def _stat(p, *a, **k):  # type: ignore[no-untyped-def]
        st = real_stat(p, *a, **k)
        return _Flagged(st, stat.UF_NODUMP) if Path(p) == target else st

    monkeypatch.setattr(io_utils.os, "stat", _stat)
    check_writable(target)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r.json"]


def _sticky_dir(tmp_path: Path, sticky: bool) -> Path:
    d = tmp_path / "shared"
    d.mkdir()
    d.chmod(0o1777 if sticky else 0o777)
    (d / "r.json").write_bytes(OLD)
    return d / "r.json"


def test_another_users_file_in_a_sticky_directory_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seen as a different euid: neither the file nor the directory is ours."""
    target = _sticky_dir(tmp_path, sticky=True)
    monkeypatch.setattr(io_utils.os, "geteuid", lambda: target.stat().st_uid + 1)
    with pytest.raises(PermissionError, match="sticky directory"):
        check_writable(target)


@pytest.mark.parametrize("case", ["not-sticky", "own-file", "root"])
def test_control_a_sticky_directory_that_allows_the_rename_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """GREEN before and after: without the sticky bit, for our own file, or as root."""
    target = _sticky_dir(tmp_path, sticky=case != "not-sticky")
    uid = target.stat().st_uid
    euid = {"not-sticky": uid + 1, "own-file": uid, "root": 0}[case]
    monkeypatch.setattr(io_utils.os, "geteuid", lambda: euid)
    check_writable(target)
    assert target.read_bytes() == OLD


def test_control_a_new_file_in_a_sticky_directory_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GREEN before and after: nothing to replace, so the sticky rule does not apply."""
    target = _sticky_dir(tmp_path, sticky=True).with_name("new.json")
    monkeypatch.setattr(io_utils.os, "geteuid", lambda: target.parent.stat().st_uid + 1)
    check_writable(target)
    assert not target.exists()
