"""The atomic writers refuse a read-only existing destination (#277).

`os.replace` swaps the directory entry, so the target's own write bit was
never consulted. Measured on main (74198a4): on a `chmod 444` file holding
``old``, `atomic_write_text`, `atomic_write_bytes` and
`atomic_write_bytes_many` all returned without error and left ``new`` with
mode still 0444, and `check_writable` passed it -- while `Path.write_text`,
the call the helpers replaced, raises `PermissionError` (errno 13). The CLI
write seams document a read-only path as `failed to write` + exit 2. Helper
cases run each writer beside `Path.write_text` on the same layout, so the
lock is parity with it.

Root bypasses permission bits (`Path.write_text` succeeds too), so the
read-only arms skip as root. GitHub's ubuntu runners run as the non-root
``runner`` user, so CI exercises them.
"""

from __future__ import annotations

import errno
import json
import os
import stat
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from emb_shootout import providers
from emb_shootout.cli import main
from emb_shootout.io_utils import (
    atomic_write_bytes,
    atomic_write_bytes_many,
    atomic_write_text,
    check_writable,
)
from tests.test_plot_output_format import _write_results
from tests.test_plot_output_preflight import OLD, _fake_writing_matplotlib
from tests.test_sweep_run_out_preflight import SENT, _CountingHash, _sweep

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits"),
    pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root bypasses file permission bits, so a 0444 file stays writable",
    ),
]

WRITERS: dict[str, Callable[[Path], None]] = {
    "atomic_write_text": lambda p: atomic_write_text(p, "new\n"),
    "atomic_write_bytes": lambda p: atomic_write_bytes(p, b"new\n"),
    "atomic_write_bytes_many": lambda p: atomic_write_bytes_many([(p, b"new\n")]),
    "write_text": lambda p: p.write_text("new\n", encoding="utf-8"),
}
ATOMIC = [name for name in WRITERS if name != "write_text"]


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def frozen(tmp_path: Path) -> Iterator[Path]:
    """A 0444 file holding ``old``; made writable again for tmp_path cleanup."""
    path = tmp_path / "r.json"
    path.write_text("old\n", encoding="utf-8")
    path.chmod(0o444)
    yield path
    path.chmod(0o644)


@pytest.mark.parametrize("writer", list(WRITERS))
def test_readonly_file_is_refused(frozen: Path, writer: str) -> None:
    with pytest.raises(PermissionError) as excinfo:
        WRITERS[writer](frozen)
    assert excinfo.value.errno == errno.EACCES
    assert frozen.read_text(encoding="utf-8") == "old\n"
    assert _mode(frozen) == 0o444


@pytest.mark.parametrize("writer", ATOMIC)
def test_refusal_leaves_no_temp_file(frozen: Path, writer: str) -> None:
    with pytest.raises(PermissionError):
        WRITERS[writer](frozen)
    assert sorted(p.name for p in frozen.parent.iterdir()) == ["r.json"]


@pytest.mark.parametrize("writer", list(WRITERS))
def test_readonly_file_behind_a_symlink_is_refused(tmp_path: Path, writer: str) -> None:
    # The check runs on the RESOLVED file (#195): a writable link name over a
    # read-only file must not get through.
    real = tmp_path / "real.json"
    real.write_text("old\n", encoding="utf-8")
    real.chmod(0o444)
    link = tmp_path / "link.json"
    link.symlink_to(real)
    try:
        with pytest.raises(PermissionError):
            WRITERS[writer](link)
        assert real.read_text(encoding="utf-8") == "old\n"
        assert link.is_symlink()
    finally:
        real.chmod(0o644)


def test_many_with_one_readonly_target_changes_none(tmp_path: Path, frozen: Path) -> None:
    # The refusal fires while staging, before any rename (#251's all-or-nothing).
    first = tmp_path / "a.png"
    first.write_bytes(OLD)
    with pytest.raises(PermissionError) as excinfo:
        atomic_write_bytes_many([(first, b"new png\n"), (frozen, b"new\n")])
    assert excinfo.value.filename == str(frozen)
    assert first.read_bytes() == OLD
    assert frozen.read_text(encoding="utf-8") == "old\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.png", "r.json"]


def test_check_writable_refuses_what_the_writer_refuses(frozen: Path) -> None:
    with pytest.raises(PermissionError) as excinfo:
        check_writable(frozen)
    assert excinfo.value.errno == errno.EACCES
    assert sorted(p.name for p in frozen.parent.iterdir()) == ["r.json"]


@pytest.mark.parametrize("writer", list(WRITERS))
def test_writable_and_missing_targets_still_write(tmp_path: Path, writer: str) -> None:
    # Controls: the refusal is scoped to an existing unwritable file.
    owner_only = tmp_path / "owner.json"
    owner_only.write_text("old\n", encoding="utf-8")
    owner_only.chmod(0o600)
    WRITERS[writer](owner_only)
    assert owner_only.read_text(encoding="utf-8") == "new\n"
    assert _mode(owner_only) == 0o600

    missing = tmp_path / "new.json"
    WRITERS[writer](missing)
    assert missing.read_text(encoding="utf-8") == "new\n"


def test_check_writable_control_passes_writable_and_missing(tmp_path: Path) -> None:
    owner_only = tmp_path / "owner.json"
    owner_only.write_text("old\n", encoding="utf-8")
    owner_only.chmod(0o600)
    check_writable(owner_only)
    check_writable(tmp_path / "new.json")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["owner.json"]


def test_sweep_run_on_a_readonly_out_exits_2_with_nothing_embedded(
    tmp_path: Path,
    frozen: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    SENT.clear()
    monkeypatch.setitem(providers.PROVIDER_REGISTRY, "hash", _CountingHash)
    assert _sweep(tmp_path, frozen) == 2
    assert SENT == []
    assert f"failed to write {frozen}" in capsys.readouterr().err
    assert frozen.read_text(encoding="utf-8") == "old\n"


def test_corpus_validate_on_a_readonly_out_exits_2(
    tmp_path: Path, frozen: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"chunk_id": "c1", "text": "hello"}) + "\n", encoding="utf-8")
    rc = main(["corpus", "validate", str(corpus), "--out", str(frozen)])
    assert rc == 2
    assert f"failed to write {frozen}" in capsys.readouterr().err
    assert frozen.read_text(encoding="utf-8") == "old\n"


def test_sweep_plot_on_a_readonly_svg_exits_2_with_the_png_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    saved = _fake_writing_matplotlib(monkeypatch)
    d = _write_results(tmp_path)
    png = tmp_path / "pareto.png"
    png.write_bytes(OLD)
    svg = tmp_path / "pareto.svg"
    svg.write_bytes(OLD)
    svg.chmod(0o444)
    try:
        rc = main(
            ["sweep", "plot", "--results-dir", str(d), "--out-png", str(png), "--out-svg", str(svg)]
        )
        assert rc == 2
        assert "failed to write plot" in capsys.readouterr().err
        assert saved == []  # refused by the preflight, before the chart was handed over
        assert png.read_bytes() == OLD
        assert svg.read_bytes() == OLD
    finally:
        svg.chmod(0o644)
