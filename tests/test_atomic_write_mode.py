"""File-mode contract for `atomic_write_text` (#165, portfolio-ops#81).

The helper used to create its temp file with `tempfile.NamedTemporaryFile`,
which always opens 0600 regardless of the umask, and `os.replace` carried that
mode onto the target. So every file it wrote was owner-only, and an overwrite of
an existing 0644 file demoted it to 0600. The `Path.write_text` it replaced did
neither.

Contract pinned here:
- a NEW file gets `0o666 & ~umask` (umask 022 -> 0644, umask 077 -> 0600: the
  second proves the umask is honoured, not a hard-coded 0644);
- an OVERWRITE keeps the existing file's mode (0644, 0600, 0640), whatever the
  umask is;
- the same holds through real callers: `corpus.write_jsonl` and the
  `corpus validate --out` CLI seam.

Every test writes only under `tmp_path`.
"""

from __future__ import annotations

import os
import re
import stat
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from emb_shootout import build_corpus, io_utils, write_jsonl
from emb_shootout.cli import main
from emb_shootout.io_utils import atomic_write_text

SetUmask = Callable[[int], None]

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits and umask")


@pytest.fixture
def set_umask() -> Iterator[SetUmask]:
    """Yield a setter for the process umask; restore the original afterwards."""
    # os.umask has no read-only form; the set-and-restore here runs on the
    # test thread before any write, not inside the helper.
    original = os.umask(0o022)
    os.umask(original)

    def _set(mask: int) -> None:
        os.umask(mask)

    try:
        yield _set
    finally:
        os.umask(original)


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.mark.parametrize(("mask", "expected"), [(0o022, 0o644), (0o077, 0o600), (0o027, 0o640)])
def test_new_file_mode_follows_umask(
    tmp_path: Path, set_umask: SetUmask, mask: int, expected: int
) -> None:
    set_umask(mask)
    target = tmp_path / "new.txt"
    atomic_write_text(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello"
    assert _mode(target) == expected


@pytest.mark.parametrize("mask", [0o022, 0o077])
@pytest.mark.parametrize("existing", [0o644, 0o600, 0o640])
def test_overwrite_keeps_existing_mode(
    tmp_path: Path, set_umask: SetUmask, mask: int, existing: int
) -> None:
    set_umask(mask)
    target = tmp_path / "existing.txt"
    target.write_text("old", encoding="utf-8")
    os.chmod(target, existing)
    atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"
    assert _mode(target) == existing


def test_new_file_in_created_parent_follows_umask(tmp_path: Path, set_umask: SetUmask) -> None:
    set_umask(0o022)
    target = tmp_path / "a" / "b" / "out.md"
    atomic_write_text(target, "x")
    assert _mode(target) == 0o644


def test_temp_name_shape_and_no_leftovers(
    tmp_path: Path, set_umask: SetUmask, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replacement temp file keeps the `.<base>.<random>.tmp` shape the
    name-length budget was sized for, and nothing is left behind."""
    set_umask(0o022)
    seen: list[Path] = []
    real_replace = io_utils.os.replace

    def spy(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        seen.append(Path(src))
        real_replace(src, dst)

    monkeypatch.setattr(io_utils.os, "replace", spy)
    target = tmp_path / "report.md"
    atomic_write_text(target, "x")
    assert len(seen) == 1
    assert seen[0].parent == tmp_path
    assert re.fullmatch(r"\.report\.md\.[0-9a-f]{8}\.tmp", seen[0].name)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.md"]


def test_bad_encoding_leaves_no_temp_file(tmp_path: Path) -> None:
    """`os.fdopen` raising (unknown codec) must not leak the temp file."""
    target = tmp_path / "out.txt"
    with pytest.raises(LookupError):
        atomic_write_text(target, "x", encoding="no-such-codec")
    assert list(tmp_path.iterdir()) == []


def test_write_jsonl_new_corpus_is_0644_under_umask_022(
    tmp_path: Path, set_umask: SetUmask
) -> None:
    set_umask(0o022)
    out = tmp_path / "corpus.jsonl"
    assert write_jsonl(build_corpus(["json"]), out) > 0
    assert _mode(out) == 0o644


def test_write_jsonl_overwrite_keeps_0644(tmp_path: Path, set_umask: SetUmask) -> None:
    set_umask(0o022)
    out = tmp_path / "corpus.jsonl"
    out.write_text("", encoding="utf-8")
    os.chmod(out, 0o644)
    write_jsonl(build_corpus(["json"]), out)
    assert _mode(out) == 0o644


def test_corpus_validate_out_report_is_0644_under_umask_022(
    tmp_path: Path, set_umask: SetUmask
) -> None:
    """Through the CLI `--out` seam an operator actually uses."""
    set_umask(0o022)
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(build_corpus(["json"]), corpus)
    report = tmp_path / "validate.md"
    rc = main(["corpus", "validate", str(corpus), "--out", str(report)])
    assert rc in (0, 1)
    assert report.exists()
    assert _mode(report) == 0o644


def test_temp_name_collision_draws_another_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`O_EXCL` never clobbers an existing file; a collision just retries."""
    squatter = tmp_path / ".out.txt.aaaaaaaa.tmp"
    squatter.write_text("someone else's", encoding="utf-8")
    draws = iter(["aaaaaaaa", "bbbbbbbb"])
    monkeypatch.setattr(io_utils.secrets, "token_hex", lambda _n: next(draws))
    target = tmp_path / "out.txt"
    atomic_write_text(target, "mine")
    assert target.read_text(encoding="utf-8") == "mine"
    assert squatter.read_text(encoding="utf-8") == "someone else's"


def test_temp_name_exhaustion_raises_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    squatter = tmp_path / ".out.txt.aaaaaaaa.tmp"
    squatter.write_text("x", encoding="utf-8")
    monkeypatch.setattr(io_utils.secrets, "token_hex", lambda _n: "aaaaaaaa")
    with pytest.raises(FileExistsError):
        atomic_write_text(tmp_path / "out.txt", "mine")
    assert sorted(p.name for p in tmp_path.iterdir()) == [squatter.name]
