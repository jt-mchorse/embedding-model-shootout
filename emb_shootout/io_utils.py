"""Atomic on-disk write helper.

The shootout writes several artifact kinds whose downstream consumers
cannot tolerate partial files:

- `sweep run --output <P>` writes one per-provider result JSON; D-007
  says the aggregator scans `*.json` and merges. A half-written file
  poisons the aggregator (`json.JSONDecodeError`) or, worse, silently
  truncates the providers that the plot subcommand renders against.
- `sweep aggregate --out docs/benchmarks.md` writes the markdown that
  the README's "Benchmarks" section is rendered from on GitHub. A
  partial render is the front-page failure mode.
- `corpus build --out <P>` writes the JSONL corpus that every sweep
  reads. The corpus is row-oriented so a truncation at a row boundary
  passes the parser silently — quality numbers drift down without a
  loud signal.
- `notebooks/_build_notebook.py` writes `notebooks/embedding_shootout.ipynb`;
  a partial notebook breaks `jupyter nbconvert` and the README link.

`Path.write_text` and the `open("w") + f.write(...)` shape are both
non-atomic: SIGINT/SIGTERM/disk-full/OOM between the implicit truncate
and `close()` flush leaves the destination zero-length or partial.

`atomic_write_text` (and `atomic_write_bytes`, for the charts
`sweep plot` writes to the committed `docs/pareto.png`/`.svg`, #245)
writes to a sibling temp file in the same directory, `fsync`s, then
`os.replace`s. Same-directory placement is load-bearing: guarantees same filesystem so the POSIX rename cannot
fall back to a copy.

This module is the package-level home for the helper, per D-009 in
this repo (matches D-015 in `llm-eval-harness` and the portfolio
standard `rag_kit/io_utils.atomic_write_text` from
`rag-production-kit#44/#45`).
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import secrets
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Cap the target basename's contribution to the temp filename. The temp name is
# `.<base>.<random>.tmp`; the affixes add ~13-20 bytes, so prepending a full
# basename that is itself near NAME_MAX (255 on ext4/APFS) overflows the limit
# and the write fails with `OSError: [Errno 63] File name too long` — even though
# a plain `Path.write_text` of that same target succeeds (sibling of
# rag-production-kit#128 and mcp-server-cookbook#96). The base in the temp name
# is cosmetic (`ls`-ability); uniqueness comes from `_create_temp`'s random
# component, so truncating it is safe. Budget is in BYTES (NAME_MAX is a byte
# limit) and we trim on a char boundary so multibyte names are never split
# mid-codepoint.
_MAX_TEMP_BASE_BYTES = 200


def _name_bytes(base: str) -> int:
    """Length of *base* in the bytes the filesystem actually sees.

    `os.fsencode`, not `base.encode("utf-8")` (#135). Both halves of the
    comment above are true and the old implementation still counted the wrong
    bytes: NAME_MAX limits the bytes handed to the kernel, which is
    `os.fsencode` — `sys.getfilesystemencoding()` together with
    `sys.getfilesystemencodeerrors()`, i.e. `surrogateescape` on POSIX.

    That handler is why the distinction bites rather than being pedantry. A
    path byte that is not valid UTF-8 arrives in Python as a lone surrogate in
    `U+DC80..U+DCFF`, and strict `str.encode("utf-8")` refuses to encode it —
    so `_cap_base_for_temp` used to raise `UnicodeEncodeError` on a destination
    the OS can name, *before* reaching the length question. `sys.argv` decodes
    with the same handler, so a shell `--out $'report\\xff.json'` is
    enough.

    `UnicodeEncodeError` is a `ValueError`, so none of the three write-seam
    guards catches it. `_cmd_sweep_run`'s says it is "the one write seam
    #75/#87 never reached; the sibling seams ... already honor the exit-2
    write-failure contract" — true, and counted in *seams*. The population is
    ways an operator-supplied `--out` can fail to be written, and an
    unencodable name is a member of it that is not an `OSError`, so all three
    guards let it through together. For `corpus validate` that is not just
    noise: the command ends `return 0 if report.ok else 1`, so the leaked
    exit 1 is the documented "the corpus has findings".

    `os.fsencode` never raises: `surrogateescape` on POSIX, `surrogatepass` on
    Windows, so every `str` a `Path` can hold round-trips. For a name that is
    valid UTF-8 it returns exactly the old number, so the budget is unchanged
    for every name that worked before.
    """
    return len(os.fsencode(base))


def _cap_base_for_temp(base: str) -> str:
    if _name_bytes(base) <= _MAX_TEMP_BASE_BYTES:
        return base
    out = base
    while out and _name_bytes(out) > _MAX_TEMP_BASE_BYTES:
        out = out[:-1]
    return out


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """Write *text* to *path* atomically.

    On success the destination contains exactly *text*. On any failure
    path (signal, disk-full, OOM during flush), the destination is
    either unchanged (overwrite case) or absent (new-file case) —
    never partial.

    Parent directories are created with `mkdir(parents=True,
    exist_ok=True)` so callers don't have to gate on the parent
    themselves.

    A symlinked destination is written THROUGH, as `Path.write_text` does
    (#195): see `_resolve_symlinked_target`.
    """
    _atomic_write(Path(path), text, "w", encoding)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """`atomic_write_text` for bytes: the same never-partial write (#245).

    `sweep plot` called `fig.savefig(path)` straight onto the committed
    `docs/pareto.png`/`.svg`, so a write that failed partway (ENOSPC, EFBIG)
    left a truncated chart where the previous one had been -- measured 58336
    bytes -> 4096 with the file size capped at 4 KB. The chart is rendered into
    memory and handed here instead. One implementation with the text writer, so
    the symlink write-through (#195), the mode copy (#165) and the temp-file
    cleanup cannot drift between the two.
    """
    _atomic_write(Path(path), data, "wb", None)


def atomic_write_bytes_many(items: Sequence[tuple[str | Path, bytes]]) -> None:
    """`atomic_write_bytes` for several files that must change together (#251).

    `sweep plot --out-png X --out-svg Y` wrote the two charts one after the
    other, each atomically, so a disk-full on the SVG came after the PNG had
    been replaced: the command exited 2 with the two charts plotting different
    results (measured on a 2 MB HFS+ image filled to 28 KB free). Every file
    is staged first -- written, fsynced and given its target's mode in a temp
    file beside the target -- and the renames start only once all of them are
    staged, so a failure while staging (ENOSPC, EFBIG, an unwritable parent)
    leaves every target as it was and no temp file behind.

    The renames themselves are not one atomic step across files; what makes
    the second one fail after the first succeeded (an immutable target, a
    sticky directory) is what `check_writable` refuses up front, and
    `render_pareto` preflights every chart with it.

    An `OSError` with no filename -- a disk-full on `write`/`fsync` names
    nothing -- is re-raised with the path the caller passed, so the error says
    which of the files could not be written.
    """
    staged: list[tuple[Path, Path, Path]] = []  # (as passed, target, temp)
    try:
        for path, data in items:
            try:
                target, tmp_path = _stage(Path(path), data, "wb", None)
            except OSError as e:
                raise _naming(e, path) from e
            staged.append((Path(path), target, tmp_path))
        while staged:
            path, target, tmp_path = staged[0]
            try:
                os.replace(tmp_path, target)
            except OSError as e:
                raise _naming(e, path) from e
            staged.pop(0)
    finally:
        for _path, _target, tmp_path in staged:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()


def _naming(e: OSError, path: str | Path) -> OSError:
    """*e*, or the same error with *path* as its filename if it names no file (#251)."""
    if e.filename is not None or e.errno is None:
        return e
    return type(e)(e.errno, e.strerror, os.fspath(path))


def _atomic_write(path: Path, payload: str | bytes, mode: str, encoding: str | None) -> None:
    target, tmp_path = _stage(path, payload, mode, encoding)
    try:
        os.replace(tmp_path, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise


def _stage(path: Path, payload: str | bytes, mode: str, encoding: str | None) -> tuple[Path, Path]:
    """Write *payload* to a temp file beside *path*'s target; return `(target, temp)`.

    Everything `_atomic_write` does short of the rename, split out so
    `atomic_write_bytes_many` can stage several files before renaming any
    (#251). On failure the temp file is removed and nothing is returned.
    """
    target = _resolve_symlinked_target(path)
    # A read-only existing target is refused, as `Path.write_text` refuses it,
    # instead of being renamed over (#277). See `_refuse_unwritable_target`.
    _refuse_unwritable_target(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = _create_temp(target)
    try:
        # `os.fdopen` owns *fd* from here: if it raises (unknown codec), `io.open`
        # has already closed it, and the `except` below removes the temp file.
        with os.fdopen(fd, mode, encoding=encoding) as tmp:
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
        _copy_existing_mode(target, tmp_path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise
    return target, tmp_path


def check_writable(path: str | Path) -> None:
    """Raise the `OSError` `atomic_write_text(path, ...)` would, without writing (#200).

    A preflight for a write that comes after paid work: `sweep run --out` was
    touched only once the provider had embedded the whole corpus and every
    query, so an unwritable path billed a full sweep and then discarded it.
    This does what the writer does -- the same symlink resolution (#195), the
    same parent `mkdir`, the same exclusively-created temp file beside the
    target -- and removes the temp file again, so a path passes exactly when
    the real write would get that far. An existing directory is refused too:
    the writer's final `os.replace` onto it would fail. llm-eval-harness#287's
    helper, ported against this module's own writer.

    The same goes for an existing target the rename is refused onto (#251):
    one flagged immutable or append-only (`chflags uchg`; BSD/macOS expose the
    flags in `st_flags`), or one in a sticky directory (`/tmp`) that belongs
    to another user. Both passed and then failed with EPERM at the
    `os.replace`, after `sweep run` had embedded the whole sweep. Linux keeps
    `chattr +i` out of `stat` (it takes an `FS_IOC_GETFLAGS` ioctl), so that
    one is still only found by the write.

    A read-only existing target (`chmod 444`) is refused with the `EACCES` the
    writer now raises for it (#277), see `_refuse_unwritable_target`.
    """
    target = _resolve_symlinked_target(Path(path))
    if target.is_dir():
        raise IsADirectoryError(21, "Is a directory", str(target))
    try:
        st = os.stat(target)  # what `_copy_existing_mode` stats: a link loop raises here
    except FileNotFoundError:
        st = None
    target.parent.mkdir(parents=True, exist_ok=True)
    if st is not None:
        _refuse_unwritable_target(target)
        _check_replaceable(target, st)
    fd, tmp_path = _create_temp(target)
    os.close(fd)
    tmp_path.unlink()


def _refuse_unwritable_target(target: Path) -> None:
    """Raise `PermissionError` if *target* exists and cannot be written (#277).

    `os.replace` swaps the directory ENTRY, so only the directory's write bit
    is checked and the target's own is never consulted: a `chmod 444` file
    was silently rewritten by all three writers, mode still 0444, while
    `Path.write_text`, the call they replaced, opens the file for writing and
    fails with `EACCES`. Raising the same error here, before the temp file
    exists, keeps that contract and leaves nothing behind; the CLI's
    `except OSError` write seams turn it into `failed to write` + exit 2, as
    their comments already said they did for a "read-only path". In
    `atomic_write_bytes_many` it fires while staging, so no target has been
    renamed yet. `check_writable` runs it too, so the preflight does not
    approve a path the writer then refuses.

    `os.access` asks the kernel the same question the open would, so root
    (which `Path.write_text` lets through) still writes. The `os.stat` first
    keeps a missing target -- the create case -- out of the check, and lets a
    link loop raise its own ELOOP rather than a misleading `EACCES`. Like
    `Path.write_text`, this is a check at open time, not a lock: a `chmod`
    racing the write is not covered. prompt-regression-suite#318's helper.
    """
    try:
        os.stat(target)
    except FileNotFoundError:
        return
    if not os.access(target, os.W_OK):
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(target))


# The `st_flags` bits that make `rename(2)` onto a file fail with EPERM (#251).
# `getattr`: `stat` has these constants only where `st_flags` exists (BSD/macOS).
_NO_REPLACE_FLAGS = 0
for _name in ("UF_IMMUTABLE", "SF_IMMUTABLE", "UF_APPEND", "SF_APPEND"):
    _NO_REPLACE_FLAGS |= getattr(stat, _name, 0)


def _check_replaceable(target: Path, st: os.stat_result) -> None:
    """Raise the EPERM `os.replace(tmp, target)` would, for an existing *target* (#251)."""
    if getattr(st, "st_flags", 0) & _NO_REPLACE_FLAGS:
        raise PermissionError(
            1, "Operation not permitted (file is immutable or append-only)", str(target)
        )
    # A sticky directory lets only the file's owner, the directory's owner and
    # root replace a file in it (sticky(7)); the temp file's own create is fine.
    euid = os.geteuid() if hasattr(os, "geteuid") else None
    if euid is None or euid == 0 or st.st_uid == euid:
        return
    parent = os.stat(target.parent)
    if parent.st_mode & stat.S_ISVTX and parent.st_uid != euid:
        raise PermissionError(
            1, "Operation not permitted (another user's file in a sticky directory)", str(target)
        )


# File mode (#165, portfolio-ops#81). This helper used to create its temp file
# with `tempfile.NamedTemporaryFile`, which always opens 0600 regardless of the
# umask, and `os.replace` carries the temp file's mode onto the target. So every
# file it wrote was owner-only, and an overwrite of an existing 0644 file
# demoted it to 0600 — neither of which the `Path.write_text` it replaced did.
# The temp file is now opened with mode 0o666 so the KERNEL applies the umask
# (reading the umask via `os.umask(0); os.umask(old)` would briefly set a
# process-wide umask of 0 for every other thread), and an existing target's
# mode is copied onto the temp file before the rename.
_TEMP_ATTEMPTS = 100


def _create_temp(target: Path) -> tuple[int, Path]:
    """Exclusively create `.<capped-base>.<random>.tmp` beside *target*.

    Same name shape and random-component length (8 chars) as the
    `NamedTemporaryFile` it replaces, so `_MAX_TEMP_BASE_BYTES` budgets it
    unchanged. `O_EXCL` gives the same no-clobber guarantee; a collision just
    draws another name.
    """
    prefix = f".{_cap_base_for_temp(target.name)}."
    for _ in range(_TEMP_ATTEMPTS):
        candidate = target.parent / f"{prefix}{secrets.token_hex(4)}.tmp"
        try:
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            continue
        return fd, candidate
    raise FileExistsError(f"could not create a unique temp file beside {target}")


def _resolve_symlinked_target(target: Path) -> Path:
    """The file a write to *target* lands in: through a symlink, as `write_text` does (#195).

    `os.replace` renames onto the LINK, not the file it points at. So a
    symlinked `--out` became a regular file and the linked file kept its old
    contents, while the `Path.write_text` this helper replaced writes through
    the link. `_copy_existing_mode` already followed the link (`os.stat`), so
    the linked file's mode was copied onto a file that then replaced the link
    instead (sibling of python-async-llm-pipelines#157).

    Resolving here also places the temp file beside the RESOLVED file, which
    keeps the rename on one filesystem when the link points at a different
    one, and hands `_create_temp` the resolved basename to cap. A dangling
    link resolves to the path it names, and the write creates that file, as
    `write_text` would. With a link loop, non-strict `realpath` returns the
    path unresolved, and the `os.stat` in `_copy_existing_mode` raises
    `OSError` (ELOOP), which the CLI write-seam guards translate to exit 2. A
    plain path comes back unchanged.
    """
    if not target.is_symlink():
        return target
    return Path(os.path.realpath(target))


def _copy_existing_mode(target: Path, tmp_path: Path) -> None:
    """Give *tmp_path* the permission bits of *target*, if *target* exists."""
    try:
        mode = stat.S_IMODE(os.stat(target).st_mode)
    except FileNotFoundError:
        return
    os.chmod(tmp_path, mode)


def loads_json(text: str) -> Any:
    """`json.loads`, with every way bad input fails raised as a `JSONDecodeError` (#204).

    Every reader in this package caught `JSONDecodeError` only, and `json.loads`
    raises two other things on bad input: `RecursionError` on nesting deeper
    than the interpreter's stack allows (about 100,000 levels of ``[`` on
    3.11/3.12, which CI runs), and a plain `ValueError` for an integer literal
    past CPython's 4300-digit int-string limit. Both escaped as a traceback at
    exit 1, the code `corpus validate` uses for "findings", with no finding
    printed and the rows after it unchecked.

    Re-raised as a `JSONDecodeError` so each site's existing handler applies,
    exit 2 in the CLI and `malformed_json` in the validator. One helper is the
    point, as in llm-eval-harness#333: the next reader cannot catch one of the
    three and miss the others. A test pins that no other module calls
    `json.loads`.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise
    except RecursionError:
        raise json.JSONDecodeError("nested too deeply to parse", text, 0) from None
    except ValueError as e:
        raise json.JSONDecodeError(str(e), text, 0) from None
