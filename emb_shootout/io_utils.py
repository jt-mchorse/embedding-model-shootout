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

`atomic_write_text` writes to a sibling temp file in the same
directory, `fsync`s, then `os.replace`s. Same-directory placement is
load-bearing: guarantees same filesystem so the POSIX rename cannot
fall back to a copy.

This module is the package-level home for the helper, per D-009 in
this repo (matches D-015 in `llm-eval-harness` and the portfolio
standard `rag_kit/io_utils.atomic_write_text` from
`rag-production-kit#44/#45`).
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

# Cap the target basename's contribution to the temp filename. The temp name is
# `.<base>.<random>.tmp`; the affixes add ~13-20 bytes, so prepending a full
# basename that is itself near NAME_MAX (255 on ext4/APFS) overflows the limit
# and the write fails with `OSError: [Errno 63] File name too long` — even though
# a plain `Path.write_text` of that same target succeeds (sibling of
# rag-production-kit#128 and mcp-server-cookbook#96). The base in the temp name
# is cosmetic (`ls`-ability); uniqueness comes from `NamedTemporaryFile`'s random
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
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=target.parent,
            prefix=f".{_cap_base_for_temp(target.name)}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, target)
        tmp_path = None
    finally:
        if tmp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
