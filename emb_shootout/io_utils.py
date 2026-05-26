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
            prefix=f".{target.name}.",
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
