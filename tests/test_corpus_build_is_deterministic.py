"""Two corpus builds on one interpreter are the same corpus (#182).

`docs/corpus.md` promises "Same Python version + same module list = same
corpus, deterministically". On main, three builds on one 3.14.7 venv gave three
fingerprints: 51 chunks carried memory addresses from signature defaults
(`timeout=<object object at 0x1008b4890>`, `<weakref at 0x...; to '...'>`), so
`sweep aggregate` refused rows swept on separate builds.
"""

from __future__ import annotations

import os
import subprocess
import sys
import weakref
from pathlib import Path

from emb_shootout.corpus import _safe_signature

_SENTINEL = object()


class _Target:
    pass


_REF = weakref.ref(_Target)


def _with_sentinel(timeout=_SENTINEL):  # noqa: ANN001, ANN202
    return timeout


def _with_weakref(handlers=[_REF]):  # noqa: ANN001, ANN202, B006
    return handlers


def test_a_signature_names_no_memory_address() -> None:
    for fn in (_with_sentinel, _with_weakref):
        sig = _safe_signature(fn)
        assert " at 0x" not in sig, sig
        assert sig.startswith("(")


def test_two_builds_in_separate_processes_are_byte_identical(tmp_path: Path) -> None:
    outs = []
    for seed in ("1", "2"):
        out = tmp_path / f"corpus{seed}.jsonl"
        subprocess.run(
            [sys.executable, "-m", "emb_shootout.cli", "corpus", "build", "--out", str(out)],
            check=True,
            capture_output=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]
    assert b" at 0x" not in outs[0]
