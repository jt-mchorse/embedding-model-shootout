"""Smoke test for `scripts/capture_demo.sh` (issue #15).

The capture script is the deterministic driver for the 60-second README
demo. JT records the GIF/video while it runs; CI runs it with
`CAPTURE_PACE_SECONDS=0` to make sure the demo can't bitrot the same way
`tests/test_benchmarks_md_snapshot.py` protects the per-surface
aggregator format in isolation.

Contract this test pins:

1. The script exits 0 on a fresh clone with no API key.
2. Each of the three surfaces actually runs (the distinctive header +
   the surface's output line both appear).
3. The aggregator step emits the same markdown header signature that
   `docs/benchmarks.md` carries — the script's job is to show the
   viewer that the table is real, not a hand-rolled mock.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "capture_demo.sh"


@pytest.fixture(scope="module")
def capture_output() -> str:
    """Run the capture script once and reuse its stdout across assertions.

    `CAPTURE_PACE_SECONDS=0` removes the recording pauses. We let the
    other knobs (module, query count) take their defaults so the test
    locks the default-take recording — overrides are for operator
    flexibility across takes, not the contract.
    """
    if not SCRIPT.exists():
        pytest.fail(f"missing {SCRIPT}")
    if shutil.which("bash") is None:
        pytest.skip("bash not available")

    env = dict(os.environ)
    env["CAPTURE_PACE_SECONDS"] = "0"
    # Ensure `emb-shootout` resolves via the same interpreter pytest is
    # using — capture_demo.sh shells out to `emb-shootout`, so put the
    # active venv's bin first on PATH.
    venv_bin = Path(sys.executable).parent
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"capture_demo.sh exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return result.stdout


def test_surface_1_corpus_build(capture_output: str) -> None:
    assert "1/3 · corpus build" in capture_output
    # The CLI's corpus_build emits a JSON summary on stdout — `chunk_count`
    # and `out` are stable keys (locked by test_cli.py + test_corpus.py).
    # The `json` stdlib module has a small, stable surface; the chunk
    # count being non-zero is the contract, not the exact number.
    assert '"chunk_count":' in capture_output
    assert '"modules_requested": 1' in capture_output
    assert '"out":' in capture_output


def test_surface_2_sweep_run(capture_output: str) -> None:
    assert "2/3 · sweep run" in capture_output
    # The sweep CLI emits "<embedder>: recall@5=<N> NDCG@10=<N> → <path>"
    # on stdout (see emb_shootout/cli.py::_cmd_sweep_run). Pin the
    # provider name + the two stat keys; the values vary with corpus
    # size and query count and are not part of this contract.
    assert "hash-embedder-128d-ngram2" in capture_output
    assert "recall@5=" in capture_output
    assert "NDCG@10=" in capture_output


def test_surface_3_aggregate_renders_benchmarks_md_header(capture_output: str) -> None:
    """The rendered table is the load-bearing artifact for the viewer.

    Lock the exact header row that `docs/benchmarks.md` ships (and that
    `tests/test_benchmarks_md_snapshot.py` already locks elsewhere) —
    if the aggregator format ever drifts, this test catches it via the
    capture path too, not just via the on-disk file.
    """
    assert "3/3 · aggregate" in capture_output
    assert "aggregated 1 results" in capture_output
    expected_header = (
        "| embedder | dim | n_corpus | n_queries | recall@1 | recall@5 "
        "| recall@10 | NDCG@10 | corpus embed (ms) "
        "| query p50 (ms) | query p95 (ms) | $/1M tokens |"
    )
    assert expected_header in capture_output, (
        "aggregator markdown header drifted; "
        "test_benchmarks_md_snapshot.py and this test must agree"
    )
    # The data row carries the hash baseline name — proves the table is
    # populated, not just the empty header rendered.
    assert "| hash-embedder-128d-ngram2 |" in capture_output


def test_capture_demo_script_exists_and_is_executable() -> None:
    assert SCRIPT.exists(), f"missing {SCRIPT}"
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be executable"
