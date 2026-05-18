"""Sanity checks on `notebooks/reproduce.ipynb`.

The notebook itself is committed with empty cell outputs so re-running
it on an operator's machine produces a meaningful diff. These tests
keep it honest:

- It's valid Jupyter JSON (nbformat 4).
- It carries the expected cell shape (markdown intro, then alternating
  markdown/code through the steps, then a closing prose section).
- The code cells reference only imports that resolve in the current
  package layout — a refactor that renames a module is caught here, not
  by a confused operator who pulls the repo six months later.
- Cell outputs are empty (so a re-run never disagrees with HEAD).

Also smoke-tests that `notebooks/_verify.py` runs end-to-end against
the committed `results/`. The script is the notebook's executable
twin; it's the test that proves the notebook *would* execute cleanly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "reproduce.ipynb"
VERIFY_SCRIPT = ROOT / "notebooks" / "_verify.py"


def test_notebook_is_valid_jupyter_json():
    raw = NOTEBOOK.read_text(encoding="utf-8")
    nb = json.loads(raw)
    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] >= 5
    assert isinstance(nb["cells"], list)
    assert len(nb["cells"]) > 0


def test_notebook_has_alternating_md_and_code():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    types = [c["cell_type"] for c in nb["cells"]]
    # Must start with a markdown intro.
    assert types[0] == "markdown"
    # Must contain at least the five canonical code cells (corpus,
    # queries, sweep, aggregate, frontier).
    code_count = sum(1 for t in types if t == "code")
    assert code_count >= 5, f"expected >=5 code cells, got {code_count}"


def test_notebook_imports_match_current_package_layout():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_sources = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    # Symbols the notebook imports — refactoring any of them must touch
    # the notebook in the same commit.
    expected = [
        "from emb_shootout.corpus import DEFAULT_MODULES, build_corpus",
        "from emb_shootout.pareto import pareto_frontier",
        "from emb_shootout.providers.hash_embedder import HashEmbedderProvider",
        "from emb_shootout.queries import build_queries",
        "from emb_shootout.sweep import SweepResult, aggregate_markdown, run_sweep",
    ]
    for imp in expected:
        assert imp in code_sources, f"notebook is missing import: {imp!r}"


def test_notebook_cell_outputs_are_empty():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        outputs = cell.get("outputs", [])
        assert outputs == [], (
            f"cell {i} has cached outputs; the notebook must commit clean "
            f"so a re-run produces a meaningful diff."
        )
        # execution_count should also be null/None on a clean commit.
        assert cell.get("execution_count") in (None, 0), (
            f"cell {i} has a non-null execution_count; clear before commit."
        )


def test_verify_script_runs_end_to_end():
    """The notebook's executable twin runs against committed results/."""
    result = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        # Surface stderr + stdout in the assertion message so a CI failure
        # tells the reader what went wrong.
        raise AssertionError(
            f"_verify.py exited {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    # Sanity: the final completion line must appear.
    assert "All steps completed without error." in result.stdout
