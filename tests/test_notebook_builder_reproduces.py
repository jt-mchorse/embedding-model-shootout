"""Regenerating the notebook reproduces the committed file byte for byte (#265).

`notebooks/_build_notebook.py` is the notebook's source of truth (#197: a hand
edit is one "the builder would have undone"). Measured on `main` (5fb9d89):

    python notebooks/_build_notebook.py && ruff format --check .
    -> Would reformat: notebooks/reproduce.ipynb   (rc 1)
    git diff --stat -> 48+/60-: every cell "id" dropped, em dashes and
       middle dots \\u-escaped, single-quoted code in cells 3/5/7/9/11/13

so the committed notebook only passed CI because it had been reformatted and
given ids after generation, and running the documented builder broke the lint
job. Comparing against the committed bytes covers all three at once: that file
is what CI's `ruff format --check .` passes.

Stdlib only. CI installs `.[dev]`, which has no nbformat, so the schema arm
skips there and the id-shape arm checks the same rule by hand.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "reproduce.ipynb"
BUILDER = ROOT / "notebooks" / "_build_notebook.py"

# nbformat 4.5's cell-id rule (nbformat/v4/nbformat.v4.5.schema.json).
CELL_ID = re.compile(r"^[a-zA-Z0-9-_]{1,64}$")


def _builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_build_notebook_265", BUILDER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_equals_the_committed_notebook() -> None:
    assert _builder().render() == NOTEBOOK.read_text(encoding="utf-8")


def test_running_the_builder_rewrites_the_committed_bytes(tmp_path: Path) -> None:
    """The documented command, run in a scratch copy so the checkout is untouched."""
    shutil.copytree(ROOT / "notebooks", tmp_path / "notebooks")
    target = tmp_path / "notebooks" / "reproduce.ipynb"
    target.write_text("{}\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(tmp_path / "notebooks" / "_build_notebook.py")],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert target.read_bytes() == NOTEBOOK.read_bytes()


def test_every_cell_has_a_unique_valid_id() -> None:
    cells = _builder().build()["cells"]
    ids = [cell.get("id") for cell in cells]
    assert all(isinstance(i, str) and CELL_ID.match(i) for i in ids), ids
    assert len(set(ids)) == len(ids), ids


def test_non_ascii_is_written_literally() -> None:
    text = _builder().render()
    assert "—" in text
    assert "·" in text
    assert "\\u" not in text


def test_build_refuses_a_duplicated_cell_id(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _builder()
    first = module.CELLS[0]
    monkeypatch.setattr(module, "CELLS", [*module.CELLS, dict(first)])
    with pytest.raises(ValueError, match=first["id"]):
        module.build()


def test_committed_notebook_validates_under_nbformat() -> None:
    nbformat = pytest.importorskip("nbformat")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nb = nbformat.reads(_builder().render(), as_version=4)
        nbformat.validate(nb)


def test_committed_notebook_is_parseable_json_with_minor_5() -> None:
    """Control, GREEN against the unfixed tree: the version the ids are required by."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert (nb["nbformat"], nb["nbformat_minor"]) == (4, 5)
