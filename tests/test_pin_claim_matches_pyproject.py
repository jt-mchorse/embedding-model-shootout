"""The docs may call the corpus interpreter "pinned" only if something pins it (#160).

README.md and the architecture diagram said, in the present tense, that the
corpus is built "on a pinned Python version". Nothing pinned it:
`requires-python = ">=3.11"`, and CI runs 3.11 and 3.12. The README's own
reproducibility section measured the consequence (3.11 builds 11 108 chunks,
3.14 builds 12 010) and leaves the remedy open as #115.

This ties the phrase to the fact rather than to #115's outcome. If #115 pins
a single minor version in `requires-python`, the phrase becomes true and this
test allows it again. The D-002 citations ("the pinned-CPython corpus
(D-002)") name the decision as recorded and are not matched.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = (REPO_ROOT / "README.md", REPO_ROOT / "docs" / "architecture.md")
CLAIMS = ("pinned python version", "pinned interpreter")


def _requires_python() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["requires-python"]


def _pins_one_minor(spec: str) -> bool:
    """`==3.14.*`, `==3.14`, `~=3.14.0`: one minor version and nothing else."""
    return bool(re.fullmatch(r"\s*(?:==\s*3\.\d+(?:\.\*)?|~=\s*3\.\d+\.\d+)\s*", spec))


def test_the_pin_predicate_reads_real_specifiers() -> None:
    # The predicate is the whole test; pin its reading of both kinds of spec.
    assert not _pins_one_minor(">=3.11")
    assert not _pins_one_minor(">=3.11,<3.15")
    assert _pins_one_minor("==3.14.*")
    assert _pins_one_minor("~=3.14.0")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_docs_do_not_claim_a_pin_that_does_not_exist(doc: Path) -> None:
    if _pins_one_minor(_requires_python()):
        pytest.skip("requires-python pins one minor version; the claim is true")
    # Collapse whitespace so a claim wrapped across lines is still one phrase.
    text = " ".join(doc.read_text(encoding="utf-8").lower().split())
    found = [c for c in CLAIMS if c in text]
    assert not found, (
        f"{doc.name} says {found}, but requires-python is {_requires_python()!r}. "
        "Nothing pins the corpus interpreter (#115); describe what happens today."
    )


def test_the_reproducibility_section_still_names_the_open_question() -> None:
    # The premise of the wording above: the README says where the pin is decided.
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "open as issue #115" in text
