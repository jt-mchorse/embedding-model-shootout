"""Every install hint names this distribution and an extra it has (#158).

Five providers told a user missing their SDK to `pip install
'emb-shootout[openai]'` (and voyage/cohere/sbert). There is no such
distribution -- `emb-shootout` is the console script; the distribution is
`embedding-model-shootout` -- so the fix they were handed failed with "No
matching distribution". The `all-providers` extra self-referenced the same
wrong name and could not install at all.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_PROJECT = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
_NAME = _PROJECT["name"]
_EXTRAS = set(_PROJECT["optional-dependencies"])

_HINT = re.compile(r"pip install (?:-e )?'([^'\[]*)\[([a-z0-9,-]+)\]'")


def _surfaces() -> list[Path]:
    paths = [
        *(_ROOT / "emb_shootout").rglob("*.py"),
        _ROOT / "README.md",
        _ROOT / ".env.example",
        *(_ROOT / "docs").glob("*.md"),
    ]
    return sorted(p for p in paths if p.exists())


def _hints() -> list[tuple[str, str, str]]:
    out = []
    for path in _surfaces():
        for target, extras in _HINT.findall(path.read_text(encoding="utf-8")):
            out.append((str(path.relative_to(_ROOT)), target, extras))
    return out


def test_hints_are_found() -> None:
    """A pass over no hints is not a pass: five providers and the README carry one."""
    assert len(_hints()) >= 6


@pytest.mark.parametrize(("where", "target", "extras"), _hints(), ids=lambda v: str(v)[:40])
def test_each_hint_installs(where: str, target: str, extras: str) -> None:
    assert target in {".", _NAME}, f"{where}: installs {target!r}, not '.' or {_NAME!r}"
    missing = set(extras.split(",")) - _EXTRAS
    assert not missing, f"{where}: extras {sorted(missing)} do not exist in pyproject.toml"


def test_the_all_providers_extra_self_references_this_distribution() -> None:
    for req in _PROJECT["optional-dependencies"]["all-providers"]:
        name = re.split(r"[\[<>=!~; ]", req, maxsplit=1)[0]
        assert name == _NAME, f"all-providers requires {name!r}, not this distribution {_NAME!r}"
