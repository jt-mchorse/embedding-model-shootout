"""Public-surface tests for ``emb_shootout/__init__.py``.

``emb_shootout`` re-exports 12 names from four submodules (``corpus``,
``providers``, ``queries``, ``sweep``) and declares them in ``__all__`` +
``__version__``. The other three submodules (``cli``, ``pareto``,
``plot``) are *intentionally* not re-exported at the top level — they
are accessed via dotted path (e.g., the README cites
``emb_shootout.pareto.pareto_frontier`` directly). Every other test in
this suite imports submodules directly, so silent renames or accidental
``__all__`` drops in ``__init__.py`` don't fail any test.

These four standalone + 4 parametrized tests lock the surface:

1. ``__version__`` is set to a semver-ish string.
2. Every name in ``__all__`` is bound on the package and non-None.
3. ``__all__`` agrees with the actual top-level relative ``from .X import
   …`` names (filter on ``level >= 1`` — same adaptation
   ``prompt-regression-suite#20`` and ``rag-production-kit#24`` used).
4. The README's quoted dotted-path ``emb_shootout.pareto.pareto_frontier``
   resolves to a callable — guards against ``pareto`` being renamed or
   ``pareto_frontier`` being moved without updating the README.
5. One anchor per *re-exported* submodule (4 anchors) survives at the
   top level. ``pareto``/``plot``/``cli`` are deliberately excluded —
   they are dotted-path-only by design.

Same hygiene posture as the public-surface snapshots landed in
``llm-eval-harness`` (#25), ``llm-cost-optimizer`` (#23),
``prompt-regression-suite`` (#20), and ``rag-production-kit`` (#24)
this week. Orthogonal to ``tests/test_readme_snapshot.py``, which locks
the README's *numeric* claims (the Takeaways table); this test locks
the *Python public surface* the README implicitly depends on.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

import emb_shootout

_INIT_PATH = Path(emb_shootout.__file__)
_SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[-+].+)?$")

# README quotes one Python dotted path by name; lock it so a rename of
# ``pareto.py`` or a move of ``pareto_frontier`` fails loudly here
# rather than silently desyncing the README's prose.
README_DOTTED_PATHS = (("emb_shootout.pareto", "pareto_frontier"),)

# Anchor names that prove each *re-exported* submodule survived. One
# name per submodule; ``cli``/``pareto``/``plot`` are intentionally
# excluded — they are accessed via dotted path, not re-exported.
SUBMODULE_ANCHORS = {
    "corpus": "build_corpus",
    "providers": "HashEmbedderProvider",
    "queries": "build_queries",
    "sweep": "run_sweep",
}


def _parse_init_relative_imports() -> set[str]:
    """Return the set of names imported into ``__init__.py`` via
    top-level relative ``from .X import (...)`` blocks."""
    tree = ast.parse(_INIT_PATH.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level >= 1:
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def test_version_is_set_to_semver_ish_string() -> None:
    """``__version__`` is published; downstream importers and PyPI
    builds rely on it."""
    assert hasattr(emb_shootout, "__version__"), (
        "emb_shootout.__version__ is missing — packaging tools and "
        "downstream `emb_shootout.__version__` lookups will break."
    )
    version = emb_shootout.__version__
    assert isinstance(version, str), (
        f"emb_shootout.__version__ should be a string, got {type(version).__name__}: {version!r}."
    )
    assert version, "emb_shootout.__version__ is an empty string."
    assert _SEMVER_PATTERN.match(version), (
        f"emb_shootout.__version__ = {version!r} doesn't look like "
        f"semver (expected MAJOR.MINOR.PATCH[-prerelease][+build])."
    )


def test_all_names_are_bound_and_non_none() -> None:
    """Every name in ``__all__`` must be importable and non-None."""
    missing: list[str] = []
    none_valued: list[str] = []
    for name in emb_shootout.__all__:
        if not hasattr(emb_shootout, name):
            missing.append(name)
            continue
        if getattr(emb_shootout, name) is None:
            none_valued.append(name)
    assert not missing, (
        f"emb_shootout.__all__ advertises names that are not bound on "
        f"the package: {missing}. The most likely cause is a re-import "
        f"line was deleted from __init__.py but __all__ wasn't updated."
    )
    assert not none_valued, (
        f"emb_shootout.__all__ entries bound to None: {none_valued}. "
        f"A re-import probably resolved to a missing submodule attribute."
    )


def test_all_matches_actual_top_level_imports() -> None:
    """``__all__`` should equal the set of top-level relative re-exports."""
    advertised = set(emb_shootout.__all__)
    imported = _parse_init_relative_imports()
    only_imported = imported - advertised
    only_advertised = advertised - imported
    assert not only_imported, (
        f"Names imported into emb_shootout/__init__.py but missing from "
        f"__all__: {sorted(only_imported)}. Add them to __all__ or stop "
        f"importing them at the top level."
    )
    assert not only_advertised, (
        f"Names in emb_shootout.__all__ but not imported at the top of "
        f"__init__.py: {sorted(only_advertised)}. Add the import or "
        f"remove the __all__ entry."
    )


@pytest.mark.parametrize(
    ("module_path", "attr"),
    README_DOTTED_PATHS,
    ids=[f"{m}.{a}" for m, a in README_DOTTED_PATHS],
)
def test_readme_dotted_path_resolves(module_path: str, attr: str) -> None:
    """README's quoted ``emb_shootout.pareto.pareto_frontier`` must
    keep resolving to a callable.

    The README literally quotes (line 210)::

        the *frontier-selection* code (`emb_shootout.pareto.pareto_frontier`)
        is pure-stdlib Python and ships in the base install

    If ``pareto.py`` is renamed or ``pareto_frontier`` is moved, that
    sentence silently lies. Locking the lookup here keeps prose ↔ code
    in sync.
    """
    module = importlib.import_module(module_path)
    assert hasattr(module, attr), (
        f"`{module_path}.{attr}` no longer resolves. The README quotes "
        f"it by name (around line 210) — either restore the export or "
        f"update the README."
    )
    obj = getattr(module, attr)
    assert callable(obj), (
        f"`{module_path}.{attr}` is no longer callable (got "
        f"{type(obj).__name__}). The README describes it as the "
        f"frontier-selection function; the lookup must return a callable."
    )


@pytest.mark.parametrize(
    ("submodule", "anchor"),
    sorted(SUBMODULE_ANCHORS.items()),
    ids=sorted(SUBMODULE_ANCHORS.keys()),
)
def test_submodule_anchor_re_exported(submodule: str, anchor: str) -> None:
    """One anchor per *re-exported* submodule survives at the top level.

    ``cli``/``pareto``/``plot`` are intentionally NOT in this map —
    they are accessed via dotted path and re-exporting them at the top
    level would expand the public surface without an explicit decision.
    """
    assert hasattr(emb_shootout, anchor), (
        f"`{anchor}` from `emb_shootout.{submodule}` is no longer "
        f"re-exported at the top level. Did `{submodule}` move or get "
        f"renamed? Update `emb_shootout/__init__.py` to re-export from "
        f"the new path."
    )
