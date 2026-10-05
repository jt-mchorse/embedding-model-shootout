"""`build_corpus` and every provider's `embed` refuse a bare string (#169).

A `str` is an `Iterable[str]` and a `Sequence[str]`, so both annotations admit
it. Measured on `main` before this change:

    len(list(build_corpus("json")))                  0    (vs 20 for ["json"])
    len(HashEmbedderProvider().embed("hello world"))  11   (vs 1 for ["hello world"])

`build_corpus` skips a module that fails to import *by design*, so "j", "s",
"o", "n" each vanished silently and the corpus came back empty. On the three
API providers, `embed("...")` is a billed request per character batch.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

import pytest

import emb_shootout.providers as providers_pkg
from emb_shootout import HashEmbedderProvider, build_corpus

_SPLIT = "would be split into its characters"
_BARE = ["json", b"json", bytearray(b"j"), ""]


def _provider_classes() -> list[type]:
    """Every class defined in a module under `emb_shootout.providers` that has
    an `embed` method -- walked, not listed, so a seventh provider is in scope
    the day it lands."""
    found = []
    for info in pkgutil.iter_modules(providers_pkg.__path__):
        module = importlib.import_module(f"{providers_pkg.__name__}.{info.name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module.__name__ and callable(getattr(cls, "embed", None)):
                found.append(cls)
    return sorted(found, key=lambda c: c.__name__)


PROVIDERS = _provider_classes()


def test_the_population_is_the_six_providers_and_not_an_empty_walk() -> None:
    # A walk that found nothing would make every arm below pass vacuously.
    assert [c.__name__ for c in PROVIDERS] == [
        "BGEProvider",
        "CohereProvider",
        "HashEmbedderProvider",
        "NomicProvider",
        "OpenAIProvider",
        "VoyageProvider",
    ]


@pytest.mark.parametrize("cls", PROVIDERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("bare", _BARE, ids=repr)
def test_every_provider_refuses_before_touching_self(cls: type, bare: Any) -> None:
    """`object.__new__` skips `__init__`, so the instance has no client, no
    encoder and no `batch_size`. Any `embed` that reached one of them before
    the check would raise `AttributeError`, not this `ValueError` -- which is
    what "the refusal happens before any client / encoder is touched" means,
    asserted without the optional extras installed."""
    instance = object.__new__(cls)
    with pytest.raises(ValueError, match=_SPLIT) as exc:
        instance.embed(bare)
    assert str(exc.value).startswith("texts must be an iterable of strings")


def test_embed_message_shows_the_value_and_the_working_spelling() -> None:
    with pytest.raises(ValueError, match=_SPLIT) as exc:
        HashEmbedderProvider().embed("hello world")
    assert "'hello world'" in str(exc.value)
    assert "pass ['hello world']" in str(exc.value)


@pytest.mark.parametrize(
    "texts",
    [["hello world", "x"], ("hello world", "x"), (t for t in ["hello world", "x"])],
    ids=["list", "tuple", "generator"],
)
def test_embed_collections_are_unchanged(texts: Any) -> None:
    got = HashEmbedderProvider().embed(texts)
    assert got == HashEmbedderProvider().embed(["hello world", "x"])
    assert len(got) == 2


def test_embed_the_working_spelling_returns_one_vector() -> None:
    assert len(HashEmbedderProvider().embed(["hello world"])) == 1


# --- build_corpus ---------------------------------------------------------------


@pytest.mark.parametrize("bare", _BARE, ids=repr)
def test_build_corpus_refuses_a_bare_string_at_the_call(bare: Any) -> None:
    # At the call, not at the first `next()`: `build_corpus` used to be a
    # generator function, and a check inside its body would not run until the
    # caller started iterating.
    with pytest.raises(ValueError, match=_SPLIT) as exc:
        build_corpus(bare)
    assert str(exc.value).startswith("modules must be an iterable of strings")


@pytest.mark.parametrize(
    "modules",
    [["json"], ("json",), {"json"}, (m for m in ["json"])],
    ids=["list", "tuple", "set", "generator"],
)
def test_build_corpus_collections_are_unchanged(modules: Any) -> None:
    chunks = list(build_corpus(modules))
    assert chunks
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in build_corpus(["json"])]


def test_build_corpus_none_still_means_the_default_modules() -> None:
    # Not compared walk-for-walk with `build_corpus()`: two default walks in one
    # process can differ, because importing a package's submodules adds
    # attributes the second walk then documents (the ems#115 territory). What
    # matters here is that `None` is not refused as a bare string.
    first = next(iter(build_corpus(None)))
    assert first.source == "python-stdlib"


def test_build_corpus_still_skips_a_module_that_does_not_import() -> None:
    # The by-design skip is what hid the bare-string case; it stays.
    assert list(build_corpus(["no_such_module_169"])) == []
