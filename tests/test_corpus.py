"""Corpus loader tests. No network, no auth, runs in <5s on CI."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from emb_shootout import DEFAULT_MODULES, Chunk, build_corpus, write_jsonl

# A small subset of the default list, large enough to verify shape but
# fast enough for unit-test latency. Each of these has many documented
# members in CPython.
SMALL_MODULES = ("os.path", "json", "itertools", "functools", "math", "re")


def test_default_modules_is_curated_and_unique():
    assert len(DEFAULT_MODULES) > 50
    assert len(set(DEFAULT_MODULES)) == len(DEFAULT_MODULES)


def test_build_corpus_yields_chunks_with_expected_shape():
    chunks = list(build_corpus(SMALL_MODULES))
    assert chunks, "small-module corpus must yield at least one chunk"
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.chunk_id == c.qualname
        assert c.module in SMALL_MODULES
        assert c.kind in ("module", "function", "class", "method")
        assert c.source == "python-stdlib"
        assert c.text  # docstring or signature+docstring


def test_chunk_ids_are_unique_in_one_build():
    chunks = list(build_corpus(SMALL_MODULES))
    ids = [c.chunk_id for c in chunks]
    duplicates = [k for k, v in Counter(ids).items() if v > 1]
    assert not duplicates, f"duplicate chunk_ids: {duplicates}"


def test_build_corpus_is_deterministic_for_same_input():
    one = [c.chunk_id for c in build_corpus(SMALL_MODULES)]
    two = [c.chunk_id for c in build_corpus(SMALL_MODULES)]
    assert one == two


def test_build_corpus_skips_unimportable_modules_silently():
    # Real module first, then a guaranteed-nonexistent one.
    chunks = list(build_corpus(["json", "this_module_does_not_exist_xyz_12345"]))
    assert chunks
    assert all(c.module == "json" for c in chunks)


def test_class_methods_are_included_as_separate_chunks():
    chunks = list(build_corpus(["json"]))
    qualnames = {c.qualname for c in chunks}
    # JSONEncoder has documented methods; the encoder itself plus its
    # `encode` method must both appear.
    assert "json.JSONEncoder" in qualnames
    assert "json.JSONEncoder.encode" in qualnames


def test_module_chunk_emitted_when_module_has_docstring():
    chunks = list(build_corpus(["json"]))
    by_id = {c.chunk_id: c for c in chunks}
    assert "json" in by_id
    assert by_id["json"].kind == "module"


def test_write_jsonl_round_trip(tmp_path: Path):
    chunks = list(build_corpus(["json", "math"]))
    out = tmp_path / "corpus.jsonl"
    count = write_jsonl(chunks, out)
    assert count == len(chunks)
    raw = out.read_text(encoding="utf-8").splitlines()
    assert len(raw) == count
    # Every line must be parseable JSON with the right fields.
    for line in raw:
        rec = json.loads(line)
        assert set(rec.keys()) == {"chunk_id", "text", "module", "qualname", "kind", "source"}


def test_write_jsonl_creates_parent_dirs(tmp_path: Path):
    out = tmp_path / "deep" / "nested" / "corpus.jsonl"
    count = write_jsonl(build_corpus(["math"]), out)
    assert out.exists()
    assert count > 0


def test_full_default_modules_clears_10k_chunks():
    """Acceptance criterion: corpus size ≥ 10k chunks for stable metrics.

    This test is the gate that fails if a future stdlib change shrinks
    the corpus below the threshold; if it does, either expand
    DEFAULT_MODULES or revisit the corpus choice. Either way, don't
    silently lose the contract.
    """
    chunks = list(build_corpus())
    assert len(chunks) >= 10_000, (
        f"corpus has {len(chunks)} chunks; expected ≥10k from the curated module list"
    )


@pytest.mark.parametrize("module_name", ["json", "math", "re"])
def test_each_module_contributes_at_least_one_chunk(module_name: str):
    chunks = list(build_corpus([module_name]))
    assert chunks, f"{module_name} produced no chunks"
