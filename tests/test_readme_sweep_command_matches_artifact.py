"""The README's documented sweep command must be the one that made the artifact (#114).

The quickstart documented

    emb-shootout sweep run --provider hash \\
      --corpus data/corpus.jsonl --queries 200 --output results/hash.json

while the committed ``results/hash.json`` carries ``n_queries: 50``. Following
the README gave recall@5 0.560; ``docs/benchmarks.md`` and the README prose
both say 0.520. The number was honest and re-derivable — with ``--queries 50``
it reproduces to all 16 digits of ``ndcg_at_10`` — but no *documented* command
produced it.

Three snapshot tests already cover this area and none could see it, because
together they form a closed loop that never touches the generator:

- ``test_readme_snapshot.py`` locks ``results/hash.json``'s cells against the
  README prose quoting them.
- ``test_benchmarks_md_snapshot.py`` locks the aggregator's *rendering* of
  ``results/*.json`` into ``docs/benchmarks.md``; its own docstring says the
  numbers themselves are "indirectly locked by test_readme_snapshot.py".

README prose ↔ artifact ↔ rendered table, all mutually consistent, with the
command that produces the artifact outside the loop entirely.

This closes it from the one direction that is **interpreter-independent**.
A regenerate-and-compare freshness test (the shape
``chunking-strategies-lab/tests/test_canonical_fixture_freshness.py`` uses)
cannot work here: the corpus is built from the local stdlib's docstrings
(D-002), so it differs per Python version — 11 108 chunks on 3.11 versus
12 010 on 3.14, moving recall@5 from 0.580 to 0.520. CI runs 3.11 and 3.12,
and the committed artifact is from a 3.14-era stdlib, so that test would fail
in CI every run; gating it on the version would make it *always skip*, which
is its own failure mode. Comparing the documented flags against the artifact's
recorded metadata needs no rebuild and holds on every interpreter.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
ARTIFACT = REPO_ROOT / "results" / "hash.json"


def _documented_hash_sweep() -> str:
    """The README's `sweep run --provider hash` invocation, line continuations joined."""
    text = README.read_text(encoding="utf-8")
    joined = text.replace("\\\n", " ")
    for line in joined.splitlines():
        if "sweep run" in line and "--provider hash" in line:
            return line
    raise AssertionError(
        "README no longer documents an `emb-shootout sweep run --provider hash` "
        "command; if the quickstart was restructured, update this lock."
    )


def _flag(command: str, name: str) -> str | None:
    match = re.search(rf"{re.escape(name)}[=\s]+(\S+)", command)
    return match.group(1) if match else None


def test_documented_queries_matches_the_committed_artifact() -> None:
    """`--queries` in the README must equal the artifact's `n_queries`.

    This is the exact drift #114 was: 200 documented, 50 in the artifact.
    """
    command = _documented_hash_sweep()
    documented = _flag(command, "--queries")
    assert documented is not None, f"README's hash sweep names no --queries: {command!r}"

    artifact_n = json.loads(ARTIFACT.read_text(encoding="utf-8"))["n_queries"]
    assert int(documented) == artifact_n, (
        f"README documents --queries {documented}, but results/hash.json was "
        f"produced with n_queries={artifact_n}. A reader following the quickstart "
        "gets different numbers than the committed table. Change whichever is "
        "wrong — but regenerating the artifact rewrites a published benchmark, "
        "so prefer fixing the command."
    )


def test_documented_command_spells_out_the_seed() -> None:
    """The query set is derived from the corpus at sweep time (D-005).

    The seed is therefore part of the result's provenance, not an incidental
    default, so the documented invocation should be self-describing rather
    than relying on argparse's default staying 42.
    """
    command = _documented_hash_sweep()
    assert _flag(command, "--seed") is not None, (
        "README's documented sweep omits --seed; per D-005 the query set is "
        "derived deterministically from the corpus using it, so it belongs in "
        "any command presented as reproducing the committed artifact"
    )


def test_documented_output_path_is_the_committed_artifact() -> None:
    """Anti-vacuous: the two assertions above are meaningless if the documented
    command writes somewhere other than the file they compare against."""
    command = _documented_hash_sweep()
    out = _flag(command, "--output") or _flag(command, "--out")
    assert out is not None, f"README's hash sweep names no output path: {command!r}"
    assert Path(out) == ARTIFACT.relative_to(REPO_ROOT), (
        f"README's hash sweep writes {out}, but this lock compares against "
        f"{ARTIFACT.relative_to(REPO_ROOT)}"
    )
