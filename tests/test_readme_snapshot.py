"""README snapshot: lock the Takeaways numbers to results/hash.json.

Sister to the portfolio-wide snapshot pattern landed on 2026-05-18.
The Takeaways section quotes specific hash-baseline numbers. Those
numbers must come from the committed `results/hash.json`. Without this
test, a tweak to the hash provider or the sweep harness can quietly
desync the README's prose from reality.

The test also checks:
- Files referenced by `[text](path)` markdown links exist on disk.
- The Takeaways prose still names every D-NNN that the methodology
  bullet promises is in flight, so demoting a decision doesn't
  accidentally orphan the prose.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
HASH_RESULTS = REPO_ROOT / "results" / "hash.json"


@pytest.fixture(scope="module")
def readme_text() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def hash_results() -> dict:
    return json.loads(HASH_RESULTS.read_text(encoding="utf-8"))


def _format_recall(v: float) -> str:
    # Takeaways uses three-decimal format for recall (e.g., 0.520).
    return f"{v:.3f}"


def _format_ndcg(v: float) -> str:
    # NDCG quoted to three decimals (0.449).
    return f"{v:.3f}"


def test_recall_at_k_quotes_match_results_hash(readme_text: str, hash_results: dict) -> None:
    # Soft-wrap the README on whitespace before matching so a line break
    # between `**recall@5 =` and `0.520**` doesn't fool the regex.
    flat = re.sub(r"\s+", " ", readme_text)
    recalls = hash_results["recall_at_k"]
    expected = {
        "recall@1": _format_recall(recalls["1"]),
        "recall@5": _format_recall(recalls["5"]),
        "recall@10": _format_recall(recalls["10"]),
    }
    for label, value in expected.items():
        pattern = rf"\*\*{label} = {re.escape(value)}\*\*"
        assert re.search(pattern, flat), (
            f"README is missing or mis-quoting {label}; "
            f"expected `**{label} = {value}**`. "
            f"Live value in results/hash.json: {value}. "
            f"Regenerate quotes from the committed JSON before re-running."
        )


def test_ndcg_at_10_quote_matches_results_hash(readme_text: str, hash_results: dict) -> None:
    flat = re.sub(r"\s+", " ", readme_text)
    value = _format_ndcg(hash_results["ndcg_at_10"])
    pattern = rf"\*\*NDCG@10 = {re.escape(value)}\*\*"
    assert re.search(pattern, flat), (
        f"README is missing or mis-quoting NDCG@10; expected `**NDCG@10 = {value}**`. "
        f"Live value in results/hash.json: {value}."
    )


def test_corpus_and_query_counts_match_results_hash(readme_text: str, hash_results: dict) -> None:
    # n_corpus is referenced as "12,010" in one place and "12010" in
    # another (the original quickstart line). Only the comma-grouped
    # form is locked here; the bare integer is corpus shape, not a
    # measurement.
    n_corpus_pretty = f"{hash_results['n_corpus']:,}"
    assert n_corpus_pretty in readme_text, (
        f"README is missing the locked corpus count `{n_corpus_pretty}`. "
        f"Live n_corpus from results/hash.json: {hash_results['n_corpus']}."
    )
    n_queries = hash_results["n_queries"]
    assert f"{n_queries} seeded queries" in readme_text, (
        f"README is missing `{n_queries} seeded queries`. "
        f"Live n_queries from results/hash.json: {n_queries}."
    )


def test_query_p95_latency_matches_results_hash(readme_text: str, hash_results: dict) -> None:
    # README quotes p95 as `0.017 ms`; tolerance is one digit because
    # the raw number can wiggle slightly between machine runs but the
    # *committed* JSON is the source of truth, not the most recent run.
    raw = hash_results["embed_latency_ms"]["query_p95"]
    quoted = f"{raw:.3f}"
    assert f"per-query p95 of **{quoted} ms**" in readme_text, (
        f"README must quote per-query p95 = `{quoted} ms` (three decimals from results/hash.json: {raw}). "
        f"If the hash baseline was re-run, regenerate the README quote."
    )


def test_corpus_embed_total_matches_results_hash_rounded(
    readme_text: str, hash_results: dict
) -> None:
    # README rounds corpus_total ms to the nearest integer (~429 ms).
    rounded = int(round(hash_results["embed_latency_ms"]["corpus_total"]))
    assert f"**~{rounded} ms**" in readme_text, (
        f"README must quote corpus embed time = `~{rounded} ms`. "
        f"Live corpus_total ms from results/hash.json: "
        f"{hash_results['embed_latency_ms']['corpus_total']:.2f}."
    )


def test_referenced_files_exist(readme_text: str) -> None:
    # Pull every relative-path target out of the README and assert it
    # resolves on disk. Absolute http(s) URLs (CI badge, shields.io)
    # are external and not checked here.
    pattern = re.compile(r"\(([^)\s]+\.(?:md|png|svg|ipynb|py))\)")
    refs = {r for r in pattern.findall(readme_text) if not r.startswith(("http://", "https://"))}
    # Footer reference-style ([D-002]: MEMORY/core_decisions_human.md).
    footer = re.findall(r"^\[[^\]]+\]: ([^\s:]+\.(?:md|png|svg))$", readme_text, re.MULTILINE)
    refs.update(r for r in footer if not r.startswith(("http://", "https://")))
    missing = sorted(r for r in refs if not (REPO_ROOT / r).exists())
    assert not missing, (
        f"README references files that don't exist: {missing}. "
        f"Either fix the link or commit the file."
    )


def test_takeaways_section_names_every_methodology_decision(readme_text: str) -> None:
    # The methodology bullet in Takeaways promises five D-NNN tags. If
    # any of them is removed or renamed without updating the prose, the
    # narrative orphans a decision.
    expected_tags = {"D-002", "D-003", "D-005", "D-006", "D-007"}
    # Slice to just the Takeaways section so we don't false-positive on
    # other parts of the README that name D-NNN.
    start = readme_text.index("## Takeaways (so far)")
    end = readme_text.index("##", start + 1)
    takeaways = readme_text[start:end]
    missing = sorted(tag for tag in expected_tags if tag not in takeaways)
    assert not missing, (
        f"Takeaways section must reference {sorted(expected_tags)} but is missing {missing}. "
        f"Either name the decision in the prose or update this test if the "
        f"methodology contract changed."
    )
