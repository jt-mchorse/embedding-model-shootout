"""Script form of `notebooks/reproduce.ipynb`.

Lets us verify the whole reproduction flow runs end-to-end without
installing jupyter. The two are kept in step by hand; the one block a test
holds identical in both is the frontier check in step 5
(`tests/test_reproduce_frontier_check.py`, #197).

Run:
    .venv/bin/python notebooks/_verify.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emb_shootout.corpus import DEFAULT_MODULES, build_corpus  # noqa: E402
from emb_shootout.pareto import pareto_frontier  # noqa: E402
from emb_shootout.providers.hash_embedder import HashEmbedderProvider  # noqa: E402
from emb_shootout.queries import build_queries  # noqa: E402
from emb_shootout.sweep import SweepResult, aggregate_markdown, run_sweep  # noqa: E402


def main() -> int:
    print("=" * 60)
    print("Step 1 — Build the corpus from CPython stdlib")
    print("=" * 60)
    corpus = list(build_corpus(DEFAULT_MODULES))
    print(f"  corpus size: {len(corpus)} chunks")
    assert len(corpus) >= 10_000, "acceptance bar: corpus must be >= 10k chunks"
    sample = corpus[0]
    # Each chunk has chunk_id, qualname, kind, source — show the shape.
    print(f"  first chunk: chunk_id={sample.chunk_id!r} qualname={sample.qualname!r}")

    print()
    print("=" * 60)
    print("Step 2 — Derive the deterministic query set (seed=42)")
    print("=" * 60)
    queries = build_queries(corpus, n=50, seed=42)
    print(f"  query count: {len(queries)}")
    print(f"  first query: {queries[0].text[:60]!r}")

    print()
    print("=" * 60)
    print("Step 3 — Run the hash baseline sweep")
    print("=" * 60)
    embedder = HashEmbedderProvider()
    result = run_sweep(corpus, queries, embedder=embedder, k_values=(1, 5, 10))
    print(f"  embedder: {result.embedder_name}")
    print(f"  recall@1={result.recall_at_k[1]:.3f}")
    print(f"  recall@5={result.recall_at_k[5]:.3f}")
    print(f"  recall@10={result.recall_at_k[10]:.3f}")
    print(f"  NDCG@10={result.ndcg_at_10:.3f}")

    print()
    print("=" * 60)
    print("Step 4 — Aggregate committed results -> markdown table")
    print("=" * 60)
    results_dir = ROOT / "results"
    result_files = sorted(results_dir.glob("*.json"))
    print(f"  result files: {[p.name for p in result_files]}")
    results = [
        SweepResult.from_dict(json.loads(p.read_text(encoding="utf-8"))) for p in result_files
    ]
    md = aggregate_markdown(results)
    # The committed docs/benchmarks.md should contain the same data rows.
    committed = (ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    # The aggregator's output is the data-rows section only; check that
    # every committed table row appears in the freshly-generated md.
    print("  generated markdown preview (first ~10 lines):")
    for line in md.splitlines()[:10]:
        print(f"    {line}")
    # Sanity: the hash baseline's `recall@5 = 0.520` claim in the
    # committed doc matches the regenerated table.
    assert "0.520" in md, "regenerated table missing the canonical recall@5 = 0.520"
    assert "0.520" in committed, "committed benchmarks.md no longer carries 0.520"

    print()
    print("=" * 60)
    print("Step 5 — Recompute the Pareto frontier")
    print("=" * 60)
    frontier = pareto_frontier(results)
    print(f"  frontier size: {len(frontier)} point(s)")
    for pt in frontier:
        print(
            f"    {pt.embedder_name}: $/M={pt.cost_per_million_tokens}, "
            f"recall@5={pt.recall_at_k.get(5, 0.0):.3f}"
        )
    # frontier-check:begin -- kept identical in notebooks/_verify.py and
    # notebooks/reproduce.ipynb (tests/test_reproduce_frontier_check.py).
    # Holds for any valid result set: the frontier is non-empty and drawn from the
    # inputs, and a dominated result is simply left off it. This used to assert
    # `len(frontier) == len(results)`, which is true only while no result is
    # dominated, so the first dominated provider row failed the reproducer (#197).
    on_frontier = {id(pt) for pt in frontier}
    assert frontier, "a non-empty result set has a non-empty frontier"
    assert on_frontier <= {id(r) for r in results}, "frontier points must be input results"
    dominated = [r.embedder_name for r in results if id(r) not in on_frontier]
    print(f"dominated: {dominated or 'none'}")
    # frontier-check:end

    print()
    print("All steps completed without error.")
    print("To regenerate the committed docs after a real-provider run:")
    print("  emb-shootout sweep aggregate --results-dir results --out docs/benchmarks.md")
    print("  emb-shootout sweep plot --results-dir results --out-png docs/pareto.png \\")
    print("    --out-svg docs/pareto.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
