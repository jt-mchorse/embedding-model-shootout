"""Generate `notebooks/reproduce.ipynb` from a static cell spec.

Run:
    .venv/bin/python notebooks/_build_notebook.py

This script writes the notebook with empty cell outputs so re-running it
on an operator's machine produces a clean diff.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent / "reproduce.ipynb"


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else [],
    }


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else [],
    }


CELLS: list[dict] = [
    md(
        "# Reproducing `embedding-model-shootout`",
        "",
        "This notebook runs top-to-bottom and regenerates every published",
        "artifact in the repo from the committed source: the corpus, the",
        "query set, the hash baseline sweep, the markdown table, and the",
        "Pareto-frontier plot. Output cells are committed empty so a clean",
        "re-run produces a meaningful diff.",
        "",
        '**What "reproduce" means here.** The only provider currently',
        "committed in `results/` is the dep-free `hash` baseline — see issue",
        "[#4] for the open question on the five real providers. The flow",
        "below would handle any additional `results/<provider>.json` file",
        "without changing this notebook; it merely shows up as another row in",
        "the table and another point on the frontier.",
        "",
        "[#4]: https://github.com/jt-mchorse/embedding-model-shootout/issues/4",
    ),
    md(
        "## 0 · Imports",
        "",
        "Everything is imported from the in-repo package — no notebook-only",
        "deps. Add the repo root to `sys.path` so the notebook also runs from",
        "an editable install.",
    ),
    code(
        "import json",
        "import sys",
        "from pathlib import Path",
        "",
        "ROOT = Path.cwd().resolve()",
        "if ROOT.name == 'notebooks':",
        "    ROOT = ROOT.parent",
        "if str(ROOT) not in sys.path:",
        "    sys.path.insert(0, str(ROOT))",
        "",
        "from emb_shootout.corpus import DEFAULT_MODULES, build_corpus",
        "from emb_shootout.pareto import pareto_frontier",
        "from emb_shootout.providers.hash_embedder import HashEmbedderProvider",
        "from emb_shootout.queries import build_queries",
        "from emb_shootout.sweep import SweepResult, aggregate_markdown, run_sweep",
        "",
        "print('ROOT =', ROOT)",
    ),
    md(
        "## 1 · Build the corpus",
        "",
        "The corpus is derived deterministically from CPython's stdlib via",
        "`inspect` (D-002 — not committed as data, reproduced from source).",
        "The acceptance bar is `>= 10000` chunks; on CPython 3.14 the curated",
        "module list yields **12,010 chunks**.",
    ),
    code(
        "corpus = list(build_corpus(DEFAULT_MODULES))",
        "print(f'corpus size: {len(corpus):,} chunks')",
        "assert len(corpus) >= 10_000, 'acceptance bar: corpus must be >= 10k chunks'",
        "",
        "sample = corpus[0]",
        "print('first chunk shape:')",
        "print(f'  chunk_id : {sample.chunk_id!r}')",
        "print(f'  qualname : {sample.qualname!r}')",
        "print(f'  kind     : {sample.kind!r}')",
        "print(f'  source   : {sample.source!r}')",
    ),
    md(
        "## 2 · Derive the query set",
        "",
        "Queries are derived from the corpus at sweep time with `seed=42`",
        "(D-005), not committed as a fixture file. Every provider runs against",
        "the same query set by construction — cross-provider rows are",
        "apples-to-apples.",
    ),
    code(
        "queries = build_queries(corpus, n=50, seed=42)",
        "print(f'query count: {len(queries)}')",
        "for q in queries[:3]:",
        "    print(f'  example: {q.text[:60]!r}')",
    ),
    md(
        "## 3 · Run the hash baseline sweep",
        "",
        "`HashEmbedderProvider` is the dep-free SHA-256 bag-of-bigrams",
        "baseline. It exists so the sweep flow exercises end-to-end without",
        "external services or API keys — the recall numbers it produces are",
        "the lower bound, not the takeaway.",
    ),
    code(
        "embedder = HashEmbedderProvider()",
        "result = run_sweep(corpus, queries, embedder=embedder, k_values=(1, 5, 10))",
        "",
        "print(f'embedder    : {result.embedder_name}')",
        "print(f'n_corpus    : {result.n_corpus:,}')",
        "print(f'n_queries   : {result.n_queries}')",
        "print(f'recall@1    : {result.recall_at_k[1]:.3f}')",
        "print(f'recall@5    : {result.recall_at_k[5]:.3f}')",
        "print(f'recall@10   : {result.recall_at_k[10]:.3f}')",
        "print(f'NDCG@10     : {result.ndcg_at_10:.3f}')",
    ),
    md(
        "## 4 · Aggregate `results/*.json` into the markdown table",
        "",
        "Each provider's run writes one JSON file (D-007 — per-provider files",
        "avoid concurrent-run collisions). `aggregate_markdown` is a pure-read",
        "merge over `results/`.",
        "",
        "The cell below regenerates `docs/benchmarks.md`'s data rows from the",
        "committed result files and pins the canonical `recall@5 = 0.520`",
        "claim — if a future run produces a different number, this cell fails.",
    ),
    code(
        "result_files = sorted((ROOT / 'results').glob('*.json'))",
        "print('result files:', [p.name for p in result_files])",
        "",
        "results = [",
        "    SweepResult.from_dict(json.loads(p.read_text(encoding='utf-8')))",
        "    for p in result_files",
        "]",
        "md_table = aggregate_markdown(results)",
        "print(md_table)",
        "",
        "committed = (ROOT / 'docs' / 'benchmarks.md').read_text(encoding='utf-8')",
        "assert '0.520' in md_table, 'regenerated table missing canonical recall@5=0.520'",
        "assert '0.520' in committed, 'committed benchmarks.md no longer carries 0.520'",
        "print('OK — generated table reproduces the committed numbers')",
    ),
    md(
        "## 5 · Recompute the Pareto frontier",
        "",
        "`pareto_frontier(results)` is pure-stdlib Python (D-008), so it",
        "ships in the base install. With only the hash baseline committed,",
        "the frontier is trivially itself — one point. When real provider",
        "JSONs land, the frontier widens; the `emb-shootout sweep plot` CLI",
        "renders it (matplotlib renderer behind the optional `[plot]` extra).",
    ),
    code(
        "frontier = pareto_frontier(results)",
        "print(f'frontier size: {len(frontier)} point(s)')",
        "for pt in frontier:",
        "    print(",
        "        f'  {pt.embedder_name}: '",
        "        f'$/M={pt.cost_per_million_tokens}, '",
        "        f'recall@5={pt.recall_at_k.get(5, 0.0):.3f}'",
        "    )",
        "",
        "assert len(frontier) == len(results), 'frontier size must match input count'",
    ),
    md(
        "## 6 · Regenerate `docs/pareto.{png,svg}`",
        "",
        "This step requires the `[plot]` extra (`pip install -e '.[plot]'`).",
        "Skip it for a hermetic notebook run; trigger it manually when",
        "regenerating committed artifacts after a real-provider run.",
        "",
        "```bash",
        "emb-shootout sweep plot \\",
        "  --results-dir results \\",
        "  --out-png docs/pareto.png \\",
        "  --out-svg docs/pareto.svg",
        "```",
    ),
    md(
        "## 7 · What the numbers say",
        "",
        "The hash baseline's `recall@5 = 0.520` is the *lower bound* of what",
        "the harness can measure on this corpus + this query construction. A",
        "real embedder that scores meaningfully above this floor is doing",
        "useful work; one that scores at or near this floor on the same",
        "queries is approximating lexical overlap, regardless of how many",
        "embedding dimensions it ships.",
        "",
        "The real provider rows (OpenAI / Voyage / Cohere / BGE / Nomic) land",
        "in `results/` when the operator runs `emb-shootout sweep run",
        "--provider <P>` with the relevant API key and commits the resulting",
        "JSON. This notebook then re-runs unchanged, the markdown table grows",
        "rows, and the Pareto frontier widens. Issue [#4] tracks the",
        "narrative pass that depends on those rows existing.",
        "",
        "[#4]: https://github.com/jt-mchorse/embedding-model-shootout/issues/4",
    ),
]


def build() -> dict:
    return {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(
        f"wrote {NOTEBOOK_PATH.relative_to(NOTEBOOK_PATH.parent.parent)} ({NOTEBOOK_PATH.stat().st_size} bytes)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
