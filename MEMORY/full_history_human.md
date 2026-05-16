# Session History (human-readable)

Chronological log of work sessions. Most recent first below the divider.

---

## 2026-05-14 — Issue #1: pick corpus + ship reproducible loader
**Duration:** ~50 min · **Branch:** `session/2026-05-14-1430-issue-01`

- Picked the corpus: **CPython standard-library docstrings** (PSF
  License, zero network/auth, semantically dense, stable per Python
  version). One chunk per documented stdlib member.
- Shipped `emb_shootout.corpus` with a curated 142-module list that
  produces **12,010 chunks** on CPython 3.14 (cleared the ≥10k
  acceptance bar). Loader treats `__all__` as the authoritative
  re-export list so `json.JSONEncoder` and friends emit cleanly even
  though their `__module__` is `json.encoder`.
- `emb-shootout corpus build` CLI; deterministic JSONL output.
- `docs/corpus.md` documents the choice, the license, the provenance
  fields, and the exact reproduction command.
- 15 tests, 100% coverage on `emb_shootout/`. Real CI (ruff + pytest
  matrix py3.11/3.12).
- Two decisions recorded: corpus is reproducible from source not
  committed as data (D-002), and the chunk shape is one stdlib member
  = one chunk (D-003).

**Why this work, this session:** Issue #1 locks the corpus and chunk
shape that #2's model sweep will measure against. The chunk shape
(`chunk_id`, `text`, `module`, `qualname`, `kind`, `source`) is now a
hard contract.

**Open questions / blockers:** None. Model sweep methodology will be
locked when #2 ships.

**Next session:** Issue #2 (benchmark across ≥5 embedding models)
sits cleanly on top of `data/corpus.jsonl`.

## 2026-05-15 — Issue #2: Embedding model sweep with 6 providers
**Duration:** ~80 min · **Branch:** `session/2026-05-15-2032-issue-02`

- Shipped `emb_shootout/sweep.py`: `Embedder` Protocol (D-004), `CorpusChunk`/`Query`/`SweepResult` dataclasses, `cosine` + `ndcg_at_k` + `percentile` math (textbook-tested), `retrieve_top_k` cosine retrieval, `run_sweep()` end-to-end (embed → retrieve → score), `aggregate_markdown()` cross-provider table renderer.
- Shipped `emb_shootout/queries.py`: deterministic verbatim-snippet query construction from the corpus (D-005). Seed pins reproducibility; corpus + queries always in sync.
- Shipped six provider modules under `emb_shootout/providers/`: `HashEmbedderProvider` (dep-free, hermetic CI), plus `OpenAIProvider`, `VoyageProvider`, `CohereProvider`, `BGEProvider`, `NomicProvider` — each lazy-imports its SDK behind a respective optional extra (`[openai]`, `[voyage]`, `[cohere]`, `[sbert]`).
- Extended CLI: `emb-shootout sweep run --provider X` writes per-provider JSON; `emb-shootout sweep aggregate` merges them into `docs/benchmarks.md` (D-007).
- Verified end-to-end: `emb-shootout sweep run --provider hash --queries 50` against the 12,010-chunk corpus yields `recall@5=0.52 NDCG@10=0.45` — committed as `results/hash.json` baseline so the table format is visible. Real-provider rows land when the operator runs them.
- 23 new hermetic tests + 15 from #1 = 38/38 passing. All three #2 acceptance bullets met explicitly.
- Backfilled README "Sweep harness (#2 · this PR)" section + rewrote `docs/benchmarks.md` with the hash baseline + reproduction commands.

**Why this work, this session:** The model sweep is the central deliverable of this repo — without it, the corpus from #1 is preparation work. Shipping all six providers behind one Protocol means the operator's "run the sweep" workflow is one CLI invocation per provider, results compose via the aggregator, and the comparison numbers are apples-to-apples by construction (D-005's seeded queries).

**Open questions / blockers:** The five real-provider rows in `docs/benchmarks.md` are pending operator API runs (each provider needs its own key + budget). The harness is ready; the operator decides when to spend.

**Next session:** None at the highest priority for this repo. The natural next work is the Pareto-frontier plot mentioned in §2 — that's a separate issue to file once the operator has at least two real-provider rows committed.

## 2026-05-16 — Issue #3: Pareto frontier plot (cost vs. recall@5)
**Duration:** ~60 min · **Branch:** `session/2026-05-16-2312-issue-3`

- Shipped `emb_shootout/pareto.py`: pure-Python `pareto_frontier(results)` selecting non-dominated points on (cost_per_million_tokens, recall@5), with explicit tie semantics — identical points stay on the frontier (dropping ties silently would hide co-located providers from the plot), and a point that is no-worse on both axes plus strictly-better on at least one dominates. Dep-free, so it tests in the standard CI matrix without pulling in matplotlib.
- Shipped `emb_shootout/plot.py`: matplotlib renderer (lazy-imported, behind a new `[plot]` extra). Non-frontier points plot in muted grey, frontier in red with black outline; the connecting dashed polyline draws only when ≥2 distinct frontier coordinates exist. The figure title is auto-chosen — single point, all-co-located, or the canonical "Cost vs recall@5 — Pareto frontier highlighted in red" — so a thin dataset doesn't pretend to be more than it is. Axis padding clamps recall@5 to ≤1.0.
- Extended CLI: `emb-shootout sweep plot --results-dir results --out-png docs/pareto.png --out-svg docs/pareto.svg` (lazy-imports the renderer so the CLI loads without the `[plot]` extra installed). Emits a JSON summary with the frontier members for scriptable inspection.
- Generated the real plot from the committed `results/hash.json`. Today it's a single hash-baseline point at (cost=$0.000, recall@5=0.520) — that's exactly what's in the data, and the figure title says so. Committed `docs/pareto.png` (47KB) and `docs/pareto.svg` (37KB). When the operator commits real-provider JSONs, this same image regenerates with multiple points and a visible frontier curve.
- 26 new tests: 15 frontier-math unit tests (empty, single, dominated, two-non-dominated, middle-dominated, identical-points, tie-on-cost, tie-on-recall, sort order, missing recall@5, real hash baseline, parametrized size bound) and 11 plot integration tests (PNG/SVG writes, parent-dir creation, single-output variants, no-output rejection, empty-results rejection, dominated-point exclusion, CLI subcommand wiring across success + four error paths). Plot tests use `pytest.importorskip("matplotlib")` so the standard CI matrix without extras still passes. Full suite 64/64. Ruff clean.
- README "Pareto frontier (cost vs. recall@5)" subsection added under Benchmarks/Results with the embedded image, the honest single-point caveat, and the regeneration command. D-008 records the axis choice and the dep-free-frontier + matplotlib-behind-extra split (parallel to D-004's provider-extras pattern).

**Why this work, this session:** Issue #3 was the lowest-numbered unblocked priority:med issue in this repo, and it sits naturally on top of #2's `SweepResult` shape — every field the plot needs (cost, recall@5, embedder_name) was already locked. Closing #3 with one new decision (D-008) ships the third of three acceptance criteria with real data, no fabricated points, and keeps the dep-free posture of the core package intact.

**Open questions / blockers:** Plot has one point until real-provider runs land. The renderer is built and tested for the multi-point case; no further code change is needed when the operator commits e.g. `results/openai.json`.

**Next session:** Loop to a different portfolio repo per the multi-issue session prompt. Remaining work in this repo is documentation-shaped (#4 narrative README, #5 reproducer notebook) and `priority:med`/`priority:low`, so a different repo's `priority:high` queue is the better next target.
