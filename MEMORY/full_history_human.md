# Session History (human-readable)

Chronological log of work sessions. Most recent first below the divider.

---

## 2026-05-19 — Issue #11: snapshot test for docs/benchmarks.md aggregator output
**Duration:** ~20 min · **Branch:** `session/2026-05-19-1515-issue-11` · **PR:** #12

- Added `tests/test_benchmarks_md_snapshot.py` (2 tests). Loads every `results/*.json` into `SweepResult` instances exactly as the CLI's aggregate subcommand does, calls `emb_shootout.sweep.aggregate_markdown(results)`, and asserts the produced header + separator + data block appears verbatim inside `docs/benchmarks.md`. A header-signature presence guard ensures the table can't be silently dropped from the file.
- The committed `docs/benchmarks.md` numbers are already indirectly locked by `test_readme_snapshot.py` (via `results/hash.json`). This new test closes the orthogonal gap — **aggregator-format drift** — that would silently desync `docs/benchmarks.md` from the CLI's actual output if column order, decimal precision, sort key, or separator alignment ever changed.
- Failure messages name the regen command (`emb-shootout sweep aggregate --results-dir results --out docs/benchmarks.md`) and a `git diff` hint. Tamper-verified by flipping `recall@5` `0.520 → 0.999` in `docs/benchmarks.md`; the substring assertion fires with the full aggregator output printed inline so the operator can copy-paste-fix.

**Why this work, this session:** Third snapshot test in this repo's lineage (after the README snapshot and the rag-production-kit eval/rewriter snapshots), all enforcing the portfolio's "no fabricated benchmarks" rule structurally. `docs/benchmarks.md` is the operator-facing artifact that real-provider rows will land into, and its rendering needed the same protection the README already has on its prose quotes.

**Open questions / blockers:** None — PR ready for review.

**Next session:** Both this repo's documentation artifacts (README Takeaways + docs/benchmarks.md table) are now drift-locked. Continue the multi-issue loop into the next zero-issue repo (chunking-strategies-lab is next in §8).

## 2026-05-19 — Issue #4: honest narrative takeaway in README
**Duration:** ~45 min · **Branch:** `session/2026-05-19-issue-04`

- Rewrote `What this is` to drop the corpus-PR framing (issue #1 is closed and so are #2, #3, #5) and replace it with a present-tense four-bullet picture of what the repo actually ships today: corpus, sweep harness, Pareto frontier, reproducer.
- Added a `Takeaways (so far)` section grounded only in `results/hash.json`. Quotes recall@1/5/10 = 0.320/0.520/0.620, NDCG@10 = 0.449, corpus embed ~429 ms, query p95 = 0.017 ms, 12,010 chunks × 50 seeded queries, $0/M tokens. Refuses to make any cross-provider winner claim until the operator commits real-provider JSONs.
- Wired `tests/test_readme_snapshot.py` (7 tests) to lock the Takeaways quotes to `results/hash.json` so the prose can't drift from the measurement. Same hygiene pattern as the seven sister PRs landed across the portfolio on 2026-05-18.

**Why this work, this session:** Issue #4 was the last open priority:med across the portfolio after the 2026-05-18 PR review pass merged four other repos' README cleanups; the build-sequence tie-break picked embedding-model-shootout (#5) over chunking-strategies-lab (#6).

**Open questions / blockers:** None. The real-provider results table will arrive when the operator commits a `results/<provider>.json`; the harness is wired, the queries are deterministic, the snapshot test will fail until the README catches up.

**Next session:** Either chunking-strategies-lab #11 (the snapshot test currently has a CI failure on PR #12 because `results/` is gitignored) or a demo-capture issue if the autonomous run can produce a usable asset.

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

## 2026-05-18 — Issue #5: Reproduction notebook
**Duration:** ~30 min · **Branch:** `session/2026-05-18-issue-05` · **PR:** #9

- Shipped `notebooks/reproduce.ipynb`: top-to-bottom regeneration of every artifact currently committed in the repo (corpus → query set → hash baseline sweep → markdown table → Pareto frontier) with cell-level commentary linking each step to its D-NNN. Output cells committed empty so a clean re-run produces a meaningful diff.
- Shipped `notebooks/_verify.py` (executable twin) and `notebooks/_build_notebook.py` (regenerates the `.ipynb` from a static Python cell spec so the notebook is diffable in code review, not as hand-edited JSON).
- Five shape tests in `tests/test_reproduce_notebook.py`: valid Jupyter JSON, canonical imports present, ≥5 code cells, no cached outputs, `_verify.py` runs end-to-end. Suite total 69 (was 64).
- README gains a Reproduce section under Quickstart pointing at both forms.
- No new D-NNN — this is pure reproducibility infrastructure, no tradeoff worth recording.

**Why this work, this session:** Low-priority backlog item, but ~30 minutes of contained work and a real value-add (the notebook was advertised as future work in the README). Closes one of the three remaining "open" items in the embedding-model-shootout repo.

**Open questions / blockers:** PR explicitly notes that direct Jupyter execution is *not* exercised in CI — the script form is the honest minimal contract; adding nbconvert+jupyter to dev deps to execute the notebook in CI would trade the dep-free posture for one test.

**Next session:** With 9 issues fully closed tonight (plus 1 skipped + 1 partial), wrapping is the right call. The remaining backlog is the data-blocked #4 here and the larger 70-min mcp-server-cookbook #5 Python parity.

## 2026-05-18 — Issue #5 (continuation): Unblock PR #9 lint
**Duration:** ~10 min · **Branch:** `session/2026-05-18-issue-05` · **PR:** [#9](https://github.com/jt-mchorse/embedding-model-shootout/pull/9) (ready, awaiting CI re-run)

- Added a `[tool.ruff.lint.per-file-ignores]` block to `pyproject.toml` that ignores E402 in `*.ipynb`. The five lint failures on PR #9 were all from `notebooks/reproduce.ipynb` cell 3 where the cell does `sys.path.insert(0, str(ROOT))` before the project imports — a standard notebook idiom for "make this runnable from a fresh clone", not a real smell. Production `.py` code still gets the E402 check.
- All other CI jobs were already green (tests 3.11 + 3.12 + memory-check); this was the only blocker.

**Why this work, this session:** Phase A auto-review left this PR commented with a clear lint blocker; rather than starting fresh work, the higher-leverage move was the small fix that gets a working PR over the merge line.

**Open questions / blockers:** None — pending CI re-run on the workflow.

**Next session:** Continue the multi-issue loop; next target is mcp-server-cookbook PRs #13 and #14 (both need rebase against main after this session's #12 merge).

## 2026-05-20 — Issue #13: lock emb_shootout public surface
**Duration:** ~25 min · **Branch:** `session/2026-05-20-0317-issue-13`

- Added `tests/test_public_surface.py` (4 standalone + 4 parametrized submodule anchors = 8 test items) and `__version__ = "0.0.1"` on the package. Locks five axes: `__version__` semver, every `__all__` entry bound and non-None, `__all__` ↔ AST-parsed `from .X import` bidirectional agreement (filter `level >= 1`), README's quoted `emb_shootout.pareto.pareto_frontier` dotted path resolves to a callable, and one anchor per *re-exported* submodule. `cli`/`pareto`/`plot` are intentionally excluded from the anchor list — they're dotted-path-only by design, and re-exporting them at the top level would expand the public surface without a deliberate decision.
- Tamper-verified four of five axes locally: bad version → axis (1); drop `"Chunk"` from `__all__` → axis (3); rename `pareto.py` → `pareto_renamed.py` → axis (4) fires with `ModuleNotFoundError`; alias-rename `HashEmbedderProvider as HashEmbedderProviderV2` → axis (5)[providers].
- Full suite 86/86 (was 78; +8 new).

**Why this work, this session:** Fifth strike of the portfolio-wide public-surface hygiene pattern. Adapts cleanly to this repo's split between re-exported and dotted-path-only submodules — the README quotes one dotted path by name (`emb_shootout.pareto.pareto_frontier`), so locking it is the natural fourth axis here, replacing the "README quickstart imports" axis used in the four prior repos (this README's quickstart uses CLI commands instead of Python imports).

**Open questions / blockers:** None — PR ready for review.

**Next session:** Continue the loop into `chunking-strategies-lab`, then `python-async-llm-pipelines`, then the Python example in `mcp-server-cookbook`.

## 2026-05-21 — Issue #15: 60-second demo capture script
**Duration:** ~30 min · **Branch:** `session/2026-05-21-1910-issue-15` · **PR:** #16

- Added `scripts/capture_demo.sh` driving the repo's three highest-leverage surfaces — `emb-shootout corpus build --module json` → `sweep run --provider hash` → `sweep aggregate --out <tmp>.md` then `cat` the rendered table — end-to-end on a fresh clone with no API key and no network. `CAPTURE_PACE_SECONDS` (default 2 for recording, 0 for CI), `CAPTURE_DEMO_MODULE`, and `CAPTURE_DEMO_QUERIES` env knobs let JT vary recordings across takes without editing the script. Per-run tempdir trapped on EXIT/INT/TERM so concurrent runs and the smoke test don't collide.
- Added `tests/test_capture_demo_smoke.py` (4 tests) that runs the script with `PACE=0` in CI and asserts each surface's distinctive output line (corpus JSON summary keys, `recall@5=` line, aggregator markdown header), plus the executable bit + existence pin. The header assertion is belt-and-braces with `test_benchmarks_md_snapshot.py` — if the aggregator format drifts, both tests fire.
- README "Demo" section replaces the `*60-second demo pending — depends on issue #2.*` placeholder with one paragraph framing the three surfaces plus the recording-vs-headline distinction (single-module corpus is for tempo; the floor numbers stay locked in `docs/benchmarks.md` and the Takeaways section). 90/90 tests pass, ruff clean.

**Why this work, this session:** Sixth repo to land the `scripts/capture_demo.sh` pattern after the four sister-repo PRs landed today (llm-eval-harness, prompt-regression-suite, llm-cost-optimizer, rag-production-kit) — closing the README "Demo" gap that's the last unchecked item on the six-item v0.1 quality bar for this repo. The repo had zero open issues at session start, so the issue itself was filed against the §2 quality-bar gap before any code.

**Open questions / blockers:** None. The capture is hermetic so it can be re-recorded any time without coordination.

**Next session:** Pick the next stale repo per Phase A selection rules.
