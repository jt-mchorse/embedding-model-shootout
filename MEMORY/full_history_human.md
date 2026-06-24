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

## 2026-05-22 — Honesty fix: hash baseline is word-bigrams, not character-bigrams (#17)

**Duration:** ~25 min. **Issue:** [#17](https://github.com/jt-mchorse/embedding-model-shootout/issues/17). **PR:** TBD.

The README's `## Takeaways (so far)` section described the dep-free hash baseline three times as character-bigrams ("a SHA-256 bag-of-character-bigrams projection", "preserves character n-gram overlap", "doing no better than character-overlap"). The code in `emb_shootout/providers/hash_embedder.py` tokenizes on whitespace (`text.lower().split()`) and then joins n-gram-sized windows of those tokens — i.e. word-bigrams. For a research-flavored repo whose opening paragraph promises the opposite of "published shootouts hide their corpus, vary their k", a wrong description of the baseline is exactly the credibility leak the repo can't afford.

I picked the cheap, reversible direction: fix the prose to match the code, not the other way around. The committed `results/hash.json` numbers, the existing snapshot tests (recall@k, NDCG, latency, corpus count, query count, methodology decision IDs), and the provider name (`hash-embedder-128d-ngram2`) all stand unchanged. To prevent the drift from recurring, `HashEmbedderProvider` now exposes a structured `self.tokenizer = "word"` attribute and validates the kwarg — any future character-n-gram variant has to ship as its own provider per D-007 (one JSON per provider, aggregator merges). A new snapshot test indexes into that attribute and asserts the Takeaways prose uses the matching descriptor: required "word-bigrams / word n-gram overlap / word-overlap", forbidden "character-bigrams / character n-gram overlap / character-overlap". If anyone re-introduces the wrong wording, the test fails before merge.

Why prioritized: this is the first session after v0.1 reached "all twelve repos shipped" cadence. The portfolio's pending open issues are all `[demo]` priority:low waiting on real screencast capture; the protocol's Phase A step 5 fall-through for that case is "file one issue that fills in real README content, then work on it." A factual correction to the README's narrative qualifies and is hermetic. Open questions / followups: none. The same prose-vs-code drift could exist in any other repo whose README claims something its source code subtly doesn't do; a portfolio-wide audit pass is worth doing, but each repo's audit is its own ~20-min session, not bundled.

## 2026-05-22 — Issue #19: architecture doc reflects all six shipped layers, not the corpus-PR-only pre-shipping state

**Duration:** ~30 min. **Issue:** [#19](https://github.com/jt-mchorse/embedding-model-shootout/issues/19). **PR:** [#20](https://github.com/jt-mchorse/embedding-model-shootout/pull/20).

`docs/architecture.md` was committed alongside the corpus PR (issue #1) and never reframed when issues #2 (sweep harness), #3 (Pareto frontier), #5 (notebook reproducer), #11 (benchmarks-md aggregator + snapshot), #13 (public-surface lock), and #15 (capture script) shipped over the following months. The L3 section title still said "Shipped (this PR — issue #1)", the mermaid diagram marked the sweep / queries / metrics / Pareto layers as `pending` (the yellow `classDef pending`) while `results/hash.json`, `docs/benchmarks.md`, and the `emb_shootout/{sweep,pareto,plot}.py` modules had been on disk and exercised by CI for months. The bottom-of-doc "Pending" section described shipped surfaces as future work. The root README was already correct (locked by `tests/test_readme_snapshot.py`); only `docs/architecture.md` had drifted.

Rewrote the diagram to a steady-state pipeline: every shipped surface in green; only the paid-provider sweeps (OpenAI / Voyage / Cohere / BGE / Nomic, which need API keys) and the matplotlib renderer (behind the `[plot]` extra) stay yellow, because both are *wired* end-to-end but require operator-supplied inputs. Each diagram node carries its origin issue annotation as a `(#NN)` token. Replaced the bullet-point "Components shipped" with a richer enumeration covering every public Python module (`corpus`, `queries`, `sweep`, `providers`, `pareto`, `plot`, `cli`) plus the notebook reproducer and the capture script. Replaced "Pending" with "What's still operator-supplied" naming exactly what genuinely needs operator action: paid-provider sweeps, the second-point-on-the-Pareto-frontier rendering, and the 60-second walkthrough binary (`docs/demo.{gif,mp4}` tracked by #16; the capture script itself shipped in #15). Dropped the "this PR — issue #1" framing — the doc is now a steady-state reference.

Lock-against-drift: `tests/test_architecture_doc.py` is the Python sister of `mcp-server-cookbook`'s `tools/check-architecture-doc.{mjs,test.mjs}` (also landed this session in PR #23) and of `nextjs-streaming-ai-patterns` / `ai-app-integration-tests`'s `test/architecture-doc.test.ts` (vitest). Three invariants: every backtick-quoted `emb_shootout/...`, `results/...`, etc. path token resolves on disk; every closed-feature-issue in `KNOWN_SHIPPED_ISSUES = (1, 2, 3, 5, 11, 13, 15)` is referenced at least once; banned phrases (`this pr`, `pending issue`, `*(unfiled)*`, `to-be-filed`) are absent. Three additional belt-and-braces hard-pin tests lock `BANNED_PHRASES`, `KNOWN_SHIPPED_ISSUES`, and `RESOLVABLE_PREFIXES` to their exact contents so a future loose edit of the test that drops one entry fails immediately. Tamper-verified three ways (reinject "this PR" in the title; remove all three `#2` references; add a quoted `emb_shootout/nonexistent.py`); each fires the relevant assertion with the specific drift quoted in the error message. Full suite 99 / 99 (was 92; +7 new, no existing test changed); `ruff check . && ruff format --check .` clean.

Twelfth post-v0.1 drift fix in the portfolio pattern, third architecture-doc lock test in this session. Same family: one part of the docs claims a state the rest of the repo contradicts. The portfolio now has six repos with an architecture-doc lock test (the three from this session + two earlier).

**Why this work, this session:** Loop iteration in a day session. Earlier in the session, the same drift pattern was fixed in three other repos (`nextjs-streaming-ai-patterns`, `ai-app-integration-tests`, `mcp-server-cookbook`); this is the natural fourth strike. Filed as a fresh `priority:med` issue mid-session per the session prompt's "If the repo has zero open issues, file one issue that fills in real README content for that repo per §2 spec" handling, generalized to "architecture doc" since the README was already correct.

**Open questions / blockers:** None — PR opened ready for review.

**Next session:** Loop forward to another zero-issue repo missing an architecture-doc lock test; `chunking-strategies-lab` (build sequence position 6) is the natural next target if pace allows.

## 2026-05-23 — Architecture-doc active-decision-range axis (#21)

**Duration:** ~15 min. **Issue:** [#21](https://github.com/jt-mchorse/embedding-model-shootout/issues/21). **PR:** [#22](https://github.com/jt-mchorse/embedding-model-shootout/pull/22).

Sixth of twelve repos to ship the active-decision-range upper-bound axis on its architecture-doc lock (sister to `llm-eval-harness` PR #32 earlier today + four repos this week). Today's `docs/architecture.md` cites D-002…D-008 in full, so the new test passes as a posture lock on first run with no drift to backfill — fast follow-on, hours not days. Tamper-verified three axes (D-099 in MEMORY, D-002 → D-XXX in doc, MIN constant flip).

**Why this work, this session:** Second issue in today's multi-issue loop after llm-eval-harness PR #32. The pattern was missing in 8 of 12 arch-doc tests; this is fast extension work where the doc already happens to be in compliance.

**Open questions / blockers:** none — PR ready for review.

**Next session:** Apply same pattern to `vector-search-at-scale`, `prompt-regression-suite`, `agent-orchestration-platform` in the same loop.

## 2026-05-24 — Issue #23: `sweep aggregate --format markdown|json`

**Duration:** ~20 min. **Issue:** [#23](https://github.com/jt-mchorse/embedding-model-shootout/issues/23). **Branch:** `session/2026-05-24-0337-issue-23`.

`emb-shootout sweep aggregate` only emitted a markdown table. CI consumers that wanted to assert `recall@5 >= 0.7` for a specific provider had to parse the markdown — the per-provider `results/*.json` files exist, but there was no aggregated cross-provider view that matched the markdown columns.

Added an `aggregate_json()` helper mirroring `aggregate_markdown`'s column set: returns `{"results": [<row>...], "ks": [...]}`, rows sorted by `embedder_name` so the JSON order matches the markdown table row-by-row (downstream consumers can diff the two formats). Extracted the ks-union derivation into a shared `_aggregate_ks` helper so both renderers can't drift apart. CLI gains `--format markdown|json` with markdown as the default — existing CI, the README snapshot test, and the committed `docs/benchmarks.md` workflow are all untouched.

Four new tests cover JSON shape + per-row keys, the markdown-matching row order, the union-not-intersection `ks` derivation across mismatched provider runs, and a regression guard that `aggregate_markdown`'s output bytes are unchanged after the ks helper extraction.

**Why this work, this session:** Fifth issue in the night-session multi-issue loop. The pattern repeats — every repo had at least one CLI parity gap that surfaced from reading the dispatch surface against the README claims.

**Open questions / blockers:** none — PR ready for review.

**Next session:** Continue to build-sequence #6 (`chunking-strategies-lab`).

## 2026-05-24 — Issue #25: `sweep run` accepts `--out` as alias for `--output`

**Duration:** ~15 min. **Issue:** [#25](https://github.com/jt-mchorse/embedding-model-shootout/issues/25). **Branch:** `session/2026-05-24-1532-issue-25`.

`emb-shootout sweep run` was the last path-emitting subcommand in this CLI to use the older `--output` name while `corpus build`, `sweep aggregate`, and `sweep plot --out-png` / `--out-svg` all use `--out`. The asymmetry surfaced every time someone scripted the sweep — `--out` from muscle memory tripped the required-arg guard with `error: the following arguments are required: --output`. After this PR the cookbook's `--out` convention is uniform across every subcommand.

argparse's `add_argument("--out", "--output", ..., dest="output", required=True)` shape makes the fix purely additive — both flag names bind to the same `args.output` destination, so `_cmd_sweep_run`'s body is unchanged and the existing CI / Makefile / shell scripts that already pass `--output` keep working. `--output` is not deprecated, only joined.

Three new tests in `tests/test_cli_sweep_run_out_alias.py`: `--out PATH` happy path producing the documented `RunResult` shape with parent dirs auto-created; `--output PATH` regression-pin so the existing callers stay green; neither-supplied exits 2 under argparse's required-arg machinery. argparse renders the error message as `--out/--output`, the cleanest possible signal to the user that both spellings are accepted.

**Why this work, this session:** Fourth Phase B+C target of a 180-min day session. The earlier targets (`llm-eval-harness` #37, `prompt-regression-suite` #32, `mcp-server-cookbook` #31) were each closing a half-implemented capability — feature parity or observability surface. This one is naming parity: the same flag should mean the same thing across every subcommand of the same CLI.

**Open questions / blockers:** none — PR ready for review.

**Next session:** Continue the day-session loop. Strong candidates: `python-async-llm-pipelines` (#26's `timeout` kwarg landed but `__init__.py` docstring may still list pre-#26 signature); `chunking-strategies-lab` (run_matrix.py recently got `--strategy` filter — neighbors might have similar gaps); `agent-orchestration-platform` (recent retry-cap work; `withRetry` callback semantics could use a doc pin). 

## 2026-05-24 — Issue #27: run_sweep rejects non-positive k in k_values (full set in one pass)
**Duration:** ~20 min · **Branch:** `session/2026-05-24-issue-27`

- `run_sweep` validated that `k_values` was non-empty but didn't check each element. Non-positive `k` passed through `retrieved_ids[:k]` slicing without raising: `k=0` produced a tautological `recall@0=0` baked into the output, and `k<0` silently miscounted via "all but the last N" semantics. **The wrong number, not an absent number.**
- Added a per-element guard that collects every offender in one pass, raises `ValueError(f"every k in k_values must be positive; got {sorted(bad_k)}")`. Operators see the full set of bad values in canonical sorted form so they can copy-paste the fix instead of running N rounds of fix-and-retry. `retrieve_top_k` and `ndcg_at_k` keep their own guards (each is also part of the public surface).
- Six new tests in `tests/test_sweep.py` under a `#27` block: zero raises with `[0]`; negative raises with `[-1]`; mixed `(-3, 0, 5)` lists both bad values as `[-3, 0]` in the message (lints clean with the proper `match=` arg); parametrized positive acceptance over `(1,), (5, 10), (1, 5, 10, 20)` runs cleanly and produces `recall_at_k` keys exactly equal to the input set.

**Why this work, this session:** `retrieve_top_k` and `ndcg_at_k` already raised on non-positive `k` — `k_values` was the one inlet flowing through to a permissive slicing operation. Sister to today's `rag-production-kit` #34 (Retriever k_rrf at construction), `llm-cost-optimizer` #32 (signal-name uniqueness), `llm-eval-harness` #38 (threshold_drop), `prompt-regression-suite` #33 (exception contract). Five repos in a row, same family — "the rest of the surface enforces this; one corner doesn't."

**Open questions / blockers:** none — PR ready for review.

**Next session:** Fifth iteration of today's day-session; within 15-min cleanup buffer of the 180-min cap. Stop after this iteration's PR and write the final report.

## 2026-05-25 — Issue #29: SweepResult validates cost and count fields in __post_init__
**Duration:** ~20 min · **Branch:** `session/2026-05-24-issue-29`

- `SweepResult` at `emb_shootout/sweep.py:58` is a frozen dataclass with `cost_per_million_tokens: float`, `embedder_dim: int`, `n_corpus: int`, `n_queries: int`. No constructor validation. The cost field is the **load-bearing x-axis of the Pareto frontier** at `pareto.py:33-34` — a `SweepResult(cost_per_million_tokens=-1.0, ...)` silently dominates every other point on the frontier, in `docs/pareto.png`, and in the `aggregate_markdown` table's `$` column.
- Added `__post_init__` raising `ValueError(f"{field} ...")` for: `cost_per_million_tokens < 0.0`, `embedder_dim < 1`, `n_corpus < 0`, `n_queries < 0`. Centralized at `SweepResult` (not at the five providers) so user-BYO Embedder protocol implementers benefit automatically without copying the validation per provider. All three construction paths protected: `run_sweep` (production), `from_dict` (deserialization round-trip), direct test fixtures.
- Nine new collected cases in `tests/test_sweep.py` under a `#29` block: parametrized over 3 negative-cost values; parametrized over 4 (count-field × bad-value) combos covering `embedder_dim < 1`, `n_corpus < 0`, `n_queries < 0`; one inclusive-zero acceptance test; one round-trip-corrupt-JSON test that pins `from_dict` raises on tampered cost. `_valid_sweep_result_kwargs()` test helper centralizes the fixture so each negative test only mutates the field under test. Full suite 124/124 (was 115 after #27/#28).

**Why this work, this session:** Mirror of today's `llm-cost-optimizer` PR #35 (`ModelPricing.__post_init__`) and `rag-production-kit` PR #37 (`ModelPrice.__post_init__`). The three cost-aware repos in the portfolio now defend their dashboards consistently against sign-corrupting values. D-006 anchors the contract — operator-supplied cost — extended from "no missing" to "no sign-corrupting".

**Open questions / blockers:** none — PR ready for review.

**Next session:** Fourth Phase B+C target of today's day session after `llm-eval-harness` #40, `llm-cost-optimizer` #34, `rag-production-kit` #36. Time remaining in the 180-min cap permits one more if a viable target surfaces in the remaining repos. Build sequence #6 (`chunking-strategies-lab`) is the natural next pickup.

## 2026-05-25 — Issue #31: SweepResult finiteness + integer guards
**Duration:** ~25 min · **Branch:** `session/2026-05-24-issue-31`

- Four sign-only checks let `NaN`/`+Infinity`/non-int through. `SweepResult.cost_per_million_tokens` accepted NaN which propagated into the Pareto frontier comparator at `pareto.py:33-34` where every NaN comparison is false → dominance check silently degrades. The three count fields (`embedder_dim`, `n_corpus`, `n_queries`) were typed `int` but accepted float/NaN at runtime — NaN dim makes `len(vec) == dim` always false; fractional dim truncates silently. `ndcg_at_k(k)` and `retrieve_top_k(k)` sign-only `<=0` accepted NaN (cryptic TypeError deep inside slicing) and fractional k (silent truncation).
- Tightened `cost_per_million_tokens` to `math.isfinite + >= 0.0`. Tightened the three count fields to `isinstance(int)` (bool excluded explicitly since Python's bool subclasses int) + existing comparison. Tightened both math helpers `k` to `isinstance(int) + positive`. Existing tests with loose `"positive"` / `"must be"` matchers unchanged; three tests pinning the exact old message updated.
- 12 new parametrized tests in `tests/test_sweep.py` under a `#31` block: rejection per field over `[NaN, +Infinity, -Infinity]` for cost; `[1.5, NaN, True, fractional]` for int fields; `[1.5, NaN, True, str]` for math helpers. Test count 148.

**Why this work, this session:** Ninth (and final) Phase B+C target in the 360-min night session. Second PR in embedding-model-shootout tonight; the first was via the Phase A fixup-merge of #30 (sign-only `__post_init__` shipped in #29). The two together close the silent-zero → silent-negative → silent-NaN/Infinity arc on `SweepResult` and on the math helpers it depends on.

**Open questions / blockers:** none — PR ready for review.

**Next session:** Continue the loop across remaining unvisited-tonight-for-second-iteration repos: `chunking-strategies-lab`, `vector-search-at-scale`, `python-async-llm-pipelines`. Each had a fixup-merge today but no Phase B+C finiteness sweep yet.

## 2026-05-25 — Issue #33: provider batch_size sign-only sweep + hoist before lazy import
**Duration:** ~25 min · **Branch:** `session/2026-05-25-1620-issue-33`

- Five lazy-loaded API provider classes (CohereProvider, VoyageProvider, OpenAIProvider, BGEProvider, NomicProvider) shared the identical sign-only `batch_size <= 0` check. All five tightened to the portfolio-wide `isinstance(int) + reject bool + positive` pattern. Two failure modes closed: `Provider(batch_size=True)` constructed cleanly and embedded one item at a time, inflating API call count and cost telemetry; `Provider(batch_size=0.5)` constructed cleanly and crashed inside `range(0, len(items), self.batch_size)` with a far-from-the-call-site `TypeError`.
- Structural change worth noting: validation is now hoisted ABOVE the lazy `import <provider-sdk>` block in each `__init__`. This makes the contract testable in CI without `cohere`, `voyageai`, `openai`, or `sentence-transformers` installed AND fail-fast on misconfig before slow client init. The behavior change is benign — bad input still raises; the error type just becomes `ValueError` (preferred) instead of `ImportError` (when extras missing) or post-client-init `ValueError` (when extras installed).
- New test file `tests/test_provider_batch_size_validation.py` runs a 5 × 10 parametrize matrix (5 providers × 10 bad values) across all five classes with no provider extras installed; +50 net tests (142 → 192). Ruff clean.

**Why this work, this session:** Fourth Phase B+C target in the 180-min day session. After mcp-cookbook README drift, llm-cost-optimizer batch __post_init__, and rag-production-kit retrieval-fusion sweep, embedding-model-shootout's 5-provider uniform pattern was a clean, well-scoped finisher. The hoist-before-lazy-import insight transfers to other repos with lazy-import constructors — file follow-up issues if relevant.

**Open questions / blockers:** none — PR ready for review.

**Next session:** Pattern is now at three repos today (cost batch, rag retrieval, emb providers). If the trending workflow surfaces fresh topics, switch. Otherwise consider `chunking-strategies-lab` strategies sweep (chunk_chars/overlap_chars/max_chunk_chars across 5 strategy classes — same uniform shape as this PR).

## 2026-05-26 — Issue #35: Close #34's deferred validation gaps (hash embedder + query gen)
**Duration:** ~25 min · **Branch:** `session/2026-05-25-1800-issue-35`

- `HashEmbedderProvider.dim` and `.ngram`: replaced sign-only checks with the portfolio positive-int contract. `dim=True` no longer silently produces a 1-element name-tagged provider; `dim=128.0` no longer surfaces as `TypeError: can't multiply sequence by non-int` deep inside `_embed_one`.
- `build_queries.n`, `.min_words`, `.max_words`: each parameter now checked independently with the positive-int contract so the error message names the offending field; the paired `max_words >= min_words` invariant runs after both type contracts.
- 71 new parametrize tests. Pre-existing `test_build_queries_validates_inputs` updated to the new message shape (`"n must be a positive integer"`). `test_hash_provider_validates_dim` used loose `match="dim"` and still passes unchanged. Full suite 192 → 263. Ruff clean.

**Why this work, this session:** Third Phase B+C target in today's 180-min DAY session. PR #34 explicitly listed these five sites as the "Out of scope (file separately if needed)" follow-ups. Closing them keeps the contract uniform across all construction sites in the repo and finishes the portfolio-wide closure of named deferred-lists for the day.

**Open questions / blockers:** PR ready for review. Note for documentation: #34's body called the function `synthesize_queries`; the actual name in `emb_shootout/queries.py` is `build_queries` — issue body documents the rename so future readers can grep both names.

**Next session:** With the three explicit deferred-lists closed (`llm-eval-harness#45`, `rag-production-kit#43`, `embedding-model-shootout#36`), the next natural target is a survey-driven discovery pass on the remaining repos for boundary-validation gaps that match the same shape.

## 2026-05-26 — Issue #37: Add `emb_shootout/io_utils.atomic_write_text`, route cli / corpus / notebook writes through it
**Duration:** ~30 min · **Branch:** `session/2026-05-26-1930-issue-37`

- The 2026-05-26 morning atomic-write arc shipped helpers across six portfolio repos (rag-production-kit, llm-eval-harness, llm-cost-optimizer, prompt-regression-suite, mcp-server-cookbook, ai-app-integration-tests). This repo wasn't part of that pass and had four production write sites still using `Path.write_text` or the streaming `open("w") + f.write()` shape, both non-atomic.
- The four routed sites: `cli.py:_cmd_sweep_run` (per-provider result JSON written under `results/` and consumed by both aggregator and plot per D-007); `cli.py:_cmd_sweep_aggregate` (renders the markdown that the README's "Benchmarks" section is built from); `corpus.py::write_jsonl` (the corpus that every sweep reads — row-oriented, so a truncation at a row boundary passes the parser silently and quality numbers drift down without a loud signal); and `notebooks/_build_notebook.py:main` (the reproduce notebook). The corpus helper changed shape from streaming-write to build-string-in-memory-then-atomic-write; D-002 bounds corpus size at one chunk per stdlib member, on the order of a few thousand rows — fits in memory comfortably, and the streaming shape's only upside was memory amortization at scales the spec doesn't reach.
- D-009 codifies the package-level `io_utils` pattern for this repo, matching `rag_kit/io_utils.atomic_write_text` (rag-production-kit#44/#45) and `eval_harness/io_utils.atomic_write_text` (llm-eval-harness#51, D-015). Tests: 6 unit tests on the helper (happy path, parent-dir create, overwrite, three load-bearing failure invariants — destination-absent on rename failure, no leftover `.tmp` siblings, pre-existing-file-unchanged on overwrite failure) and 4 per-call-site integration tests proving each routes through the helper. The corpus write test pins the new load-bearing invariant the prior `open("w")` shape could never offer: a corpus re-build that fails mid-write leaves the prior corpus intact so subsequent sweeps aren't silently degraded. Full suite 263 → 273. Lint/format green.

**Why this work, this session:** Second Phase B target of today's 180-min DAY session, after Phase A merged six atomic-write PRs and Phase B's first issue (llm-eval-harness#50) promoted that repo's helper to module level. This repo had the most non-atomic write sites of the five repos NOT in the morning arc (others have zero or one). Closing it brings portfolio atomic-write coverage to 7 of 12 repos. The other four candidates (`agent-orchestration-platform`, `nextjs-streaming-ai-patterns`, `python-async-llm-pipelines`, `chunking-strategies-lab`, `vector-search-at-scale`) need a similar scan; `agent-orchestration-platform` already known to have two non-atomic `fs.writeFile` sites (TypeScript pattern, separate PR shape).

**Open questions / blockers:** none — PR ready for review.

**Next session:** Continue the multi-issue loop. Plausible next targets: (a) `agent-orchestration-platform` (TypeScript parallel of this PR — `src/bin/eval-runner.ts:72` `fs.writeFile` for eval output, `scripts/render-eval-snapshot.ts:25` for markdown render), (b) scan the remaining four repos for atomic-write gaps and close any found, (c) pivot to a different harm class entirely (input-trust on external API responses, resource leaks on error paths, test-determinism guarantees).

## 2026-05-26 — Issue #39: README decision-range upper-bound lock
**Duration:** ~7 min · **Branch:** `session/2026-05-26-2328-issue-39`

- Added `tests/test_readme_decision_range.py`.
- Added `D-002…D-009` citation in `## Why these decisions`.

**Why this work, this session:** Propagation 5 of 10 of the cross-portfolio drift class authored in chunking-strategies-lab.

**Open questions / blockers:** none.

**Next session:** Continue propagation to vector-search-at-scale.

## 2026-05-27 — Issue #41: drop stale "· this PR" from sweep-harness header + banned-phrase lock
**Duration:** ~10 min · **Branch:** `session/2026-05-27-0328-issue-41`

- `README.md:163` carried PR-time framing ("· this PR") for the sweep harness — the repo's primary shipped surface, not a PR in flight. Same drift class `prompt-regression-suite#43` and `llm-cost-optimizer#46` just resolved.
- Rewrote the header to steady-state form.
- New lock: `tests/test_readme_banned_phrases.py` with `BANNED_PHRASES = ("this pr",)` + hard-pin tuple test. Mirrors the lock from `prompt-regression-suite#43`.
- Smallest diff in the propagation arc so far (only one stale header here vs. four in the prior two repos), but the lock is identical and future-protects.

**Why this work, this session:** Iteration 5 of an autonomous NIGHT session, third repo in the README banned-phrase lock propagation arc.

**Open questions / blockers:** none — PR ready for review.

**Next session:** `python-async-llm-pipelines` (2 hits) is the last known repo with this drift; loop continues.

## 2026-05-27 — Issue #43: CONTRIBUTING.md cadence-wording propagation
**Duration:** ~3 min · **PR:** #44

- Replaced pre-D-008 `~60-minute session cap` line with D-008 (180/360 min, multi-issue loop) and D-004 (Phase A PR auto-merge) wording, matching the bootstrap template post-portfolio-ops#3.

**Why this work, this session:** Iteration in the autonomous NIGHT session propagation arc for portfolio-ops#3.

**Open questions / blockers:** none.

**Next session:** continue portfolio propagation.

## 2026-06-01 — Issue #45: Corpus-JSONL collecting-mode validator
**Duration:** ~13 min · **Branch:** `session/2026-06-01-2311-issue-45`

- Shipped `emb_shootout.validate.validate_corpus(path)` — walks a corpus JSONL in collecting mode and returns a frozen `ValidationReport` with `n_rows`, `n_valid`, and a tuple of findings, each `(line_no, reason, code)`. Ten finding codes cover `json.loads` failures (`malformed_json`, `not_an_object`), schema gaps (`missing_chunk_id`, `missing_text`, `non_string_*`, `empty_*`), uniqueness (`duplicate_chunk_id`), and the empty-file case. Blank lines silently skipped so `validate` and `_read_corpus_jsonl` agree on row counts.
- Wired `emb-shootout corpus validate <path> [--json]` into `cli.py`. Exit codes 0 clean / 1 findings / 2 I/O error — same as `eval-harness validate` and `prompt-snap validate` so consumers can chain validators uniformly.
- Coverage: `tests/test_validate.py` is 21 cases — happy path, accumulating-errors (does not fail fast), one parametrized positive per finding code, duplicate detection (shadowed row not counted as valid), blank-line silent-skip, empty-file `empty` finding, `FileNotFoundError`, `to_dict` JSON shape, `frozen=True` dataclass shape lock, and three CLI end-to-end cases.
- README "What this is" gains a fifth bullet and `[#45]` reference link. `docs/architecture.md` gains a `validate` module bullet and `corpus validate` to the CLI subcommand list with `(#45)` on each; `tests/test_architecture_doc.py` `KNOWN_SHIPPED_ISSUES` and its hard-pin tuple extended to include `#45`.
- Live-tested against the real `data/corpus.jsonl` (12,010 CPython stdlib rows): exit 0 in one pass with `ok: ... rows=12010 valid=12010 findings=0`. Confirms no false positives on a healthy corpus. Full suite 298 / 298 pass, ruff clean.

**Why this work, this session:** First iteration of a multi-issue day-session. Phase A merged four observability-parity PRs (one each in eval-harness, prompt-regression-suite, cost-optimizer, rag-kit) leaving every portfolio repo with zero open issues. The pattern that just shipped in two of those repos — collecting-mode validate as a pre-flight before fail-fast input readers — generalized cleanly to emb-shootout's `_read_corpus_jsonl`. Filed the issue, posted the plan, then shipped.

**Open questions / blockers:** None — ready for review. Result-JSON validate (`sweep validate-results`) is a low-priority follow-up only if the pattern proves out further.

**Next session:** Continue the day-session loop on another repo that hasn't been touched since 2026-05-27. `chunking-strategies-lab` is next in build-sequence order among the still-untouched repos and is the natural home for the next iteration.

## 2026-06-02 — Issue #47: Chunk.to_dict + explicit SweepResult.to_dict
**Duration:** ~18 min · **Branch:** `session/2026-06-02-0410-issue-47`

- Closed the last two `dataclasses.asdict` usages in this repo:
  - `emb_shootout/sweep.py`: `SweepResult.to_dict` was using `asdict(self)` with a `recall_at_k` post-processing step. Rewrote explicit field-by-field (9-field contract). Preserves the int→str transformation for `recall_at_k` keys (JSON has no integer key type). `notes` is shallow-copied so caller mutation doesn't bleed back into the frozen dataclass. `asdict` import dropped.
  - `emb_shootout/corpus.py`: `Chunk` had no `to_dict` at all; the load-bearing `chunks.jsonl` artifact was shaped by raw `asdict(chunk)`. Added `Chunk.to_dict` (6-field contract); replaced the asdict call site in `_write_corpus_jsonl`. `asdict` import dropped.
- 7 new tests: 4 in `test_sweep.py` (sorted-keys pin, `recall_at_k` key stringification, `from_dict` round-trip regression to confirm the explicit shape parses identically, notes list-copy guard) + 3 in `test_corpus.py` (Chunk sorted-keys pin, value round-trip, **`_write_corpus_jsonl` shape acceptance regression** asserting every JSONL row has the 6-field set). Full suite 298 → 305 pass. Ruff check + format clean.
- `grep -rn asdict emb_shootout/ scripts/` returns only documentation references (the inline comments describing the prior shape). No source-level asdict serialization remains.

**Why this work, this session:** Iteration 8 of the night session loop. Audit of recently-touched Python repos surfaced `embedding-model-shootout` as the last one in the observability-parity arc with remaining `asdict` reliance. Closing both surfaces completes the arc at six Python repos.

**Open questions / blockers:** none — ready for review.

**Next session:** Portfolio observability-parity arc is now fully saturated across all Python JSON-emitting repos at both package and script levels. Future iterations should pivot to either operator-blocked items or novel parity opportunities outside the asdict / to_dict arc.

## 2026-06-17 — Issue #49: Workflow YAML-parseability lock
**Duration:** ~8 min · **Branch:** `session/2026-06-17-1922-issue-49`

Added `tests/test_workflows_yaml_parseable.py` (3 tests for `ci.yml`)
and pulled `pyyaml>=6.0` into `dev` extras.

**Why this work, this session:** Seventh hop of the `portfolio-ops#30`
propagation arc.

**Open questions / blockers:** none — PR #50 open.

**Next session:** continue propagation to the remaining 5 repos.

## 2026-06-18 — Issue #51: timeout-minutes guard + lock test
**Duration:** ~15 min · **Branch:** `session/2026-06-18-0324-issue-51`

- Added `timeout-minutes: 15` to every job in `ci.yml` (`lint`, `test`, `memory-check`).
- Added `tests/test_workflows_timeout_minutes.py` — 10 new tests (1 smoke + 3 jobs × 3 parametrized invariants).

**Why this work, this session:** part of the portfolio-wide propagation of the timeout-minutes silent-rot lock. `llm-eval-harness` PR #63 shipped first; this is the fifth propagation hop. The portfolio-ops audit (#36) surfaces every unprotected repo weekly; this PR drops `embedding-model-shootout` from its findings.

**Open questions / blockers:** none. Full pytest + ruff clean.

**Next session:** continue propagation to the remaining 6 unprotected repos.

## 2026-06-18 — Issue #53: concurrency guard + lock test
**Duration:** ~8 min · **Branch:** `session/2026-06-18-1529-issue-53`

- Added top-level `concurrency:` to `ci.yml`.
- Copied lock test from llm-eval-harness; docstring origin updated.

**Why this work, this session:** seventh per-repo hop in the
concurrency-lock arc.

**Open questions / blockers:** none. Test count 315 → 322.

**Next session:** continue propagation to remaining 5 repos.

## 2026-06-19 — Issue #55: emb-shootout corpus validate --out (close arc)
**Duration:** ~22 min · **Branch:** `session/2026-06-19-0330-issue-55`

- Added `--out PATH` to `emb-shootout corpus validate` (the last hop
  in the four-repo validate-CLI sink-parity propagation arc).
- `_cmd_corpus_validate` builds the rendered string once, routes
  through `emb_shootout/io_utils.atomic_write_text` when `--out` is
  set, else `sys.stdout.write(rendered)`. Findings continue to print
  to stderr in human-readable mode regardless of `--out`.
- 6 new tests; README unchanged.

**Why this work, this session:** closing the arc. After this PR, all
four Python validate CLIs in the portfolio share one shape — one
atomic-write helper per repo, one operator workflow, one stderr
findings contract.

**Open questions / blockers:** none. 322 → 328 pytest passes. PR #56
open and ready.

**Next session:** consider extending the validate-CLI shape to a
`--format` choice covering text/json/sarif so CI workflows can route
shape uniformly across repos. Or close additional sibling propagation
gaps that surfaced (e.g., bench-script `--out` in rag-production-kit).

## 2026-06-22 — Issue #57: run_sweep — reject duplicate k_values
**Duration:** ~25 min · **Branch:** `session/2026-06-22-1916-issue-57`

- Found via a Phase A Explore-subagent sweep over the core package (sweep/pareto/corpus/queries/io_utils/validate/providers) — the only real bug in an otherwise-saturated repo, picked because emb-shootout was the staleest (>36h) at build-sequence position 5. `run_sweep`'s per-query loop iterates `k_values` directly and increments `hits_at_k[k]` once per occurrence, while `hits_at_k` and the output `recall_at_k` dict carry one entry per *distinct* k. So a k appearing N times counted each hit N times: `k_values=(5, 5)` produced `recall@5 = 2.0` — a mathematically invalid number from a benchmark whose whole job is trustworthy retrieval scores.
- This is the sibling gap of the `#27` per-element k guard: `#27` rejects non-positive k fail-loud, but every `#27` test uses distinct k values, so the duplicate case fell through. Fix extends that same block to reject duplicates one-pass (every duplicated value surfaced once, sorted; non-duplicates omitted). Chose reject over silent de-dupe to match the adjacent fail-loud philosophy — a duplicate k in a sweep signals an operator mistake, and the result dict is keyed by k so a duplicate carries no distinct meaning.
- 3 new tests (rejection, all-dupes-in-one-message, and a pin that the duplicate form never reaches the corrupt computation). Verified all 3 fail pre-fix. Suite 328 → 331, ruff clean. PR #58 ready.

**Why this work, this session:** the repo had zero open issues; rather than a synthetic README fill, a dogfood sweep surfaced a real silent-corruption bug in the headline metric — strictly higher value.

**Open questions / blockers:** none.

**Next session:** the k_values guard is now complete on both the non-positive (#27) and duplicate (#57) axes. No specific lead remains in `sweep.py`; pareto/corpus/queries scanned clean this session.

## 2026-06-23 — Issue #59: validate_corpus skipped the duplicate-chunk_id check on field-error rows
**Duration:** ~20 min · **Branch:** `session/2026-06-23-1940-issue-59`

- After fixing the same bug in `chunking-strategies-lab` (#60), a cross-repo propagation check found the shared collecting-mode validate pattern carried the bug here too: `validate_corpus` early-`continue`d on any field finding before the duplicate-chunk_id check, so a row that was both a duplicate and field-invalid reported only the field error, and the duplicated chunk_id was never recorded.
- Mirrored the chunking fix: run the dup check independently, guarded on a valid `chunk_id`, register valid chunk_ids even on field-invalid rows, count `n_valid` only for fully-clean rows. Also audited `prompt-regression-suite`'s validator and confirmed it is not affected (its continue is structurally forced when the snapshot fails to load). Suite 331 → 333, ruff clean.

**Why this work, this session:** fifth dogfood find of the DAY session, and the highest-leverage kind — a real silent-rot instance that had propagated across sibling repos with a shared pattern.

**Open questions / blockers:** none.

**Next session:** none specific to this issue.

---
## 2026-06-24 — Issue #61: SweepResult didn't validate recall@k / nDCG@10 values
**Duration:** ~25 min · **Branch:** `session/2026-06-24-0407-issue-61`

- `SweepResult.__post_init__` guarded cost (finite/≥0, #31), `embedder_dim`, and the counts — but not the metric values. recall@k and nDCG@10 are [0,1] proportions, yet a corrupt/hand-edited result (loaded via `from_dict`) with recall=1.5 or NaN was accepted, silently winning the Pareto dominance comparison and rendering nonsensical plot points.
- Validated each `recall_at_k` value and `ndcg_at_10` are finite and in [0,1] with a descriptive ValueError, covering both direct construction and the `from_dict` read path. Same loader value-validation class as chunking #62.
- 16 new tests (parametrized out-of-range/non-finite recall + ndcg, inclusive boundaries, from_dict tampered recall). Red via `git stash`, green after. Suite 333 → 349, ruff clean.

**Why this work, this session:** embedding-model-shootout was next non-tier in build sequence; sweep/validate/corpus were saturated, so a dogfood sweep of the Pareto/metric path surfaced this asymmetry in the existing `__post_init__` guard chain.

**Open questions / blockers:** none.

**Next session:** pareto.py / plot.py / the provider modules remain the dogfood frontier; `embed_latency_ms` value validation is a small deferred follow-up.
