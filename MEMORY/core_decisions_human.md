# Core Decisions

Strategic decisions for this repo, with reasoning. Append-only — superseded decisions are marked, not removed.

## D-001 — Scope locked to portfolio handoff §2 (2026-05-10)
**Decision:** Scope of this repo is fixed by the portfolio handoff document, section 2.

**Why:** The handoff spec was deliberated; ad-hoc scope expansion within a session is the failure mode this prevents.

**Alternatives considered:** None — this is a baseline.

**Reversibility:** Expensive. Scope changes require a deliberate revisit and a new decision entry.

**Related issues:** —

## D-002 — Corpus is reproducible from source, not committed as data (2026-05-14)
**Decision:** The benchmark corpus (CPython stdlib docstrings) is regenerated from the running Python interpreter via `inspect`, not committed to the repo as a static `data/corpus.jsonl` file. A user running `emb-shootout corpus build` on the same Python version produces the same corpus, deterministically.

**Why:** Three reasons. (1) Reproducibility: pinning the corpus to "what Python ships" makes the corpus precisely auditable — anyone can run `python -c "import json; help(json.JSONEncoder)"` and see the source text. (2) Repo size: a 12k-chunk JSONL is several megabytes; for a repo whose value is the *methodology*, committing the data adds nothing but bloat. (3) Licensing clarity: re-distributing the docstrings in a separate file invites unnecessary attribution complexity; running them out of the live interpreter sidesteps it.

**Alternatives considered:**
- Commit `data/corpus.jsonl` directly — rejected; adds MBs to the repo for no methodological gain, and the loader's `__all__`-handling logic is the actual contribution, not the byte stream.
- Fetch the corpus from a remote URL at build time — rejected; introduces network dependence and a third-party point of failure for what is fundamentally a local operation.

**Reversibility:** Cheap. If a future requirement demands a committed corpus snapshot (e.g., to lock the benchmark to one Python version regardless of where it runs), `emb-shootout corpus build --out data/corpus.jsonl` produces it.

**Related issues:** #1, #2.

## D-003 — Chunk shape is one stdlib member = one chunk (2026-05-14)
**Decision:** Each documented stdlib member (module, class, function, method) becomes one chunk. Multi-paragraph docstrings stay together within the chunk; class members emit as separate chunks under the class.

**Why:** The retrieval task this corpus benchmarks is "find the answer to a developer's question." The unit of answer is one symbol's documentation — splitting a docstring across multiple chunks fragments the answer (the embedder would return half of a method's behavior); merging a module's overview with all its members would dilute every chunk's topical specificity. One-member-one-chunk lines up with how the corpus is consumed at the call site.

**Alternatives considered:**
- Split long docstrings by paragraph — rejected; embedders would surface fragments that lack the full behavior.
- Merge a module's docstring with each of its members' — rejected; every chunk would be dominated by repeated module-level boilerplate, hurting topical specificity.

**Reversibility:** Cheap. The chunk-emission logic is one function (`_iter_module_members`); changing the granularity is a localized edit, and the chunk shape's `kind` field already distinguishes module/class/function/method.

**Related issues:** #1, #2.

## D-004 — `Embedder` is a single-method Protocol (2026-05-15)
**Decision:** `emb_shootout.sweep.Embedder` is a Protocol with `embed(texts) -> list[list[float]]` plus `name`, `dim`, `cost_per_million_tokens` properties. All six providers conform structurally.

**Why:** Same single-method Protocol seam as the rest of the portfolio. One-line wrapper to swap providers in or out.

**Alternatives considered:**
- Hard-coded OpenAI client — rejected: vendor lock-in.
- Abstract base class — rejected: ceremony.
- sklearn-style estimator — rejected: no `fit`.

**Reversibility:** Cheap.

**Related issues:** #2

## D-005 — Queries derived from corpus at sweep time, deterministic seed (2026-05-15)
**Decision:** `build_queries(corpus, n=200, seed=42)` derives queries by picking random chunks (with replacement) and taking verbatim word-window snippets. Seed pins reproducibility.

**Why:** Pre-committed query sets drift as the corpus changes. CPython's stdlib evolves between Python versions, which moves the corpus on every release. Derivation-at-sweep-time keeps corpus + queries always in sync.

**Alternatives considered:**
- Pre-committed `data/queries.jsonl` — rejected: drifts; needs re-curation per Python version.
- Hand-curated query set per corpus version — rejected: doesn't scale.

**Reversibility:** Cheap.

**Related issues:** #2

## D-006 — `cost_per_million_tokens` is operator-supplied at provider construction (2026-05-15)
**Decision:** Each provider takes `cost_per_million_tokens` as a constructor arg, defaulting to public list price as of 2026-05. Recorded on `SweepResult` so historical comparisons capture the price the operator used at run time.

**Why:** Embedding pricing changes. Hard-coding would mean recorded $/MTok drifts away from reality and historical results become uninterpretable. Operator-supplied + recorded means the JSON files are self-describing.

**Alternatives considered:**
- Hard-coded in provider module — rejected: drifts.
- Fetch from provider pricing API at runtime — rejected: no public APIs; brittle.

**Reversibility:** Cheap.

**Related issues:** #2

## D-007 — One result JSON per provider; aggregator merges them (2026-05-15)
**Decision:** `emb-shootout sweep run --provider X --output results/X.json` writes one JSON per provider. `emb-shootout sweep aggregate` reads `results/*.json` and emits the markdown table.

**Why:** Per-provider JSON files compose cleanly: parallel operator runs don't collide, partial state is recoverable, aggregator is pure-read. Single-file shapes invite concurrent-write bugs; SQLite would be overkill for tens of rows.

**Alternatives considered:**
- Single `results.jsonl` appended per run — rejected: concurrent runs collide.
- SQLite results DB — rejected: dep for `n < 100` rows is overkill.

**Reversibility:** Cheap.

**Related issues:** #2

## D-008 — Pareto axes are `cost_per_million_tokens` (x) and `recall@5` (y); frontier math is dep-free, matplotlib renderer behind a `[plot]` extra (2026-05-16)
**Decision:** The Pareto plot plots cost on the x-axis and recall@5 on the y-axis, matching the acceptance criteria on issue #3. Frontier *selection* (`pareto_frontier`) is pure-stdlib Python in `emb_shootout.pareto` and ships in the base install. The matplotlib *renderer* (`emb_shootout.plot`) is lazy-imported behind a new `plot = ["matplotlib>=3.8"]` optional extra. The CLI subcommand `emb-shootout sweep plot` lazy-imports the renderer so the CLI loads without the extra installed.

**Why:** The acceptance criteria explicitly names cost vs. recall@5 — that decision is already made by issue #3 and adding a latency axis would be scope creep. Splitting frontier math from rendering means the math gets exercised in the standard CI matrix (no extras) on every PR, which is where regressions tend to hide, while keeping the core package dep-free. This parallels D-004's provider-extras pattern: real work behind an opt-in install. A `plot` extra is the smallest new surface that still ships a publication-quality figure.

**Alternatives considered:**
- Add latency as a third axis or face a 2D NDCG vs. cost plot — rejected; not in the acceptance criteria, and a third axis is hard to read in a 2D PNG. Filing a separate issue is cheaper.
- Put matplotlib in the base install — rejected; breaks the dep-free default and adds a heavy transitive (numpy, fonttools, etc.) to every CI run.
- Hand-roll an SVG renderer to avoid the extra — rejected; gives up axis labels, legends, and standard styling that matplotlib provides for free.

**Reversibility:** Cheap. The Pareto module is one file, the renderer is one file, the CLI subcommand is one function, the extra is one line in `pyproject.toml`.

**Related issues:** #3
