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

## D-009 — Atomic-write helpers live in package-level `io_utils` (2026-05-26)
**Decision:** Atomic-write helpers in this repo live in a package-level module at `emb_shootout/io_utils.py`, exposing public `atomic_write_text(path, text, encoding="utf-8")`. The pattern mirrors `rag_kit/io_utils.atomic_write_text` from `rag-production-kit#44/#45` and `eval_harness/io_utils.atomic_write_text` from `llm-eval-harness#51` (D-015 there).

**Why:** The 2026-05-26 atomic-write arc landed similar helpers across six other portfolio repos. Each repo that converged on a package-level helper kept the test surface clean (one `io_utils.os.replace` to monkey-patch), let every module in the package reach the helper without re-implementing the pattern, and stayed consistent with the cross-repo standard. File-private helpers (the shape `llm-eval-harness#49` originally landed and that D-015 superseded) fragment the test surface and prevent cross-module reuse. Re-implementing the pattern at each call site is the worst shape — every copy can drift, every needs its own tests.

**Alternatives considered:**
- File-private helper per module — rejected; the call sites are in three modules (`cli.py`, `corpus.py`, `notebooks/_build_notebook.py`) so this would mean three copies of the same 25 lines. The llm-eval-harness arc landed on D-015 specifically to retire this pattern.
- Separate distribution package — rejected; over-engineering for a 25-line helper with one consumer.
- In-place re-implementation at each call site — rejected; same drift hazard as per-module copies, and no central test surface.

**Reversibility:** Cheap. The helper is two dozen lines and a stable API; any future evolution is a localized rewrite.

**Related issues:** #37

## D-010 — an unmeasured aggregate cell is absent, never 0.0 (2026-08-25)

**Decision.** Every cell of both aggregate formats distinguishes "not measured"
from "measured zero". Markdown renders an em dash (the `ABSENT_RECALL_CELL`
convention #123 introduced); JSON renders `null`. This covers `recall_at_k` and
all three `embed_latency_ms` fields, in both `aggregate_markdown` and
`aggregate_json`.

**Why.** #123 established the rule — "do not invent benchmark numbers" — and
applied it to one cell of eight: `recall_at_k` in the markdown aggregator. The
other seven still published a fabricated `0.0`.

Two harms followed. The two formats disagreed about the same cell: for a result
swept without `k=5`, markdown said `—` and JSON said `0.0`, while
`aggregate_json`'s own docstring promises a consumer can "cross-check the two
formats line-by-line" — and the JSON is the format CI actually parses. And for
latency, the fabricated default is not merely wrong, it is the *best possible
value*: a provider that reported no timings read as `0` ms corpus embed, `0.0` ms
p50 and `0.0` ms p95, and won any "which is fastest" read of the published
benchmark. A default landing at an extreme of a comparison does not abstain, it
ranks.

**Alternatives considered.** *An em dash in JSON too* — rejected; it turns a
number column into a string column for a typed consumer, and JSON has a spelling
for absent that markdown does not. *Omitting the key* — rejected; a missing key
and a null key are different contracts, and the column set is a property of the
aggregate (the union of every result's `k`), not of an individual row. *Rejecting
a partial `embed_latency_ms` at `from_dict`* — rejected for the reason #123 gave
about its own case: a sweep result is a *reported measurement*, and a provider
that legitimately cannot time its calls should still be comparable on recall and
cost. *Leaving the JSON alone because no committed artifact has absent cells* —
rejected; the external-result-file path through `from_dict` is exactly the
reachability #123 cites, and `sweep aggregate` walks it for every
`results/*.json`.

**Reversibility.** Cheap. Recorded because it widens a documented machine
artifact's field type: `recall[k]` and the three `*_ms` fields go from `number`
to `number | null`. No committed artifact moves — `results/hash.json` carries all
three `k`s and all three latency keys — so `docs/benchmarks.md` is byte-identical.

## D-011 — a generated markdown artifact marks the region the generator owns

**Date:** 2026-09-11 · **Reversibility:** cheap · **Issues:** #145

**Decision.** `docs/benchmarks.md` carries
`<!-- emb-shootout:table:begin -->` / `<!-- emb-shootout:table:end -->` around the
aggregator's table, and `emb-shootout sweep aggregate --out <file>` replaces only
the text between those markers when the destination has them. A destination
without markers — a scratch path, or a file intended to be table-only — is written
whole exactly as before.

**Why.** `docs/benchmarks.md` opens by telling the operator that the file "is
**regenerated** by `emb-shootout sweep aggregate`" and "Don't hand-edit". Running
that command as documented took the file from 44 lines to 3. It deleted the
`## Current results` framing, the interpretation paragraph, the whole
`## Reproducing` section, the apples-to-apples note, and the sentence "Per the
no-fabricated-benchmarks rule, this README does not carry placeholder numbers for
those providers" — which is this portfolio's first quality rule written down in the
one file where the numbers live.

And all 873 tests stayed green. `test_benchmarks_md_snapshot.py` locks the artifact
by *containment*: it asserts the aggregator's table is **in** the file, and a file
truncated *to* the table still contains the table. A containment lock cannot see a
deletion, and a green suite is exactly why an operator would have believed the
regeneration had gone fine. So the fix is both halves: the generator stops owning
the whole file, and the lock becomes an equality check that can see a deletion.

**Alternatives considered.** Moving the prose into the README (rejected — it
explains the table and belongs beside it, and the disclosure has to live where the
numbers are). Refusing to overwrite an existing file without `--force` (rejected —
it breaks the documented one-liner and makes the honest path the longer one).
Dropping the "regenerated" claim and hand-maintaining the table (rejected — the
table is exactly what a generator should own). An equality lock with no markers
(rejected — it catches the deletion only after the operator has already destroyed
their working copy).

**Not append-only.** The generator has to be able to *shrink* its region when a
provider's JSON is removed from `results/`, and an append-only rule would
accumulate stale tables. Markers make replacement and preservation the same
operation.

**A note on the other half of #145.** A *measured* latency below half of
`10**-places` rendered as `0.0`, indistinguishable from a genuine zero — and both
of the committed result's query latencies sat in that band, so the published table
read `0.0 | 0.0` for a provider that measured 0.0135 ms and 0.0171 ms while the
README quoted the honest `0.017 ms`. That is recorded as **applying D-010** rather
than as a new decision. D-010's rule is "an unmeasured cell is absent, never
`0.0`", and its rationale argues from the observable: "`0.0` is the best possible
value… a default landing at an extreme of a comparison does not abstain, it
ranks." That argument never depended on how the `0.0` arrived. D-010 closed the
path where it arrives as a default; extending it to the rounding path is the same
decision reaching the rest of its own reason. Flagged here rather than assumed,
because it does widen what D-010 is understood to cover.

---

## D-012 — the chart's labels stop keying on a name #69 proved is not unique

**Date.** 2026-09-25 · **Issue.** #151 · **Reversibility.** cheap

**Decision.** Every surface in `plot.py` that names an embedder goes through
`disambiguated_labels`, which returns a unique name exactly as it is and
suffixes only a *colliding* name with its 1-based position in the sequence.

**Why.** #69 established — and the code still says so in a comment — that two
distinct `SweepResult`s can share an `embedder_name`, because D-007 writes one
file per run and the same provider run twice yields two same-named results. #69
stopped keying the frontier **colour** on that name. The annotation eleven lines
below it still did, and so did `_default_title`.

Measured: two results named `openai-3-small`, one on the frontier and one
dominated. Post-#69 the colours are right and both labels read
`openai-3-small`, while the title said "openai-3-small dominates every other
model" with a dominated `openai-3-small` on the same chart.

**The fix is sparse, and that is the load-bearing difference from
`llm-cost-optimizer` D-021 shipped the same night.** There the set-wide widening
is *uniform*, because a column of numbers at mixed precision reads as mixed
quantities. Here the labels are names: an unsuffixed name is not ambiguous about
anything, and decorating every point would churn every chart this repo has
produced in order to disambiguate two of them. Built the uniform neighbour: four
arms red.

**The ordinal is the sequence position, not a per-name occurrence count, and the
gaps are the point.** `["a", "b", "a"]` labels as `a #1` / `a #3`. "The third
result" maps to the third entry of `sorted(results_dir.glob("*.json"))`; "the
second `a`" is not a file anyone can open. The occurrence-counter neighbour
reads more naturally, which is exactly why it needed an arm — two red.

**What this deliberately is not.** The honest answer to "which run is this
point" is a run id on `SweepResult`. The loader drops the filename that actually
distinguishes the two, and `render_pareto` takes only `Sequence[SweepResult]`.
That is a schema change touching `from_dict`, `to_dict`, every committed result
JSON and D-007, and it is out of scope here. If provenance ever becomes a
first-class field, labelling from it beats an ordinal.

**The title matches the winner by identity**, not by name — the same trap #69
fixed one function down. A name-keyed lookup returns the first result carrying
that name, which need not be the frontier point; the arm puts the loser first so
that version goes red.

**The arms read what the chart was handed**, via a fake `matplotlib` injected
into `sys.modules` with a recording axes. This render path has no CI coverage,
which `_default_title`'s own docstring calls out as "how a caption stating the
opposite of the data survived" — so an `importorskip` arm would skip in exactly
the place that let the last bug through.

**And the population arm needed a scope, not a text match.** Its first version
exempted the helper by matching the text of its lines, and two neighbour probes
that changed only the helper's internals tripped it for no real reason. It now
slices the function out by position. A text-keyed exemption is a wildcard: it
keeps exempting whatever resembles the string it was written for, and stops
exempting the helper the moment the helper is rewritten.

**Alternatives considered.** All built and run except the last.
- *Decorate every label uniformly.* Rejected: 4 red.
- *A per-name occurrence counter.* Rejected: 2 red.
- *Thread the file stem through `render_pareto`.* Rejected for now: more
  informative, but it changes a public signature and the CLI to carry provenance
  that is not part of `SweepResult`'s contract.
- *Add a run id to `SweepResult`.* Deferred: the real fix, and a schema change.
