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
