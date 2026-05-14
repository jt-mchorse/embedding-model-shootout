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
