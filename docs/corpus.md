# Corpus

## Choice

**CPython standard-library docstrings.** One chunk per documented stdlib
member (function, class, or method).

## Why this corpus

A good shootout corpus needs three properties: it's
**publicly-licensed** so the repo can ship a reproducer; it's
**semantically dense** so embedding-quality differences show up; and
it's **stable** so a re-run a year from now produces comparable numbers.
Python's stdlib docstrings hit all three.

- **Public license.** PSF License v2 — a permissive, BSD-style license.
  Each chunk records its source (`source: "python-stdlib"`) so
  attribution is preserved.
- **Semantic density.** Each chunk is a real "what does X do?" answer.
  The retrieval task lines up cleanly with how the corpus is used at
  the call site.
- **Stability.** The corpus regenerates from the running Python
  interpreter — no remote fetch, no auth, no rate limit. Pinning a
  Python version pins the corpus.

## License

Python's stdlib is licensed under the [Python Software Foundation
License Version 2][psf-license]. It is permissive, BSD-style, and
explicitly allows derivative works.

[psf-license]: https://docs.python.org/3/license.html

## Provenance

Every chunk records: the `module` it came from, its `qualname`, and a
`source: "python-stdlib"` tag. The CLI's build summary additionally
reports which requested modules failed to import (e.g., `readline` on
Windows), so anyone re-running can see whether their build matches
ours.

## Reproduction

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.'

# Default curated module list, writes data/corpus.jsonl
emb-shootout corpus build --out data/corpus.jsonl

# Or restrict to a single module (repeatable):
emb-shootout corpus build --out /tmp/json-only.jsonl --module json
```

Same Python version + same module list = same corpus, deterministically.
The corpus is **not committed as data** — that's a deliberate decision
([D-002]). The loader is the source of truth.

[D-002]: ../MEMORY/core_decisions_human.md

## Size

A reference build on CPython 3.14 (this repo's CI matrix is 3.11 / 3.12;
the count drifts slightly across versions as the stdlib evolves):

| Metric                          | Value     |
| ------------------------------- | --------- |
| Total chunks                    | **12,010** |
| Modules in the curated list     | 142       |
| Kind: module                    | 181       |
| Kind: class                     | 973       |
| Kind: method                    | 7,649     |
| Kind: function                  | 3,207     |

The benchmark's acceptance bar is **≥10k chunks** (see
`tests/test_corpus.py::test_full_default_modules_clears_10k_chunks` —
that test is the gate; if a future stdlib change drops the corpus below
the threshold, CI fails and we revisit the module list).

## Chunk shape

Each line of `data/corpus.jsonl` is one JSON object:

```json
{
  "chunk_id":   "os.path.join",
  "text":       "os.path.join(a, *p)\n\nJoin two or more path segments...",
  "module":     "os.path",
  "qualname":   "os.path.join",
  "kind":       "function",
  "source":     "python-stdlib"
}
```

`text` is signature-then-docstring so the embedder sees the same surface
a developer encounters via `help(os.path.join)`.

`chunk_id` is the dotted qualname — unique within a single build.
