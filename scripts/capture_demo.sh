#!/usr/bin/env bash
# Deterministic driver for the 60-second README demo (issue #15).
#
# Runs the three highest-leverage surfaces in sequence on a fresh clone
# with no API key:
#
#   1. corpus build   — `emb-shootout corpus build --module json` writes
#                       a 20-chunk JSONL so the recording is snappy. The
#                       JSON summary line (`chunk_count`, `out`) lands as
#                       the visible artifact of D-002 (reproducible from
#                       source) + D-003 (one-stdlib-member-per-chunk).
#
#   2. sweep run      — `emb-shootout sweep run --provider hash` against
#                       the just-built corpus. Hash baseline is dep-free
#                       (D-004), so this works on a fresh clone with no
#                       extras and no network. recall@5 + NDCG@10 print
#                       to stdout.
#
#   3. aggregate      — `emb-shootout sweep aggregate` merges the single
#                       result JSON into the same markdown table format
#                       that `docs/benchmarks.md` ships (locked by
#                       `tests/test_benchmarks_md_snapshot.py`). Then we
#                       cat the rendered file so the table itself is on
#                       camera.
#
# The output is the recording — when JT records the GIF/video, this
# script's stdout is what gets captured. Hermetic: no API key, no
# network, no committed artifacts touched (everything writes under a
# per-run tempdir).
#
# Why single-module (not full 12,010-chunk) corpus: the README's headline
# recall@5=0.520 figure is the floor on the full curated corpus; the
# capture's job is to show the *pipeline*, not re-publish that number.
# `docs/benchmarks.md` + the "Takeaways" README section already carry
# the full-corpus numbers, locked by snapshot tests. The capture banner
# says so explicitly so a viewer doesn't read trivial-corpus recall as
# the headline.
#
# Variables:
#   CAPTURE_PACE_SECONDS  pause between sections (default 2 for
#                         recording; tests/test_capture_demo_smoke.py
#                         sets this to 0).
#   CAPTURE_DEMO_MODULE   stdlib module passed to `corpus build`. Defaults
#                         to `json`. Override to `os` / `pathlib` / etc.
#                         to vary the recording across takes.
#   CAPTURE_DEMO_QUERIES  query count for the sweep (default 10).
#
# Exit: 0 on full success; non-zero on any sub-step failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACE="${CAPTURE_PACE_SECONDS:-2}"
MODULE="${CAPTURE_DEMO_MODULE:-json}"
QUERIES="${CAPTURE_DEMO_QUERIES:-10}"

banner() {
  printf '\n'
  printf '═══ %s\n' "$1"
  printf '\n'
}

pace() {
  if [ "$PACE" != "0" ]; then
    sleep "$PACE"
  fi
}

cd "$REPO_ROOT"

# Per-run scratch so concurrent recordings (and the smoke test) don't
# collide. Cleaned up on exit including error paths.
TMPDIR_DEMO="$(mktemp -d -t emb-shootout-capture-XXXXXX)"
cleanup() {
  rm -rf "$TMPDIR_DEMO"
}
trap cleanup EXIT INT TERM

CORPUS_PATH="$TMPDIR_DEMO/corpus.jsonl"
RESULTS_DIR="$TMPDIR_DEMO/results"
RESULT_JSON="$RESULTS_DIR/hash.json"
BENCH_MD="$TMPDIR_DEMO/benchmarks.md"
mkdir -p "$RESULTS_DIR"

banner "embedding-model-shootout · 60-second demo"
printf 'three surfaces · hash baseline · no API key, no network\n'
printf 'single-module corpus (CAPTURE_DEMO_MODULE=%s) keeps the recording snappy.\n' "$MODULE"
printf 'full-corpus headline numbers live in docs/benchmarks.md and README "Takeaways".\n'
pace

banner "1/3 · corpus build (D-002 reproducible from source + D-003 one-member-per-chunk)"
printf 'emb-shootout corpus build --module %s --out <tmp>\n\n' "$MODULE"
emb-shootout corpus build --module "$MODULE" --out "$CORPUS_PATH"
pace

banner "2/3 · sweep run · hash baseline (D-004 dep-free Embedder Protocol)"
printf 'emb-shootout sweep run --provider hash --corpus <tmp> --queries %s\n' "$QUERIES"
printf '  queries derived from corpus at sweep time, seed=42 (D-005).\n'
printf '  recall@1/5/10 + NDCG@10 + latency p50/p95 written to <tmp>/results/hash.json.\n\n'
emb-shootout sweep run \
  --provider hash \
  --corpus  "$CORPUS_PATH" \
  --queries "$QUERIES" \
  --output  "$RESULT_JSON"
pace

banner "3/3 · aggregate · markdown table (D-007 one-json-per-provider, aggregator merges)"
printf 'emb-shootout sweep aggregate --results-dir <tmp> --out <tmp>.md\n'
printf '  same format that docs/benchmarks.md ships (locked by test_benchmarks_md_snapshot).\n\n'
emb-shootout sweep aggregate --results-dir "$RESULTS_DIR" --out "$BENCH_MD"
printf '\n─── rendered table ──────────────────────────────────────────────────\n\n'
cat "$BENCH_MD"
pace

banner "done · the four surfaces of the repo are demonstrably wired end-to-end"
printf 'next stop for a real-provider row:\n'
printf '  pip install -e .[openai]\n'
printf '  OPENAI_API_KEY=... emb-shootout sweep run --provider openai \\\n'
printf '    --corpus data/corpus.jsonl --queries 200 --output results/openai.json\n'
printf '  emb-shootout sweep aggregate          # re-renders docs/benchmarks.md\n'
printf '  emb-shootout sweep plot --out-png docs/pareto.png   # needs .[plot]\n'
