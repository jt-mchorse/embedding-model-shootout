"""CLI:

    emb-shootout corpus build [--out path] [--module M ...]
    emb-shootout sweep run --provider P --corpus PATH [--queries N --seed N --output PATH]
    emb-shootout sweep aggregate [--results-dir results] [--out docs/benchmarks.md]

Kept dep-free where possible (argparse only). The provider adapters are
lazy-imported inside `_cmd_sweep_run` so the CLI loads without any of the
optional extras installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corpus import DEFAULT_MODULES, build_corpus, write_jsonl


def _cmd_corpus_build(args: argparse.Namespace) -> int:
    modules = args.module or DEFAULT_MODULES
    out_path = Path(args.out)
    chunks = list(build_corpus(modules))
    count = write_jsonl(chunks, out_path)
    summary = {
        "out": str(out_path),
        "chunk_count": count,
        "modules_requested": len(modules),
    }
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0


def _cmd_sweep_run(args: argparse.Namespace) -> int:
    # Lazy-import inside the command so loading the CLI doesn't pull in
    # provider modules whose optional deps may not be installed.
    from .providers import PROVIDER_REGISTRY
    from .queries import build_queries
    from .sweep import run_sweep

    if args.provider not in PROVIDER_REGISTRY:
        sys.stderr.write(
            f"unknown provider {args.provider!r}; choices: {sorted(PROVIDER_REGISTRY)}\n"
        )
        return 2

    corpus = _read_corpus_jsonl(Path(args.corpus))
    queries = build_queries(corpus, n=args.queries, seed=args.seed)
    embedder = PROVIDER_REGISTRY[args.provider]()
    result = run_sweep(corpus, queries, embedder=embedder, k_values=(1, 5, 10))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sys.stdout.write(
        f"{result.embedder_name}: recall@5={result.recall_at_k.get(5, 0.0):.3f} "
        f"NDCG@10={result.ndcg_at_10:.3f} → {out_path}\n"
    )
    return 0


def _cmd_sweep_aggregate(args: argparse.Namespace) -> int:
    from .sweep import SweepResult, aggregate_markdown

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        sys.stderr.write(f"{results_dir} is not a directory\n")
        return 2
    files = sorted(results_dir.glob("*.json"))
    if not files:
        sys.stderr.write(f"no result JSON files found under {results_dir}\n")
        return 2
    results = [SweepResult.from_dict(json.loads(p.read_text(encoding="utf-8"))) for p in files]
    md = aggregate_markdown(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    sys.stdout.write(f"aggregated {len(results)} results → {out_path}\n")
    return 0


def _read_corpus_jsonl(path: Path) -> list:
    """Read a corpus JSONL produced by `corpus build` and adapt to CorpusChunk."""
    from .sweep import CorpusChunk

    chunks: list[CorpusChunk] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunks.append(CorpusChunk(chunk_id=obj["chunk_id"], text=obj["text"]))
    return chunks


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="emb-shootout")
    sub = parser.add_subparsers(dest="cmd", required=True)

    corpus = sub.add_parser("corpus", help="Corpus management")
    corpus_sub = corpus.add_subparsers(dest="corpus_cmd", required=True)
    build = corpus_sub.add_parser("build", help="Build the stdlib corpus to JSONL")
    build.add_argument(
        "--out", default="data/corpus.jsonl", help="Output JSONL path (default: %(default)s)"
    )
    build.add_argument(
        "--module",
        action="append",
        help="Restrict to a single module (repeatable). Defaults to the curated list.",
    )
    build.set_defaults(func=_cmd_corpus_build)

    sweep = sub.add_parser("sweep", help="Embedder sweep")
    sweep_sub = sweep.add_subparsers(dest="sweep_cmd", required=True)

    run = sweep_sub.add_parser("run", help="Run the sweep against one provider")
    run.add_argument(
        "--provider",
        required=True,
        help="Provider name; one of hash | openai | voyage | cohere | bge | nomic",
    )
    run.add_argument("--corpus", required=True, help="Corpus JSONL path (from `corpus build`)")
    run.add_argument("--queries", type=int, default=200, help="Query count (default: %(default)s)")
    run.add_argument(
        "--seed", type=int, default=42, help="Query-construction seed (default: %(default)s)"
    )
    run.add_argument("--output", required=True, help="Output JSON path (e.g. results/openai.json)")
    run.set_defaults(func=_cmd_sweep_run)

    aggregate = sweep_sub.add_parser("aggregate", help="Combine result JSONs into a markdown table")
    aggregate.add_argument(
        "--results-dir", default="results", help="Directory of result JSONs (default: %(default)s)"
    )
    aggregate.add_argument(
        "--out", default="docs/benchmarks.md", help="Output markdown path (default: %(default)s)"
    )
    aggregate.set_defaults(func=_cmd_sweep_aggregate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
