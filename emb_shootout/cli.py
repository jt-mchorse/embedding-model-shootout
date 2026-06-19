"""CLI:

    emb-shootout corpus build [--out path] [--module M ...]
    emb-shootout corpus validate <path> [--json]
    emb-shootout sweep run --provider P --corpus PATH [--queries N --seed N --output PATH]
    emb-shootout sweep aggregate [--results-dir results] [--out docs/benchmarks.md]
    emb-shootout sweep plot [--results-dir results] [--out-png PATH] [--out-svg PATH]

Kept dep-free where possible (argparse only). The provider adapters are
lazy-imported inside `_cmd_sweep_run` so the CLI loads without any of the
optional extras installed. The plot subcommand lazy-imports matplotlib so
the base CLI keeps loading without the `[plot]` extra installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corpus import DEFAULT_MODULES, build_corpus, write_jsonl
from .io_utils import atomic_write_text


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


def _cmd_corpus_validate(args: argparse.Namespace) -> int:
    """Lint a corpus JSONL; exit 0 clean / 1 findings / 2 I/O error (#45).

    Collecting-mode walk: surfaces every malformed row in one pass instead
    of failing fast like ``_read_corpus_jsonl`` (which is what ``sweep
    run`` invokes). Exit-code shape matches ``eval-harness validate`` in
    llm-eval-harness so consumers can chain validators uniformly. The
    human summary prints one stderr line per finding and a one-line totals
    row to stdout; ``--json`` emits the full ``ValidationReport`` dict.
    """
    from .validate import validate_corpus

    try:
        report = validate_corpus(args.corpus)
    except FileNotFoundError as e:
        sys.stderr.write(f"corpus not found: {e}\n")
        return 2
    except OSError as e:
        sys.stderr.write(f"failed to read {args.corpus}: {e}\n")
        return 2

    if args.as_json:
        rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    else:
        # Findings go to stderr regardless of --out so the operator's diagnostic
        # channel is preserved even when stdout is captured to a file. Parity
        # with llm-eval-harness validate (#66), chunking_lab.validate (#45), and
        # prompt-snap validate (#59).
        for finding in report.findings:
            line_label = f"line {finding.line_no}" if finding.line_no else "file"
            sys.stderr.write(f"{line_label} [{finding.code}]: {finding.reason}\n")
        status = "ok" if report.ok else "fail"
        rendered = (
            f"{status}: {args.corpus} rows={report.n_rows} valid={report.n_valid} "
            f"findings={len(report.findings)}\n"
        )
    if args.out:
        atomic_write_text(args.out, rendered)
    else:
        sys.stdout.write(rendered)
    return 0 if report.ok else 1


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
    atomic_write_text(out_path, json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    sys.stdout.write(
        f"{result.embedder_name}: recall@5={result.recall_at_k.get(5, 0.0):.3f} "
        f"NDCG@10={result.ndcg_at_10:.3f} → {out_path}\n"
    )
    return 0


def _cmd_sweep_aggregate(args: argparse.Namespace) -> int:
    from .sweep import SweepResult, aggregate_json, aggregate_markdown

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        sys.stderr.write(f"{results_dir} is not a directory\n")
        return 2
    files = sorted(results_dir.glob("*.json"))
    if not files:
        sys.stderr.write(f"no result JSON files found under {results_dir}\n")
        return 2
    results = [SweepResult.from_dict(json.loads(p.read_text(encoding="utf-8"))) for p in files]
    if args.format == "json":
        rendered = json.dumps(aggregate_json(results), indent=2, sort_keys=True) + "\n"
    else:
        rendered = aggregate_markdown(results)
    out_path = Path(args.out)
    atomic_write_text(out_path, rendered)
    sys.stdout.write(f"aggregated {len(results)} results → {out_path}\n")
    return 0


def _cmd_sweep_plot(args: argparse.Namespace) -> int:
    from .plot import render_pareto
    from .sweep import SweepResult

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        sys.stderr.write(f"{results_dir} is not a directory\n")
        return 2
    files = sorted(results_dir.glob("*.json"))
    if not files:
        sys.stderr.write(f"no result JSON files found under {results_dir}\n")
        return 2
    results = [SweepResult.from_dict(json.loads(p.read_text(encoding="utf-8"))) for p in files]

    if args.out_png is None and args.out_svg is None:
        sys.stderr.write("must provide at least one of --out-png or --out-svg\n")
        return 2

    try:
        frontier, png_path, svg_path = render_pareto(
            results,
            out_png=args.out_png,
            out_svg=args.out_svg,
            title=args.title,
        )
    except RuntimeError as exc:  # matplotlib missing
        sys.stderr.write(f"{exc}\n")
        return 3

    summary = {
        "results": len(results),
        "frontier_size": len(frontier),
        "frontier": [r.embedder_name for r in frontier],
        "png": str(png_path) if png_path else None,
        "svg": str(svg_path) if svg_path else None,
    }
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
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

    validate = corpus_sub.add_parser(
        "validate",
        help=(
            "Lint a corpus JSONL; report every malformed row in one pass "
            "(exit 0 clean / 1 findings / 2 I/O error)."
        ),
    )
    validate.add_argument("corpus", help="Corpus JSONL path (e.g. data/corpus.jsonl).")
    validate.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit the report as JSON instead of the human-readable summary.",
    )
    validate.add_argument(
        "--out",
        default=None,
        help=(
            "Write the rendered output to this path instead of stdout. Parent dirs "
            "are auto-created via emb_shootout/io_utils.atomic_write_text. Parity "
            "with llm-eval-harness validate --out (#66), chunking-strategies-lab "
            "validate --out (#45), and prompt-snap validate --out (#59). Findings "
            "still print to stderr in human-readable mode even when --out is set, "
            "so the operator's diagnostic channel is preserved."
        ),
    )
    validate.set_defaults(func=_cmd_corpus_validate)

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
    run.add_argument(
        "--out",
        "--output",
        dest="output",
        required=True,
        help=(
            "Output JSON path (e.g. results/openai.json). Accepts both `--out` "
            "(cookbook convention used by `corpus build` and `sweep aggregate`) "
            "and `--output` (the pre-#25 spelling, kept for back-compat)."
        ),
    )
    run.set_defaults(func=_cmd_sweep_run)

    aggregate = sweep_sub.add_parser(
        "aggregate", help="Combine result JSONs into a markdown table or aggregated JSON"
    )
    aggregate.add_argument(
        "--results-dir", default="results", help="Directory of result JSONs (default: %(default)s)"
    )
    aggregate.add_argument(
        "--out",
        default="docs/benchmarks.md",
        help="Output path (default: %(default)s; switch with --format json).",
    )
    aggregate.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format (default: markdown). JSON is for programmatic CI consumers.",
    )
    aggregate.set_defaults(func=_cmd_sweep_aggregate)

    plot = sweep_sub.add_parser("plot", help="Render the cost-vs-recall@5 Pareto plot")
    plot.add_argument(
        "--results-dir", default="results", help="Directory of result JSONs (default: %(default)s)"
    )
    plot.add_argument("--out-png", help="Output PNG path (e.g. docs/pareto.png)")
    plot.add_argument("--out-svg", help="Output SVG path (e.g. docs/pareto.svg)")
    plot.add_argument("--title", help="Override the figure title (auto-chosen by default)")
    plot.set_defaults(func=_cmd_sweep_plot)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
