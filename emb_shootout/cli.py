"""Tiny CLI: ``emb-shootout corpus build [--out path] [--module M ...]``.

Kept dependency-free (argparse only) so the project's "no required
runtime deps" promise holds for the corpus layer.
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
