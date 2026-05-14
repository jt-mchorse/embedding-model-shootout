"""Reproducible corpus loader for the embedding-model shootout.

The corpus is one chunk per documented stdlib member (function, class, or
method), generated from the running Python interpreter's introspection.
This means the corpus is **not committed as data** — it's regenerated
from source by anyone who clones the repo (see D-002).

Provenance: every chunk records the module it came from. Python's
standard-library docstrings are licensed under the PSF License, a
permissive, BSD-style license — see ``docs/corpus.md`` for the full
license note.

Public surface:

    DEFAULT_MODULES   : the curated module list used by the benchmark.
    Chunk             : the dataclass each emitted record materializes to.
    build_corpus      : iterator over chunks for a given module list.
    write_jsonl       : convenience JSONL writer (one JSON object per line).
"""

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path

# Curated module list. Chosen for breadth across the stdlib so the corpus
# hits the ≥10k-chunk acceptance bar without scraping anything outside
# the running Python interpreter. Update with care — adding a module
# changes the corpus hash and invalidates downstream benchmark numbers.
DEFAULT_MODULES: tuple[str, ...] = (
    "argparse",
    "array",
    "ast",
    "asyncio",
    "base64",
    "bisect",
    "builtins",
    "calendar",
    "cmath",
    "cmd",
    "codecs",
    "collections",
    "collections.abc",
    "configparser",
    "contextlib",
    "contextvars",
    "copy",
    "csv",
    "ctypes",
    "curses",
    "dataclasses",
    "datetime",
    "decimal",
    "difflib",
    "dis",
    "doctest",
    "email",
    "enum",
    "errno",
    "fileinput",
    "fnmatch",
    "fractions",
    "functools",
    "getpass",
    "gettext",
    "glob",
    "gzip",
    "hashlib",
    "heapq",
    "hmac",
    "html",
    "html.parser",
    "http",
    "http.client",
    "http.cookiejar",
    "http.cookies",
    "http.server",
    "imaplib",
    "inspect",
    "io",
    "ipaddress",
    "itertools",
    "json",
    "keyword",
    "linecache",
    "locale",
    "logging",
    "logging.handlers",
    "lzma",
    "math",
    "mimetypes",
    "multiprocessing",
    "numbers",
    "operator",
    "os",
    "os.path",
    "pathlib",
    "pickle",
    "pkgutil",
    "platform",
    "plistlib",
    "poplib",
    "posixpath",
    "pprint",
    "queue",
    "quopri",
    "random",
    "re",
    "readline",
    "reprlib",
    "secrets",
    "select",
    "selectors",
    "shelve",
    "shlex",
    "shutil",
    "signal",
    "site",
    "smtplib",
    "socket",
    "socketserver",
    "sqlite3",
    "ssl",
    "stat",
    "statistics",
    "string",
    "stringprep",
    "struct",
    "subprocess",
    "symtable",
    "sys",
    "syslog",
    "tarfile",
    "telnetlib",
    "tempfile",
    "termios",
    "textwrap",
    "threading",
    "time",
    "timeit",
    "tokenize",
    "tomllib",
    "trace",
    "traceback",
    "tracemalloc",
    "types",
    "typing",
    "unicodedata",
    "unittest",
    "unittest.mock",
    "urllib",
    "urllib.parse",
    "urllib.request",
    "urllib.error",
    "urllib.response",
    "uuid",
    "warnings",
    "weakref",
    "webbrowser",
    "wsgiref",
    "xml",
    "xml.dom",
    "xml.dom.minidom",
    "xml.etree",
    "xml.etree.ElementTree",
    "xml.parsers",
    "xml.parsers.expat",
    "xml.sax",
    "zipfile",
    "zipimport",
    "zlib",
    "zoneinfo",
)


@dataclass(frozen=True)
class Chunk:
    """One corpus record.

    ``chunk_id`` is the dotted qualname (e.g., ``os.path.join``) and is
    unique within a single corpus build. ``text`` is what the embedder
    sees — signature first, then docstring — so the textual unit matches
    how a developer encounters the symbol in help output.
    """

    chunk_id: str
    text: str
    module: str
    qualname: str
    kind: str  # "module" | "function" | "class" | "method"
    source: str  # "python-stdlib"


def _safe_signature(obj: object) -> str:
    """Best-effort ``str(inspect.signature(obj))``; '' on failure.

    Many C-implemented stdlib callables refuse `inspect.signature`. We
    swallow those and proceed with just the docstring — losing the
    signature doesn't invalidate the chunk.
    """
    try:
        return str(inspect.signature(obj))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""


def _classify(obj: object, parent: object | None) -> str:
    if inspect.ismodule(obj):
        return "module"
    if inspect.isclass(obj):
        return "class"
    if parent is not None and inspect.isclass(parent):
        return "method"
    return "function"


def _chunk_text(name: str, kind: str, obj: object, doc: str) -> str:
    sig = _safe_signature(obj) if kind in ("function", "method", "class") else ""
    header = f"{name}{sig}" if sig else name
    return f"{header}\n\n{doc}".strip()


def _iter_module_members(module: object, module_name: str) -> Iterator[Chunk]:
    # Module-level chunk first, if the module has its own docstring.
    mod_doc = inspect.getdoc(module)
    if mod_doc:
        yield Chunk(
            chunk_id=module_name,
            text=f"{module_name}\n\n{mod_doc}".strip(),
            module=module_name,
            qualname=module_name,
            kind="module",
            source="python-stdlib",
        )

    # Walk the module's documented members. `__all__` is treated as an
    # explicit re-export list — items in it are always emitted under
    # this module's name (e.g., `json.JSONEncoder` even though its
    # `__module__` is `json.encoder`). Items found via `dir(module)`
    # but not in `__all__` are filtered to avoid re-exports of unrelated
    # third-party symbols.
    explicit = set(getattr(module, "__all__", None) or ())
    for name in sorted(explicit or dir(module)):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(module, name)
        except AttributeError:
            continue
        if name not in explicit:
            own_module = getattr(obj, "__module__", None)
            if own_module is not None and own_module != module_name:
                continue

        doc = inspect.getdoc(obj) or ""
        if not doc:
            # No doc → no chunk. Avoid manufacturing content.
            continue
        kind = _classify(obj, parent=None)
        qualname = f"{module_name}.{name}"
        yield Chunk(
            chunk_id=qualname,
            text=_chunk_text(qualname, kind, obj, doc),
            module=module_name,
            qualname=qualname,
            kind=kind,
            source="python-stdlib",
        )

        # If it's a class, walk its members too.
        if kind == "class":
            for child_name, child_obj in sorted(inspect.getmembers(obj)):
                if child_name.startswith("_") and child_name not in ("__init__", "__call__"):
                    continue
                child_doc = inspect.getdoc(child_obj) or ""
                if not child_doc:
                    continue
                child_kind = _classify(child_obj, parent=obj)
                child_qualname = f"{qualname}.{child_name}"
                yield Chunk(
                    chunk_id=child_qualname,
                    text=_chunk_text(child_qualname, child_kind, child_obj, child_doc),
                    module=module_name,
                    qualname=child_qualname,
                    kind=child_kind,
                    source="python-stdlib",
                )


def build_corpus(modules: Iterable[str] | None = None) -> Iterator[Chunk]:
    """Yield chunks for every documented member of ``modules``.

    Modules that fail to import are skipped with no error — the corpus
    contents depend on which optional stdlib pieces this Python build
    includes (e.g., `readline` on Windows). The set of skipped modules
    is reported by the CLI in JSON output for visibility.
    """
    modules_list = list(modules) if modules is not None else list(DEFAULT_MODULES)
    seen_chunk_ids: set[str] = set()
    for mod_name in modules_list:
        try:
            module = importlib.import_module(mod_name)
        except Exception:
            continue
        for chunk in _iter_module_members(module, mod_name):
            if chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.chunk_id)
            yield chunk


def write_jsonl(chunks: Iterable[Chunk], path: PathLike[str] | str) -> int:
    """Write chunks to a JSONL file. Returns the count written.

    Output is deterministic when the input order is deterministic — one
    JSON object per line, no trailing newlines beyond the per-line one,
    sort-stable keys via ``json.dumps(... , sort_keys=True)``.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), sort_keys=True))
            f.write("\n")
            count += 1
    return count
