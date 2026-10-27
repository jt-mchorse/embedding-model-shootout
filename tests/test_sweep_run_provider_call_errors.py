"""A provider that fails while EMBEDDING is `error:` + exit 2, not a traceback (#264).

#202 translated a failure to *construct* a provider. An OpenAI client does not
check its key until the first request, so a wrong key failed inside
`run_sweep`'s embed call instead. Measured on main 5fb9d89 with `.[openai]`
against a 127.0.0.1 stub: 401 -> 25-line traceback ending
`openai.AuthenticationError`, 429 -> `openai.RateLimitError`, both rc=1.

`sweep run` is the only CLI subcommand that calls a provider, and every
provider in `PROVIDER_REGISTRY` reaches it through the same two call sites:
the corpus `embed` and the per-query `embed_query`-or-`embed`. Both sides are
covered below, for a provider with and without `embed_query`.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from collections.abc import Iterator, Sequence
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from emb_shootout import providers, sweep
from emb_shootout.cli import main
from emb_shootout.sweep import CorpusChunk, Query, run_sweep

N_QUERIES = 5


def _corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    rows = [
        {"chunk_id": f"c{i:03d}", "text": f"document number {i} discusses topic {i % 3}"}
        for i in range(20)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _sweep(tmp_path: Path, provider: str) -> int:
    return main(
        [
            "sweep",
            "run",
            "--provider",
            provider,
            "--corpus",
            str(_corpus(tmp_path)),
            "--queries",
            str(N_QUERIES),
            "--out",
            str(tmp_path / "r.json"),
        ]
    )


class _AuthenticationError(Exception):
    """Stands in for an SDK's own HTTP error (e.g. `openai.AuthenticationError`)."""


class _RateLimitError(Exception):
    """Stands in for `openai.RateLimitError` / a vendor 429."""


def _failing_provider(exc: BaseException, *, fail_on_call: int, has_embed_query: bool):
    """A registry entry whose Nth provider call (1 = the corpus) raises *exc*."""

    class _Fake:
        name = "fake/model"
        dim = 4
        cost_per_million_tokens = 0.0

        def __init__(self) -> None:
            self.calls = 0

        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            self.calls += 1
            if self.calls == fail_on_call:
                raise exc
            return [[1.0, float(len(t)), 0.5, 0.25] for t in texts]

    if has_embed_query:
        _Fake.embed_query = _Fake.embed  # type: ignore[attr-defined]
    return _Fake


_ERRORS = [
    _AuthenticationError("Error code: 401 - Incorrect API key provided"),
    _RateLimitError("Error code: 429 - Rate limit reached"),
    ConnectionResetError(54, "Connection reset by peer"),
    RuntimeError("CUDA out of memory"),
]
_ERROR_IDS = ["auth-401", "rate-limit-429", "network-reset", "local-model-oom"]


@pytest.mark.parametrize("has_embed_query", [False, True], ids=["embed-only", "embed_query"])
@pytest.mark.parametrize(
    ("fail_on_call", "side"),
    [
        (1, "the corpus"),
        (3, f"query 2 of {N_QUERIES} (after the corpus was embedded)"),
    ],
    ids=["corpus-side", "query-side"],
)
@pytest.mark.parametrize("exc", _ERRORS, ids=_ERROR_IDS)
def test_a_provider_that_fails_while_embedding_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    exc: Exception,
    fail_on_call: int,
    side: str,
    has_embed_query: bool,
) -> None:
    monkeypatch.setitem(
        providers.PROVIDER_REGISTRY,
        "fake",
        _failing_provider(exc, fail_on_call=fail_on_call, has_embed_query=has_embed_query),
    )
    assert _sweep(tmp_path, "fake") == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1, err
    assert err.startswith(f"error: embedder 'fake/model' failed embedding {side}: "), err
    assert f"{type(exc).__name__}: {exc}" in err
    assert not (tmp_path / "r.json").exists()


# --- run_sweep: what is and is not translated --------------------------------


def _inputs() -> tuple[list[CorpusChunk], list[Query]]:
    corpus = [CorpusChunk(chunk_id=f"c{i}", text=f"text {i}") for i in range(3)]
    queries = [Query(query_id="q0", text="text 0", expected_chunk_id="c0")]
    return corpus, queries


def test_the_original_error_is_kept_as_the_cause() -> None:
    exc = _RateLimitError("Error code: 429")
    embedder = _failing_provider(exc, fail_on_call=1, has_embed_query=False)()
    corpus, queries = _inputs()
    with pytest.raises(ValueError, match="failed embedding the corpus") as info:
        run_sweep(corpus, queries, embedder=embedder)
    assert info.value.__cause__ is exc


def test_a_providers_own_value_error_passes_through_unchanged() -> None:
    exc = ValueError("input too long for model")
    embedder = _failing_provider(exc, fail_on_call=1, has_embed_query=False)()
    corpus, queries = _inputs()
    with pytest.raises(ValueError, match="^input too long for model$") as info:
        run_sweep(corpus, queries, embedder=embedder)
    assert info.value is exc


def test_an_interrupt_is_not_translated() -> None:
    embedder = _failing_provider(KeyboardInterrupt(), fail_on_call=1, has_embed_query=False)()
    corpus, queries = _inputs()
    with pytest.raises(KeyboardInterrupt):
        run_sweep(corpus, queries, embedder=embedder)


def test_a_bug_outside_the_provider_call_still_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the provider call is wrapped, not `run_sweep` (the issue's constraint)."""

    def _broken(*args: object, **kwargs: object) -> object:
        raise ZeroDivisionError("a scoring bug")

    monkeypatch.setattr(sweep, "retrieve_top_k", _broken)
    embedder = _failing_provider(RuntimeError("unused"), fail_on_call=0, has_embed_query=False)()
    corpus, queries = _inputs()
    with pytest.raises(ZeroDivisionError, match="a scoring bug"):
        run_sweep(corpus, queries, embedder=embedder)


def test_no_provider_call_bypasses_the_translation() -> None:
    """Every provider call in `sweep.py` goes through `_call_embedder`.

    A third seam calling `embedder.embed(...)` (or the bound query method)
    directly would reopen #264 for that seam while the tests above stay green.
    """
    import ast

    tree = ast.parse(Path(sweep.__file__).read_text(encoding="utf-8"))
    query_fn_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_query_embed_fn"
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    direct = []
    wrapped = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Attribute) and func.attr in ("embed", "embed_query")) or (
            isinstance(func, ast.Name) and func.id in query_fn_names
        ):
            direct.append(node.lineno)
        elif isinstance(func, ast.Name) and func.id == "_call_embedder":
            wrapped += 1
    assert direct == [], f"provider called directly at sweep.py lines {direct}"
    assert wrapped >= 2, "expected the corpus and query seams to go through _call_embedder"


# --- the real openai SDK against a local stub --------------------------------


def _stub(statuses: list[int]) -> Iterator[str]:
    """Serve one response per request from *statuses*; 200 returns embeddings."""

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            status = statuses.pop(0) if statuses else statuses_tail
            if status == 200:
                n = len(body["input"])
                payload = {
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "index": i,
                            "embedding": [1.0, i + 1.0] + [0.0] * 1534,
                        }
                        for i in range(n)
                    ],
                    "model": body["model"],
                    "usage": {"prompt_tokens": n, "total_tokens": n},
                }
            else:
                payload = {"error": {"message": f"stub {status}", "type": "x", "code": None}}
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            # The SDK retries a 429; tell it to retry at once.
            self.send_header("retry-after-ms", "1")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args: object) -> None:
            pass

    statuses_tail = statuses[-1]
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


_needs_openai = pytest.mark.skipif(
    importlib.util.find_spec("openai") is None, reason="needs the optional openai extra"
)


@_needs_openai
@pytest.mark.parametrize(
    ("statuses", "side", "error"),
    [
        ([401], "the corpus", "AuthenticationError"),
        ([429], "the corpus", "RateLimitError"),
        ([200, 429], f"query 1 of {N_QUERIES}", "RateLimitError"),
    ],
    ids=["wrong-key", "rate-limited", "rate-limited-after-the-corpus"],
)
def test_the_real_openai_provider_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    statuses: list[int],
    side: str,
    error: str,
) -> None:
    stub = _stub(list(statuses))
    base_url = next(stub)
    try:
        monkeypatch.setenv("OPENAI_BASE_URL", base_url)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-wrong")
        assert _sweep(tmp_path, "openai") == 2
    finally:
        stub.close()
    err = capsys.readouterr().err
    assert err.count("\n") == 1, err
    assert err.startswith(
        f"error: embedder 'openai/text-embedding-3-small' failed embedding {side}"
    ), err
    assert f"{error}: Error code: {statuses[-1]}" in err
    assert not (tmp_path / "r.json").exists()
