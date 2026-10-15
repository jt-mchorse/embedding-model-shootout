"""`run_sweep` refuses a lone surrogate before embedding anything (#212).

#189 applied #176's rule in `cli._read_corpus_jsonl`, the reader `sweep run`
uses, because a paid provider embedded the whole corpus before
`fingerprint_corpus` raised `UnicodeEncodeError`. #210 then made `run_sweep`
the pre-embed choke point for inputs it cannot score, but carried only the
id-set rules there. Measured on `main` with an embedder that escapes the
character the way an HTTP SDK does: a lone surrogate in corpus text, query text
or `query_id` embedded every text (2 corpus + 1 query) before the crash. The
query fields were never checked anywhere.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from emb_shootout import CorpusChunk, run_sweep
from emb_shootout.cli import main
from emb_shootout.providers import HashEmbedderProvider
from emb_shootout.sweep import Query

# Built from codepoints: the high and low ends of the surrogate block.
_HIGH = chr(0xD800)
_LOW = chr(0xDFFF)


class _EscapingEmbedder(HashEmbedderProvider):
    """Counts texts and escapes a surrogate the way an HTTP SDK does, so the
    only thing that can stop the sweep before the fingerprint is the guard."""

    def __init__(self) -> None:
        super().__init__()
        self.texts = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts += len(texts)
        return super().embed(
            [t.encode("utf-8", "surrogatepass").decode("utf-8", "replace") for t in texts]
        )


def _inputs(field: str, bad: str) -> tuple[list[CorpusChunk], list[Query]]:
    corpus = {"a": "alpha beta gamma", "b": "delta epsilon zeta"}
    query = {"query_id": "q1", "text": "alpha beta", "expected_chunk_id": "a"}
    if field == "corpus.chunk_id":
        corpus = {"a": corpus["a"], f"b{bad}": corpus["b"]}
    elif field == "corpus.text":
        corpus["b"] = f"delta {bad} zeta"
    elif field == "queries.query_id":
        query["query_id"] = f"q{bad}1"
    else:
        query["text"] = f"alpha {bad}"
    return [CorpusChunk(k, v) for k, v in corpus.items()], [Query(**query)]


_CASES = [
    ("corpus.chunk_id", r"corpus\[1\]\.chunk_id", 1),
    ("corpus.text", r"corpus\[1\]\.text", 6),
    ("queries.query_id", r"queries\[0\]\.query_id", 1),
    ("queries.text", r"queries\[0\]\.text", 6),
]


@pytest.mark.parametrize("bad", [_HIGH, _LOW], ids=["U+D800", "U+DFFF"])
@pytest.mark.parametrize(("field", "name", "index"), _CASES, ids=[c[0] for c in _CASES])
def test_refused_by_name_before_any_embed(field: str, name: str, index: int, bad: str) -> None:
    corpus, queries = _inputs(field, bad)
    embedder = _EscapingEmbedder()
    cp = f"{ord(bad):04X}"
    with pytest.raises(
        ValueError, match=rf"{name} contains the lone surrogate U\+{cp} at index {index}"
    ):
        run_sweep(corpus, queries, embedder=embedder, k_values=(1,))
    assert embedder.texts == 0, "a paid provider would have been billed before the crash"


def test_clean_inputs_still_sweep() -> None:
    # Control: the escaping embedder and these inputs are fine without a surrogate.
    corpus, queries = _inputs("queries.text", "beta")
    embedder = _EscapingEmbedder()
    result = run_sweep(corpus, queries, embedder=embedder, k_values=(1,))
    assert embedder.texts == 3
    assert result.corpus_fingerprint is not None
    assert result.query_fingerprint is not None


def test_sweep_run_maps_a_surrogate_query_to_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        json.dumps({"chunk_id": "a", "text": "alpha beta gamma"}) + "\n", encoding="utf-8"
    )
    # A BYO query builder -- `build_queries` draws from the guarded corpus.
    monkeypatch.setattr(
        "emb_shootout.queries.build_queries",
        lambda corpus, n, seed: [Query("q1", f"alpha {_HIGH}", "a")],  # noqa: ARG005
    )
    out = tmp_path / "out.json"

    rc = main(
        ["sweep", "run", "--provider", "hash", "--corpus", str(corpus_path), "--output", str(out)]
    )

    err = capsys.readouterr().err
    assert rc == 2
    assert err.startswith("error: queries[0].text contains the lone surrogate U+D800")
    assert not out.exists()
