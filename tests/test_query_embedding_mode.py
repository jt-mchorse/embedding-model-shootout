"""Asymmetric providers embed queries in query mode (#186, D-016).

`run_sweep` embedded queries through `embed`, the document path. For the two
providers whose models are trained asymmetrically that is the wrong mode:
Nomic prefixed every query with "search_document: " and Cohere sent
`input_type="search_document"`. Measured by a hunt agent with both SDKs
stubbed and `run_sweep` driven end to end:

    nomic encoded:  [... 'search_document: how do I parse json']
    cohere calls:   [('search_document', [...corpus]), ('search_document', ['how do I parse json'])]

The SDKs are stubbed here the same way: these arms check what the provider
SENDS, which is the whole defect, without weights or a key.
"""

from __future__ import annotations

import sys
import types

import pytest

from emb_shootout.sweep import CorpusChunk, Query, run_sweep

CORPUS = [
    CorpusChunk("c1", "json.loads parses a JSON document"),
    CorpusChunk("c2", "re.match a regex"),
]
QUERIES = [Query("q1", "how do I parse json", "c1")]


def _vec(text: str) -> list[float]:
    return [1.0, 0.0] if "json" in text else [0.0, 1.0]


@pytest.fixture
def nomic(monkeypatch: pytest.MonkeyPatch):
    seen: list[str] = []

    class SentenceTransformer:
        def __init__(self, _model: str, **_kw: object) -> None: ...

        def encode(self, texts: list[str], **_kw: object) -> list[list[float]]:
            seen.extend(texts)
            return [_vec(t) for t in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=SentenceTransformer),
    )
    from emb_shootout.providers.nomic import NomicProvider

    return NomicProvider(dim=2), seen


@pytest.fixture
def cohere(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, list[str]]] = []

    class ClientV2:
        def __init__(self, **_kw: object) -> None: ...

        def embed(
            self, *, model: str, texts: list[str], input_type: str, embedding_types: list[str]
        ):
            calls.append((input_type, list(texts)))
            return types.SimpleNamespace(
                embeddings=types.SimpleNamespace(float=[_vec(t) for t in texts])
            )

    monkeypatch.setitem(sys.modules, "cohere", types.SimpleNamespace(ClientV2=ClientV2))
    from emb_shootout.providers.cohere_provider import CohereProvider

    return CohereProvider(dim=2, api_key="test"), calls


def test_nomic_queries_get_the_query_prefix_and_documents_keep_theirs(nomic) -> None:
    provider, seen = nomic
    run_sweep(CORPUS, QUERIES, embedder=provider, k_values=(1,))
    assert "search_query: how do I parse json" in seen
    assert "search_document: how do I parse json" not in seen
    assert "search_document: json.loads parses a JSON document" in seen


def test_cohere_queries_use_search_query(cohere) -> None:
    provider, calls = cohere
    run_sweep(CORPUS, QUERIES, embedder=provider, k_values=(1,))
    assert ("search_query", ["how do I parse json"]) in calls
    assert all(t == "search_document" for t, texts in calls if texts != ["how do I parse json"])


def test_cohere_query_input_type_is_configurable(cohere, monkeypatch: pytest.MonkeyPatch) -> None:
    from emb_shootout.providers.cohere_provider import CohereProvider

    _provider, calls = cohere
    custom = CohereProvider(dim=2, api_key="test", query_input_type="classification")
    custom.embed_query(["x"])
    assert calls[-1] == ("classification", ["x"])


def test_embed_query_refuses_a_bare_string_like_embed(nomic, cohere) -> None:
    for provider in (nomic[0], cohere[0]):
        with pytest.raises((TypeError, ValueError)):
            provider.embed_query("how do I parse json")


def test_a_provider_without_embed_query_still_embeds_queries_with_embed() -> None:
    # D-004's single required method is unchanged: the hash baseline has no
    # embed_query, and the sweep falls back to `embed`.
    from emb_shootout.providers.hash_embedder import HashEmbedderProvider

    h = HashEmbedderProvider()
    assert not hasattr(h, "embed_query")
    result = run_sweep(CORPUS, QUERIES, embedder=h, k_values=(1,))
    assert result is not None
