"""Embedder providers.

Each provider implements the `Embedder` Protocol from `emb_shootout.sweep`.
Five non-trivial providers (OpenAI, Voyage, Cohere, BGE, Nomic) are lazy-
imported behind their respective optional extras. The `HashEmbedderProvider`
is dep-free and ships in the base install so CI exercises the sweep flow
hermetically.

Production runs:

    pip install 'emb-shootout[openai]'
    OPENAI_API_KEY=sk-... emb-shootout sweep run --provider openai \\
        --corpus data/corpus.jsonl --output results/openai.json
"""

from __future__ import annotations

from .bge import BGEProvider
from .cohere_provider import CohereProvider
from .hash_embedder import HashEmbedderProvider
from .nomic import NomicProvider
from .openai_provider import OpenAIProvider
from .voyage import VoyageProvider

__all__ = [
    "BGEProvider",
    "CohereProvider",
    "HashEmbedderProvider",
    "NomicProvider",
    "OpenAIProvider",
    "VoyageProvider",
]


PROVIDER_REGISTRY = {
    "hash": HashEmbedderProvider,
    "openai": OpenAIProvider,
    "voyage": VoyageProvider,
    "cohere": CohereProvider,
    "bge": BGEProvider,
    "nomic": NomicProvider,
}
