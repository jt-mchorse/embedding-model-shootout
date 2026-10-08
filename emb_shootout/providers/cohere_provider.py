"""Cohere embeddings provider.

Lazy-imports `cohere`; install with `pip install -e '.[cohere]'`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from .._argcheck import refuse_bare_string, require_non_negative_finite, require_positive_int

DEFAULT_MODEL = "embed-english-v3.0"
DEFAULT_DIM = 1024
DEFAULT_COST = 0.10  # embed-english-v3.0 list price as of 2026-05


class CohereProvider:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        cost_per_million_tokens: float = DEFAULT_COST,
        batch_size: int = 96,
        api_key: str | None = None,
        input_type: str = "search_document",
        query_input_type: str = "search_query",
    ) -> None:
        # Validate batch_size before the lazy import so a misconfigured caller
        # gets a fast ValueError instead of a slow ImportError-then-network-init
        # (and so the check is testable without the optional `cohere` extra
        # installed; #33).
        # Through the shared rule since #141 -- the message is unchanged; what
        # changed is that `dim` and `cost_per_million_tokens` are now held to
        # the same standard, for the same two reasons stated above, which cover
        # them exactly as well as they cover `batch_size`.
        require_positive_int("batch_size", batch_size)
        require_positive_int("dim", dim)
        require_non_negative_finite("cost_per_million_tokens", cost_per_million_tokens)
        try:
            import cohere  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "CohereProvider requires the optional 'cohere' extra. "
                "Install with: pip install -e '.[cohere]'"
            ) from e
        self._cohere = cohere
        self.client = cohere.ClientV2(api_key=api_key or os.environ.get("COHERE_API_KEY"))
        self.model = model
        self.dim = dim
        self.name = f"cohere/{model}"
        self.cost_per_million_tokens = cost_per_million_tokens
        self.batch_size = batch_size
        self.input_type = input_type
        # Cohere's v3 models are asymmetric: documents and queries take
        # different `input_type`s (#186, D-016). Queries used `input_type` too.
        self.query_input_type = query_input_type

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed DOCUMENTS (`input_type`, default `search_document`)."""
        refuse_bare_string("texts", texts)  # #169: before any client/encoder use
        return self._embed(texts, self.input_type)

    def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed QUERIES (`query_input_type`, default `search_query`; #186)."""
        refuse_bare_string("texts", texts)
        return self._embed(texts, self.query_input_type)

    def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        items = list(texts)
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            response = self.client.embed(
                model=self.model,
                texts=batch,
                input_type=input_type,
                embedding_types=["float"],
            )
            for vec in response.embeddings.float:
                out.append(list(vec))
        return out
