"""OpenAI text-embedding-3 provider.

Lazy-imports the `openai` SDK; install with `pip install 'emb-shootout[openai]'`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from .._argcheck import require_non_negative_finite, require_positive_int

# Public list price as of 2026-05; pass `cost_per_million_tokens` to override.
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIM = 1536
DEFAULT_COST = 0.02  # text-embedding-3-small: $0.02 / 1M tokens


class OpenAIProvider:
    """OpenAI embeddings via the official SDK."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        cost_per_million_tokens: float = DEFAULT_COST,
        batch_size: int = 64,
        api_key: str | None = None,
    ) -> None:
        # Validate before lazy import; see CohereProvider for rationale (#33).
        # Through the shared rule since #141 -- the message is unchanged; what
        # changed is that `dim` and `cost_per_million_tokens` are now held to
        # the same standard, for the same two reasons stated above, which cover
        # them exactly as well as they cover `batch_size`.
        require_positive_int("batch_size", batch_size)
        require_positive_int("dim", dim)
        require_non_negative_finite("cost_per_million_tokens", cost_per_million_tokens)
        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "OpenAIProvider requires the optional 'openai' extra. "
                "Install with: pip install 'emb-shootout[openai]'"
            ) from e
        self._openai = openai
        self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.dim = dim
        self.name = f"openai/{model}"
        self.cost_per_million_tokens = cost_per_million_tokens
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        items = list(texts)
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            # SDK returns objects with `.embedding`; sort by `.index` to
            # preserve input order (the API guarantees but we verify).
            sorted_data = sorted(response.data, key=lambda d: d.index)
            for d in sorted_data:
                out.append(list(d.embedding))
        return out
