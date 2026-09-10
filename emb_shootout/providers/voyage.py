"""Voyage embeddings provider.

Lazy-imports `voyageai`; install with `pip install 'emb-shootout[voyage]'`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from .._argcheck import require_non_negative_finite, require_positive_int

DEFAULT_MODEL = "voyage-3"
DEFAULT_DIM = 1024
DEFAULT_COST = 0.06  # voyage-3 list price as of 2026-05


class VoyageProvider:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        cost_per_million_tokens: float = DEFAULT_COST,
        batch_size: int = 32,
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
            import voyageai  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "VoyageProvider requires the optional 'voyage' extra. "
                "Install with: pip install 'emb-shootout[voyage]'"
            ) from e
        self._voyage = voyageai
        self.client = voyageai.Client(api_key=api_key or os.environ.get("VOYAGE_API_KEY"))
        self.model = model
        self.dim = dim
        self.name = f"voyage/{model}"
        self.cost_per_million_tokens = cost_per_million_tokens
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        items = list(texts)
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            response = self.client.embed(batch, model=self.model)
            out.extend(list(v) for v in response.embeddings)
        return out
