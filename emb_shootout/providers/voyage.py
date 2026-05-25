"""Voyage embeddings provider.

Lazy-imports `voyageai`; install with `pip install 'emb-shootout[voyage]'`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

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
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer; got {batch_size!r}")
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
