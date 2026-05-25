"""Nomic embeddings provider via sentence-transformers.

Lazy-imports `sentence_transformers`; install with `pip install 'emb-shootout[sbert]'`.
The model weights download on first use (~550MB for `nomic-embed-text-v1.5`).

`nomic-embed-text-v1.5` requires `trust_remote_code=True` because it
ships custom modeling code. The provider passes the flag explicitly so the
behavior is visible.
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_DIM = 768
DEFAULT_COST = 0.0  # local model — caller's compute, not per-token billed


class NomicProvider:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        cost_per_million_tokens: float = DEFAULT_COST,
        batch_size: int = 32,
        device: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        # Validate before lazy import; see CohereProvider for rationale (#33).
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer; got {batch_size!r}")
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "NomicProvider requires the optional 'sbert' extra. "
                "Install with: pip install 'emb-shootout[sbert]'"
            ) from e
        self._model_cls = SentenceTransformer
        kwargs: dict = {"trust_remote_code": trust_remote_code}
        if device:
            kwargs["device"] = device
        self.encoder = SentenceTransformer(model, **kwargs)
        self.model = model
        self.dim = dim
        self.name = f"nomic/{model.split('/')[-1]}"
        self.cost_per_million_tokens = cost_per_million_tokens
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # Nomic's documentation prescribes a "search_document: " or "search_query: "
        # prefix; for the sweep harness's documents we use the document prefix.
        # Operators evaluating asymmetric search would override this.
        prefixed = [f"search_document: {t}" for t in texts]
        arr = self.encoder.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, row)) for row in arr]
