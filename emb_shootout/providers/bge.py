"""BGE (BAAI General Embedding) provider via sentence-transformers.

Lazy-imports `sentence_transformers`; install with `pip install 'emb-shootout[sbert]'`.
The model weights download on first use (~110MB for `bge-small-en-v1.5`).
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIM = 384
DEFAULT_COST = 0.0  # local model — caller's compute, not per-token billed


class BGEProvider:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        cost_per_million_tokens: float = DEFAULT_COST,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        # Validate before lazy import; see CohereProvider for rationale (#33).
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer; got {batch_size!r}")
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "BGEProvider requires the optional 'sbert' extra. "
                "Install with: pip install 'emb-shootout[sbert]'"
            ) from e
        self._model_cls = SentenceTransformer
        self.encoder = (
            SentenceTransformer(model, device=device) if device else SentenceTransformer(model)
        )
        self.model = model
        self.dim = dim
        self.name = f"bge/{model.split('/')[-1]}"
        self.cost_per_million_tokens = cost_per_million_tokens
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        # SentenceTransformer.encode returns numpy by default; convert to list.
        arr = self.encoder.encode(
            items, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, row)) for row in arr]
