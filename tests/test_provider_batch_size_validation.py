"""Provider batch_size validation tests (#33).

The five lazy-loaded API providers (Cohere, Voyage, OpenAI, BGE, Nomic)
all had sign-only `batch_size <= 0` checks that accepted `bool` and
`float`. This file pins the tightened `isinstance(int) + reject bool +
positive` contract at the construction boundary.

Validation runs BEFORE the lazy import so these tests can exercise the
contract without the optional extras (`cohere`, `voyageai`, `openai`,
`sentence-transformers`) installed — that's the load-bearing structural
choice of #33.
"""

from __future__ import annotations

import pytest

from emb_shootout.providers.bge import BGEProvider
from emb_shootout.providers.cohere_provider import CohereProvider
from emb_shootout.providers.nomic import NomicProvider
from emb_shootout.providers.openai_provider import OpenAIProvider
from emb_shootout.providers.voyage import VoyageProvider

BAD_NON_POSITIVE = [0, -1, -1000]
BAD_NON_INT = [True, False, 0.5, 1.5, 64.0, "64", None]

PROVIDERS_AND_KWARGS = [
    ("cohere", CohereProvider, {}),
    ("voyage", VoyageProvider, {}),
    ("openai", OpenAIProvider, {}),
    ("bge", BGEProvider, {}),
    ("nomic", NomicProvider, {}),
]


@pytest.mark.parametrize(("label", "cls", "kwargs"), PROVIDERS_AND_KWARGS)
@pytest.mark.parametrize("bad", BAD_NON_POSITIVE)
def test_provider_rejects_non_positive_batch_size(label, cls, kwargs, bad):
    with pytest.raises(ValueError, match=r"batch_size must be a positive integer"):
        cls(batch_size=bad, **kwargs)


@pytest.mark.parametrize(("label", "cls", "kwargs"), PROVIDERS_AND_KWARGS)
@pytest.mark.parametrize("bad", BAD_NON_INT)
def test_provider_rejects_non_int_batch_size(label, cls, kwargs, bad):
    # bool is an int subclass in Python — explicitly rejected because
    # True=1 silently submits one-at-a-time, ballooning provider call
    # counts; float 0.5 raises a confusing TypeError far from the call
    # site once `range(0, len(items), self.batch_size)` runs.
    with pytest.raises(ValueError, match=r"batch_size must be a positive integer"):
        cls(batch_size=bad, **kwargs)  # type: ignore[arg-type]
