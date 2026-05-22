"""README ↔ code consistency lock for the hash baseline tokenizer.

Filed under issue #17: the README's Takeaways section previously called the
hash baseline "character-bigrams" while the provider tokenizes on whitespace
and n-grams words. That kind of prose-vs-code drift is exactly the sin the
repo's opening narrative says it won't commit, so this snapshot exists to
prevent it from recurring.

The contract:
- `HashEmbedderProvider().tokenizer` exposes a structured handle for what
  the provider actually does ("word" today; if a character-n-gram variant
  ever lands, it ships as its own provider per D-007, not by mutating this
  one).
- The README's Takeaways section must describe the floor using language
  consistent with that handle — `"word"` ↔ /word.bigram|word n-gram|
  word.overlap/, `"character"` ↔ /character.?bigram|character n-gram|
  character.?overlap/. If anyone re-introduces the wrong wording, this
  test fails before merge.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from emb_shootout.providers.hash_embedder import HashEmbedderProvider

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

# Per-tokenizer wording maps. Keys: tokenizer value the provider exposes.
# Values: (required-patterns, forbidden-patterns). Required must each
# match at least once in the Takeaways section; forbidden must not appear
# anywhere in it.
_WORDING = {
    "word": {
        "required": [
            r"word[-\s]?bigrams?",
            r"word n-gram overlap",
            r"word[-\s]?overlap",
        ],
        "forbidden": [
            r"character[-\s]?bigrams?",
            r"character n-gram overlap",
            r"character[-\s]?overlap",
        ],
    },
}


@pytest.fixture(scope="module")
def takeaways_section() -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index("## Takeaways (so far)")
    end = text.index("##", start + 1)
    return text[start:end]


def test_provider_exposes_tokenizer_handle() -> None:
    # A structured attribute is what the snapshot below indexes into.
    # If anyone removes it, the lock degenerates to regex-matching the
    # source file, which is more fragile.
    provider = HashEmbedderProvider()
    assert provider.tokenizer == "word", (
        f"HashEmbedderProvider().tokenizer should be 'word' by default; "
        f"got {provider.tokenizer!r}. If a new tokenizer was added it should "
        f"ship as a separate provider per D-007."
    )


def test_takeaways_prose_matches_tokenizer(takeaways_section: str) -> None:
    tokenizer = HashEmbedderProvider().tokenizer
    spec = _WORDING.get(tokenizer)
    assert spec is not None, (
        f"No README-wording spec for tokenizer={tokenizer!r}. "
        f"If a new tokenizer was added to HashEmbedderProvider, also add "
        f"its required/forbidden README phrases to _WORDING in this test."
    )

    missing = [p for p in spec["required"] if not re.search(p, takeaways_section, re.IGNORECASE)]
    assert not missing, (
        f"Takeaways section must describe the hash baseline using tokenizer={tokenizer!r} "
        f"language. Missing required pattern(s): {missing}. "
        f"Either fix the README prose or, if the provider's tokenizer changed, "
        f"update both the prose and _WORDING."
    )

    present = [p for p in spec["forbidden"] if re.search(p, takeaways_section, re.IGNORECASE)]
    assert not present, (
        f"Takeaways section contains language inconsistent with tokenizer={tokenizer!r}: "
        f"{present}. The provider does not do character-level n-grams; "
        f"fix the README prose. (This is the exact drift issue #17 closed.)"
    )
