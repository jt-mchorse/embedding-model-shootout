"""Architecture-doc lock: catch drift between `docs/architecture.md` and
the actual shipped surface of the repo.

Parallel to the existing `tests/test_readme_snapshot.py` (which locks the
README's Takeaways numbers to `results/hash.json`) and to the cookbook /
nextjs / ai-app architecture-doc checkers landed earlier this session.

Four invariants pinned:

1. **Path-token reachability.** Every `emb_shootout/<module>`,
   `results/<file>`, `notebooks/<file>`, `scripts/<file>`, `data/<file>`,
   and `tests/<file>` token in the doc resolves on disk. A future rename
   that doesn't update the doc fails fast.

2. **Closed-feature-issue coverage.** Every closed-feature-issue number
   in `KNOWN_SHIPPED_ISSUES` is referenced at least once in the doc.
   So a future seventh shipped surface can't slip past the doc, and
   reverting the doc to its pre-#19 "everything pending except #1"
   state fires this assertion with the missing issues named.

3. **Active-decision coverage.** Every non-superseded `D-NNN` in
   `MEMORY/core_decisions_ai.md` whose numeric id is
   `>= MIN_ACTIVE_DECISION_ID` is referenced at least once. The next
   `D-NNN` landing without a doc update fails this test loud.

4. **Banned-phrase absence.** Phrases that characterized the pre-#19
   drift are absent (case-insensitive). The phrases are pinned as a
   tuple so a future loose edit of the test that drops one
   fails the explicit hard-pin test below.

Tamper-verified by reintroducing each banned phrase on a scratch copy
and by stubbing out a shipped-issue reference; each fires the relevant
assertion with the specific drift quoted in the error message.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC = REPO_ROOT / "docs" / "architecture.md"
DECISIONS = REPO_ROOT / "MEMORY" / "core_decisions_ai.md"

# D-001 is the scope baseline (handoff §2) and isn't tied to a shipped
# code surface; it doesn't need to be cited in architecture.md. Every
# active D-NNN with id >= MIN_ACTIVE_DECISION_ID does.
MIN_ACTIVE_DECISION_ID = 2

# Closed feature issues whose work the architecture doc should
# enumerate. Issue numbers come from `gh issue list --state closed --json
# number,title`. Each represents a shipped surface that has a code or
# artifact home in the repo and therefore belongs in the doc.
#
# Intentionally excluded from the coverage check (but still allowed to
# appear in the doc):
#   - #4   README takeaways prose — narrative, not architecture
#   - #16  GIF/video capture     — operator-supplied artifact only
#   - #17  README phrasing fix   — locked separately by tests/test_hash_baseline_description.py
KNOWN_SHIPPED_ISSUES = (1, 2, 3, 5, 11, 13, 15, 45)

# Drift shapes specific to issue #19's pre-fix state. Lowercase
# substring match (case-insensitive). Pinned in a tuple so a future
# loose edit can't silently drop one — see
# `test_banned_phrases_hard_pin_set` below.
BANNED_PHRASES = (
    "this pr",  # "Shipped (this PR — issue #1)"
    "pending issue",  # nextjs / ai-app / cookbook drift shape
    "*(unfiled)*",  # nextjs drift shape
    "to-be-filed",  # nextjs drift shape
)

# Path-token prefixes that must resolve on disk if quoted in the doc.
# Backtick-quoted tokens only — prose mentions are deliberately
# excluded (matching prose would force every mention of a module name
# to be a path token, which it isn't).
RESOLVABLE_PREFIXES = (
    "emb_shootout/",
    "results/",
    "notebooks/",
    "scripts/",
    "data/",
    "tests/",
    "docs/",
)


@pytest.fixture(scope="module")
def doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def active_decisions() -> tuple[int, ...]:
    """Parse `MEMORY/core_decisions_ai.md` for non-superseded `D-NNN`
    entries whose numeric id is `>= MIN_ACTIVE_DECISION_ID`.
    """
    text = DECISIONS.read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=- id:)", text)
    active: list[int] = []
    for block in blocks:
        id_match = re.search(r"- id:\s*D-(\d+)", block)
        if not id_match:
            continue
        sup_match = re.search(r"superseded_by:\s*(\S+)", block)
        is_active = (sup_match is None) or (sup_match.group(1).strip().lower() == "null")
        if is_active:
            n = int(id_match.group(1))
            if n >= MIN_ACTIVE_DECISION_ID:
                active.append(n)
    return tuple(sorted(active))


def _extract_backtick_paths(text: str) -> set[str]:
    """Collect every backtick-quoted token that starts with one of the
    RESOLVABLE_PREFIXES. Mermaid-quoted strings (inside `[...]:` brackets)
    are out of scope — those carry HTML-like content for the diagram and
    aren't paths a reader would copy-paste.
    """
    found: set[str] = set()
    # Single-backtick spans. Multi-line code fences are excluded by the
    # newline boundary in the character class.
    for match in re.finditer(r"`([^`\n]+)`", text):
        token = match.group(1).strip()
        for prefix in RESOLVABLE_PREFIXES:
            if token.startswith(prefix):
                # Drop trailing punctuation that a reader wouldn't
                # copy as part of the path. e.g. "`results/hash.json`,"
                # → "results/hash.json".
                while token and token[-1] in ".,;:":
                    token = token[:-1]
                # Drop a trailing `()` from "foo.bar()" — common when
                # the prose names a function as a path.
                token = re.sub(r"\(\)$", "", token)
                if token:
                    found.add(token)
                break
    return found


def _resolves_on_disk(token: str) -> bool:
    """Resolve a path-or-dotted token against the repo root.

    Dotted-form modules like `emb_shootout.sweep.run_sweep` are accepted
    if the file path (with the last component dropped if it isn't a
    submodule) resolves — but that's outside the prefix list above, so
    in practice we only see slash-paths here.
    """
    p = REPO_ROOT / token
    return p.exists()


def test_doc_exists() -> None:
    assert DOC.exists(), f"missing {DOC}"


def test_decisions_file_exists() -> None:
    assert DECISIONS.exists(), f"missing {DECISIONS}"


def test_backtick_paths_resolve_on_disk(doc_text: str) -> None:
    tokens = _extract_backtick_paths(doc_text)
    unresolved = sorted(t for t in tokens if not _resolves_on_disk(t))
    assert not unresolved, (
        "docs/architecture.md quotes paths that don't exist on disk:\n"
        + "\n".join(f"  - `{t}`" for t in unresolved)
        + "\n(regenerate the doc to match the current layout, or fix the typo)"
    )


def test_every_shipped_issue_referenced(doc_text: str) -> None:
    # Match `#NN` tokens — the pattern the doc uses to annotate each
    # surface with its origin issue. The whole-word boundary keeps
    # `#NN` from matching e.g. `#NNN` (a wider issue number) by accident.
    referenced = {
        int(m.group(1))
        for m in re.finditer(r"\(#(\d+)\)|#(\d+)\b", doc_text)
        for m in [re.match(r"\D*(\d+)", m.group(0))]
        if m
    }
    missing = sorted(set(KNOWN_SHIPPED_ISSUES) - referenced)
    assert not missing, (
        "docs/architecture.md doesn't reference these closed-feature-issues "
        "even once:\n"
        + "\n".join(f"  - #{n}" for n in missing)
        + "\n(every shipped surface should have its origin issue annotated "
        "in the doc; add a `(#NN)` to the relevant component bullet or diagram node)"
    )


def test_every_active_decision_referenced(doc_text: str, active_decisions: tuple[int, ...]) -> None:
    referenced = {int(m.group(1)) for m in re.finditer(r"\bD-0*(\d+)\b", doc_text)}
    missing = sorted(set(active_decisions) - referenced)
    assert not missing, (
        "docs/architecture.md doesn't reference these active "
        "(non-superseded) core decisions even once:\n"
        + "\n".join(f"  - D-{n:03d}" for n in missing)
        + "\n(every shipped layer / posture in MEMORY/core_decisions_ai.md "
        "should be annotated in the doc where the relevant code lives; "
        "add a `D-NNN` reference to the relevant bullet)"
    )


def test_no_banned_phrases(doc_text: str) -> None:
    lowered = doc_text.lower()
    hits = [p for p in BANNED_PHRASES if p in lowered]
    assert not hits, (
        "docs/architecture.md contains pre-#19 drift phrases:\n"
        + "\n".join(f"  - {p!r}" for p in hits)
        + "\n(these phrases described the pre-shipping state; the doc is "
        "now a steady-state reference, not a PR description)"
    )


def test_banned_phrases_hard_pin_set() -> None:
    # Hard-pin BANNED_PHRASES to its expected exact contents so a future
    # loose edit of the test that drops one phrase can't silently weaken
    # the drift guard.
    assert BANNED_PHRASES == (
        "this pr",
        "pending issue",
        "*(unfiled)*",
        "to-be-filed",
    )


def test_known_shipped_issues_hard_pin_set() -> None:
    # Same belt-and-braces for the issue coverage list.
    assert KNOWN_SHIPPED_ISSUES == (1, 2, 3, 5, 11, 13, 15, 45)


def test_resolvable_prefixes_hard_pin_set() -> None:
    # And for the path-prefix list.
    assert RESOLVABLE_PREFIXES == (
        "emb_shootout/",
        "results/",
        "notebooks/",
        "scripts/",
        "data/",
        "tests/",
        "docs/",
    )


def test_min_active_decision_id_hard_pin() -> None:
    assert MIN_ACTIVE_DECISION_ID == 2
