"""Constructor-argument rules shared by the providers and the result dataclass.

Two rules, one definition each. Both already existed in several copies:
``batch_size`` in all five real providers, ``dim``/``ngram`` in
``HashEmbedderProvider``, and — for the same two values, at the far end of the
pipeline — in ``SweepResult.__post_init__``.

Why the providers validate at all, given the central check
------------------------------------------------------------

``SweepResult.__post_init__`` says of its own guards:

    Embedder Protocol-implementers also benefit from the centralized check
    without copying the validation per provider.

That is true, and it is why this module exists rather than three checks pasted
into five files. The centralized check gives **correctness**. What it cannot
give is what ``#33`` already argued for ``batch_size``, in a comment every real
provider carries verbatim:

    Validate batch_size before the lazy import so a misconfigured caller gets a
    fast ValueError instead of a slow ImportError-then-network-init (and so the
    check is testable without the optional extra installed; #33).

Both halves of that reason cover ``dim`` and ``cost_per_million_tokens`` exactly
as well, and neither was validated in any provider (#141). Measured on a host
with **no** extras installed::

    provider   batch_size=0                    cost=nan                     dim=-1
    hash       n/a                             n/a                          ValueError
    openai     ValueError: batch_size must...  ImportError (extra missing)  ImportError
    voyage     ValueError: batch_size must...  ImportError (extra missing)  ImportError
    cohere     ValueError: batch_size must...  ImportError (extra missing)  ImportError
    bge        ValueError: batch_size must...  ImportError (extra missing)  ImportError
    nomic      ValueError: batch_size must...  ImportError (extra missing)  ImportError

That table *is* the second reason, run: ``batch_size`` is reachable to a test
without the extra and the other two were not reachable at all.

And the first reason, with the cost attached: with the extra installed, a
``nan`` cost constructed cleanly, the client initialised, and the sweep ran —
``BGEProvider`` downloading ~110MB of weights, the three API providers billing
for every corpus and query embedding — before ``SweepResult.__post_init__``
refused it. The guard was in the right place for correctness and the wrong
place for a misconfigured caller.

The rules here are **derived from the downstream ones**, not from what looks
right. A provider-side rule stricter than ``SweepResult``'s would reject a
sweep the harness would have accepted, which is worse than the gap it closes.
"""

from __future__ import annotations

import math
from typing import Any


def is_positive_int(value: Any) -> bool:
    """True for a real ``int`` >= 1. A ``bool`` is not one.

    Matches ``SweepResult.__post_init__``'s ``embedder_dim`` pair of checks
    (``isinstance(..., int)`` and not ``bool``, then ``>= 1``) and
    ``HashEmbedderProvider``'s ``dim``/``ngram`` rule (#34), which records what
    the sign-only version let through: ``dim=True`` silently bound
    ``self.dim = True`` and ``[0.0] * True`` returned a 1-element vector, while
    ``dim=128.0`` slipped past and raised ``TypeError: can't multiply sequence
    by non-int`` far from the call site.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def is_non_negative_finite(value: Any) -> bool:
    """True for a real, finite number >= 0. A ``bool`` is not one.

    Matches ``SweepResult.__post_init__``'s ``cost_per_million_tokens`` rule
    (#31) arm for arm, including the ``bool`` exclusion, which that comment
    explains at length: ``math.isfinite(True)`` is ``True`` and ``True < 0.0``
    is ``False``, so a boolean cost "was stored as a fabricated $1.0/$0.0 point
    on the Pareto frontier and the committed plot".
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value >= 0.0


def require_positive_int(name: str, value: Any) -> None:
    """Raise ``ValueError`` unless *value* is a positive int.

    The message is the wording all six existing call sites already used, so
    routing them through here changes no assertion.
    """
    if not is_positive_int(value):
        raise ValueError(f"{name} must be a positive integer; got {value!r}")


def require_non_negative_finite(name: str, value: Any) -> None:
    """Raise ``ValueError`` unless *value* is a finite number >= 0."""
    if not is_non_negative_finite(value):
        raise ValueError(f"{name} must be a finite number >= 0.0; got {value!r}")


def is_non_empty_str(value: object) -> bool:
    """True for a non-empty ``str``.

    Shared by ``SweepResult.from_dict`` and ``SweepResult.__post_init__``
    (#143). ``from_dict`` has carried this rule since #94/#95 and its own
    comment names both harms it prevents -- a raw ``AttributeError`` from
    ``aggregate_markdown``'s ``.replace("|", ...)`` and a ``TypeError`` when
    sorting a batch of mixed-type names -- while ``__post_init__`` checked
    ``embedder_name`` not at all. It was the only field in that class with zero
    guards, on a class documented as the validation choke-point for "Embedder
    Protocol-implementers ... without copying the validation per provider",
    whose ``.name`` is exactly an unvalidated string.

    Measured on the unguarded constructor, every value reaching ``to_dict``::

        123 / None / ['a'] / 1.5 / True   aggregate_markdown -> AttributeError
        ''                                table cell rendered EMPTY, silently
        all six                           to_dict wrote a dict from_dict REFUSES

    Predicate rather than raiser, the split this module already uses for
    ``is_non_negative_finite``: both call sites interpolate the offending
    type name into a message their own tests pin, and a shared wording would
    be less specific than either.
    """
    return isinstance(value, str) and bool(value)
