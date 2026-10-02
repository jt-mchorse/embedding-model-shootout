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
from collections.abc import Mapping, Sequence
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


def require_sequence(name: str, value: object, *, element: str) -> None:
    """Raise ``ValueError`` unless *value* is an ordered sequence that is not itself a string.

    **The check that has to run before the copy, not after it** (#153).

    ``SweepResult.__post_init__`` copies ``notes`` with ``list(...)`` and then
    validates that every element is a ``str``. Both halves are right and the
    order defeats them: ``list("a note")`` does not raise, it splats into one
    entry per character, and every one of those *is* a ``str``, so the validator
    inspects the splatted list and passes. Measured end to end --
    ``run_sweep(..., notes="recall looks low on the small corpus")`` produced 34
    single-character notes, ``to_dict`` wrote all 36 to ``results/*.json``, and
    ``from_dict`` read them back unchanged.

    ``str`` is the whole point of the check, and it is why an element-type loop
    cannot substitute for one: a string is a perfectly good ``Sequence[str]``
    whose elements are all ``str``. ``bytes`` and ``bytearray`` are excluded on
    the same grounds one step further along -- they are sequences whose elements
    are ``int``, so the element loop *would* catch them, but with a message
    about ``notes[0]`` being an ``int`` rather than about the value being bytes.

    **A ``Sequence``, not an ``Iterable``.** A ``set`` is iterable and has no
    order, and ``notes`` is published in the order it was given. A generator is
    iterable and single-use: validating its elements would consume it and leave
    the copy empty, which is the mirror of the bug this function exists for.
    Both are refused with their type named.

    ``tuple`` passes, deliberately. ``run_sweep``'s ``notes`` parameter defaults
    to ``()`` and its siblings ``corpus`` / ``queries`` / ``k_values`` are all
    ``Sequence[...]`` on purpose, so a rule that demanded a ``list`` would
    reject this module's own default.

    **``element`` exists because the population walk found a fourth site the
    issue did not name.** ``validate_k_values`` has the same ``list(...)`` over a
    ``Sequence[int]``, and the two harms there are the same shape one step apart:
    ``k_values="15"`` *was* refused, but by the element loop and with a message
    naming ``['1', '5']`` — a list the caller never passed — while
    ``k_values=b"\x05"`` was **accepted outright**, because ``list(b"\x05")`` is
    ``[5]`` and 5 is a valid k. One rule, two element names, so neither message
    has to be written twice.
    """
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(
            f"{name} must be a sequence of {element}, not a {type(value).__name__}; "
            f"got {value!r}. A str is itself a sequence of str and a bytes is a "
            f"sequence of int, so a coercing copy splats one value into one entry "
            f"per character or per byte instead of raising."
        )
    if not isinstance(value, Sequence):
        raise ValueError(
            f"{name} must be an ordered sequence of {element}; got {type(value).__name__}"
        )


def refuse_bare_string(name: str, value: object) -> None:
    """Raise ``ValueError`` if *value* is a bare ``str``/``bytes`` (#169).

    The other half of ``require_sequence``'s first arm, for the parameters that
    must keep accepting **any** iterable: ``build_corpus(modules)`` and every
    provider's ``embed(texts)``. A generator or a ``set`` is a legitimate
    argument to both, so the ``Sequence`` half of ``require_sequence`` does not
    apply -- only the shape a coercing copy cannot see.

    Measured on ``main`` before this: ``build_corpus("json")`` iterated
    ``"j"``, ``"s"``, ``"o"``, ``"n"``, each failed to import and was skipped *by
    design*, and the corpus came back empty with no error;
    ``HashEmbedderProvider().embed("hello world")`` returned 11 one-character
    vectors, and on the three API providers that is a billed request per
    character batch.

    Called first in every ``embed`` -- before ``self`` is touched -- so the
    refusal can never reach a client or an encoder.
    """
    if isinstance(value, (str, bytes, bytearray)):
        fix = f"pass [{value!r}]" if isinstance(value, str) else "decode it to str first"
        raise ValueError(
            f"{name} must be an iterable of strings, not a bare {type(value).__name__}: "
            f"{value!r} would be split into its characters -- {fix}"
        )


def require_str_sequence(name: str, value: object) -> None:
    """``require_sequence`` for a sequence of strings (``notes``)."""
    require_sequence(name, value, element="strings")


def require_int_sequence(name: str, value: object) -> None:
    """``require_sequence`` for a sequence of ints (``k_values``)."""
    require_sequence(name, value, element="ints")


def require_mapping(name: str, value: object) -> None:
    """Raise ``ValueError`` unless *value* is already a mapping (#153).

    The quieter half of the same shape. ``dict([(5, 1.0)])`` builds a mapping
    from a list of pairs without raising, so ``recall_at_k=[(5, 1.0)]`` and
    ``embed_latency_ms=[("corpus_total", 1.0)]`` were both **accepted**: the copy
    coerced them and the validation loop then found well-formed numeric entries.

    Less harmful than the ``notes`` splat -- the resulting mapping is what such a
    caller probably meant -- but it is the same hole, and the fields are declared
    ``dict[int, float]`` and ``dict[str, float]``. Refused rather than tolerated
    so the rule reads the same at all five copy sites; no caller in this package
    passes pairs.
    """
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{name} must be a mapping; got {type(value).__name__}. A list of "
            f"pairs is coerced by dict(...) without raising, which is why this "
            f"is checked before the copy rather than after it."
        )
