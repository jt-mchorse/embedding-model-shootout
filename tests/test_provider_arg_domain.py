"""Every provider validates `dim` and `cost` before its lazy import (#141).

All five real providers carried this comment verbatim:

    Validate batch_size before the lazy import so a misconfigured caller gets a
    fast ValueError instead of a slow ImportError-then-network-init (and so the
    check is testable without the optional extra installed; #33).

Both halves of that reason cover `dim` and `cost_per_million_tokens` exactly as
well, and neither was validated in any of the five. `HashEmbedderProvider` —
the dep-free reference in the same package — validated its `dim` (#34), so the
provider nobody pays for was stricter than the ones an operator does.

Measured on a host with **no** extras installed::

    provider   batch_size=0                    cost=nan                     dim=-1
    hash       n/a                             n/a                          ValueError
    openai     ValueError: batch_size must...  ImportError (extra missing)  ImportError
    voyage     ValueError: batch_size must...  ImportError (extra missing)  ImportError
    cohere     ValueError: batch_size must...  ImportError (extra missing)  ImportError
    bge        ValueError: batch_size must...  ImportError (extra missing)  ImportError
    nomic      ValueError: batch_size must...  ImportError (extra missing)  ImportError

**This file is that table.** It runs with no extra installed, which is the
whole of #33's second reason and the thing the hermetic suite could not express
before: an `ImportError` row is a misconfiguration no test can reach.

And #33's first reason, with the cost attached: with the extra installed, a
`nan` cost constructed cleanly, the client initialised and the sweep ran —
`BGEProvider` downloading ~110MB of weights, the three API providers billing for
every corpus and query embedding — before `SweepResult.__post_init__` refused
it. That guard is right and stays; this moves the *moment*, not the contract,
and `test_the_downstream_guards_still_fire` pins that.

The counter-argument is in the code and is answered rather than ignored.
`SweepResult.__post_init__` says its central check means implementers benefit
"without copying the validation per provider" — which is why `_argcheck.py`
exists and why this change *removes* duplication: the `batch_size` rule had six
copies before it.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from emb_shootout._argcheck import (
    is_non_negative_finite,
    is_positive_int,
    require_non_negative_finite,
    require_positive_int,
)
from emb_shootout.providers import PROVIDER_REGISTRY
from emb_shootout.sweep import SweepResult

#: The five providers behind an optional extra. `hash` is dep-free and takes
#: neither `batch_size` nor `cost_per_million_tokens`, so it has its own rows.
LAZY_PROVIDERS = tuple(n for n in PROVIDER_REGISTRY if n != "hash")


def _construct(name: str, **kwargs: Any) -> None:
    PROVIDER_REGISTRY[name](**kwargs)


def test_the_registry_has_the_five_lazy_providers_and_the_reference() -> None:
    """Anti-vacuous, and discovered from the registry rather than listed: a
    seventh provider joins every row below without editing this file.
    """
    assert set(PROVIDER_REGISTRY) >= {"hash", "openai", "voyage", "cohere", "bge", "nomic"}
    assert len(LAZY_PROVIDERS) >= 5
    assert "hash" not in LAZY_PROVIDERS


# --- the table, and it must run with no extra installed --------------------

BAD_DIMS: tuple[Any, ...] = (0, -1, 1.5, 128.0, True, False, None, "128")
BAD_COSTS: tuple[Any, ...] = (
    -0.1,
    float("nan"),
    float("inf"),
    float("-inf"),
    True,
    False,
    None,
    "0.02",
)
BAD_BATCH_SIZES: tuple[Any, ...] = (0, -1, 1.5, True, False, None, "32")


@pytest.mark.parametrize("provider", LAZY_PROVIDERS)
@pytest.mark.parametrize("dim", BAD_DIMS, ids=repr)
def test_a_bad_dim_is_refused_before_the_lazy_import(provider: str, dim: Any) -> None:
    """`ValueError`, not `ImportError`. On a host without the extra this was an
    `ImportError` — the misconfiguration was unreachable.
    """
    with pytest.raises(ValueError, match="dim must be a positive integer"):
        _construct(provider, dim=dim)


@pytest.mark.parametrize("provider", LAZY_PROVIDERS)
@pytest.mark.parametrize("cost", BAD_COSTS, ids=repr)
def test_a_bad_cost_is_refused_before_the_lazy_import(provider: str, cost: Any) -> None:
    with pytest.raises(ValueError, match="cost_per_million_tokens must be a finite number"):
        _construct(provider, cost_per_million_tokens=cost)


@pytest.mark.parametrize("provider", LAZY_PROVIDERS)
@pytest.mark.parametrize("batch_size", BAD_BATCH_SIZES, ids=repr)
def test_a_bad_batch_size_is_still_refused(provider: str, batch_size: Any) -> None:
    """The row that already worked, kept as the control. Without it the two
    tables above are satisfied by a provider that refuses everything.
    """
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        _construct(provider, batch_size=batch_size)


@pytest.mark.parametrize("provider", LAZY_PROVIDERS)
def test_a_well_formed_call_still_reaches_the_lazy_import(provider: str) -> None:
    """The other control, and the one that proves these are *pre-import* checks
    rather than a blanket refusal.

    With the extra absent the well-formed call must get all the way to the
    `ImportError` — the same outcome it had before this change. If it raised
    `ValueError` instead, the new rules would be rejecting a legitimate
    configuration.
    """
    with pytest.raises(ImportError, match="extra"):
        _construct(provider, dim=8, cost_per_million_tokens=0.5, batch_size=4)


def test_the_reference_provider_is_not_stricter_than_the_paid_ones() -> None:
    """`HashEmbedderProvider` had the `dim` rule the five real providers lacked.
    It keeps it, through the same definition.
    """
    with pytest.raises(ValueError, match="dim must be a positive integer"):
        PROVIDER_REGISTRY["hash"](dim=0)
    with pytest.raises(ValueError, match="ngram must be a positive integer"):
        PROVIDER_REGISTRY["hash"](ngram=True)
    assert PROVIDER_REGISTRY["hash"](dim=8, ngram=1).dim == 8


# --- the provider rule must not be stricter than the downstream one --------


@pytest.mark.parametrize(
    "value", [1, 2, 128, 10**6, 0, -1, 1.5, 128.0, True, False, None, "128"], ids=repr
)
def test_is_positive_int_is_exactly_sweepresults_embedder_dim_domain(value: Any) -> None:
    """Derived, not restated.

    `SweepResult.__post_init__` spells the rule as two arms with two messages
    ("must be an int", "must be >= 1") because they diagnose different mistakes.
    `is_positive_int` is their conjunction, and this row is what stops the
    provider-side rule drifting *stricter* than the harness's — which would
    refuse a sweep `SweepResult` would have accepted, worse than the gap #141
    closed.
    """
    accepted_downstream = (
        isinstance(value, int) and not isinstance(value, bool) and value >= 1  # noqa: E721
    )
    assert is_positive_int(value) is accepted_downstream


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        0,
        0.02,
        1,
        1e9,
        -0.1,
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        False,
        None,
        "x",
    ],
    ids=repr,
)
def test_is_non_negative_finite_is_exactly_sweepresults_cost_domain(value: Any) -> None:
    accepted_downstream = (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0.0
    )
    assert is_non_negative_finite(value) is accepted_downstream


@pytest.mark.parametrize("dim", BAD_DIMS, ids=repr)
def test_the_downstream_guards_still_fire(dim: Any) -> None:
    """This change moves the moment, not the contract.

    `SweepResult` must keep refusing exactly what it refused before — a
    provider-side check is an addition, and a reader who deletes it must not
    thereby open the far end.
    """
    with pytest.raises(ValueError, match="embedder_dim"):
        SweepResult(
            embedder_name="x",
            embedder_dim=dim,
            cost_per_million_tokens=0.0,
            n_corpus=1,
            n_queries=1,
            ndcg_at_10=0.5,
            recall_at_k={1: 1.0},
            embed_latency_ms={"corpus": 1.0},
        )


# --- one definition, not eleven -------------------------------------------


def test_the_rules_live_in_one_place() -> None:
    """The `batch_size` rule had SIX copies before this (five providers plus
    `HashEmbedderProvider`'s `dim`/`ngram` pair), and the counter-argument in
    `SweepResult.__post_init__` is explicitly about not "copying the validation
    per provider". Adding three checks to five files would have made it
    eleven; routing all of them through `_argcheck` makes it one each.
    """
    import ast
    from pathlib import Path

    import emb_shootout.providers as providers_pkg

    package_dir = Path(providers_pkg.__file__).parent
    inline: list[str] = []
    for path in sorted(package_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # The shape every provider used to carry: an isinstance(..., bool)
            # exclusion inline in a constructor.
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "isinstance"
                and len(node.args) == 2
                and isinstance(node.args[1], ast.Name)
                and node.args[1].id == "bool"
            ):
                inline.append(path.name)
    assert not inline, (
        f"these provider modules still spell an argument rule inline: {sorted(set(inline))}; "
        "the rules live in emb_shootout/_argcheck.py"
    )


def test_the_raisers_use_the_wording_the_call_sites_already_had() -> None:
    """Routing six existing call sites through a shared raiser must not change
    any assertion; the messages are the ones they already used.
    """
    with pytest.raises(ValueError, match=r"^k must be a positive integer; got 0$"):
        require_positive_int("k", 0)
    with pytest.raises(ValueError, match=r"^c must be a finite number >= 0\.0; got nan$"):
        require_non_negative_finite("c", float("nan"))
