"""`SweepResult` validated every field's *values* and no field's *identity* (#143).

`__post_init__` was completed one field at a time — #29/#31 (cost, dim,
counts), #31 again (recall/ndcg values), #65 (`embed_latency_ms` values), #133
(`notes` elements). Every one of those is a rule about a **value**.

`from_dict` validates two things `__post_init__` did not, and both are about
**identity**:

* `embedder_name` must be a non-empty `str` (#94/#95);
* every `recall_at_k` key must be the canonical spelling of a positive integer
  (#129, delegating the range to `validate_k_values`).

`__post_init__` checked `embedder_name` **not at all** — the only field in the
class with zero guards — and checked no key of either dict. So `to_dict` wrote
`results/*.json` this package's own `from_dict` refuses.

The cost guard's own comment establishes that closing exactly this asymmetry is
the pattern here: *"`from_dict` already rejects a bool cost pre-coercion
(#108); this closes its **direct-construction sibling**."* `embedder_name` was
the sibling nobody closed.

Measured on the unguarded constructor::

    embedder_name      construction   aggregate_markdown             to_dict->from_dict
    'openai' (control) ok             cell='openai'                  round-trips
    123                CONSTRUCTED    RAW AttributeError (.replace)  REFUSED
    None               CONSTRUCTED    RAW AttributeError (.replace)  REFUSED
    ['a']              CONSTRUCTED    RAW AttributeError (.replace)  REFUSED
    1.5                CONSTRUCTED    RAW AttributeError (.replace)  REFUSED
    True               CONSTRUCTED    RAW AttributeError (.replace)  REFUSED
    ''                 CONSTRUCTED    cell=''  <- SILENT             REFUSED

    recall_at_k        keys written   to_dict -> from_dict
    {1: 0.5} control   ['1']          reload [1], identical
    {'1': 0.5}         ['1']          reload [1], NOT identical  <- SILENT
    {1.5: 0.5}         ['1.5']        REFUSED
    {True: 0.5}        ['True']       REFUSED
    {-1: 0.5}          ['-1']         REFUSED

Both silent rows are why the round-trip assertions below compare the **value**
rather than the absence of an exception. `{'1': 0.5}` round-trips without
raising and is still wrong, and `embedder_name=''` renders an empty embedder
column into the published README table with nothing complaining.

And `sorted(results, key=lambda r: r.embedder_name)` over a mixed batch raised
`TypeError: '<' not supported between instances of 'int' and 'str'` — the other
harm `from_dict`'s comment names.
"""

from __future__ import annotations

from typing import Any

import pytest

from emb_shootout._argcheck import is_non_empty_str
from emb_shootout.sweep import SweepResult, aggregate_markdown


def _result(**overrides: Any) -> SweepResult:
    base: dict[str, Any] = {
        "embedder_name": "openai-3-small",
        "embedder_dim": 8,
        "cost_per_million_tokens": 1.0,
        "n_corpus": 2,
        "n_queries": 1,
        "recall_at_k": {1: 0.5},
        "ndcg_at_10": 0.5,
        "embed_latency_ms": {"corpus_total": 1.0},
    }
    base.update(overrides)
    return SweepResult(**base)


# (label, value). Every one wrote a dict `from_dict` refuses.
BAD_NAMES: list[tuple[str, Any]] = [
    ("int", 123),
    ("None", None),
    ("empty string", ""),  # the silent row
    ("list", ["a"]),
    ("float", 1.5),
    ("bool", True),
    ("dict", {"name": "x"}),
]

# (label, value, which arm must answer)
BAD_KEYS: list[tuple[str, Any, str]] = [
    ("str key, the silent row", {"1": 0.5}, "must be an int"),
    ("float key", {1.5: 0.5}, "must be an int"),
    ("bool key", {True: 0.5}, "must be an int"),
    ("None key", {None: 0.5}, "must be an int"),
    ("negative key", {-1: 0.5}, "recall_at_k keys:"),
    ("zero key", {0: 0.5}, "recall_at_k keys:"),
]

# Accepted, and each must round-trip IDENTICALLY.
GOOD: list[tuple[str, dict[str, Any]]] = [
    ("the shipped shape", {}),
    ("several ks", {"recall_at_k": {1: 0.5, 5: 0.7, 10: 0.9}}),
    # Reachable via `from_dict` on an external result file, per #83. An
    # over-broad key rule would refuse it and break the empty-`ks` table.
    ("empty recall_at_k (#83)", {"recall_at_k": {}}),
    # `aggregate_markdown` already escapes `|`; a name containing one is a
    # markdown-cell question, not an identity question, and must stay valid.
    ("name containing a pipe", {"embedder_name": "a|b"}),
    ("non-ASCII name", {"embedder_name": "modele-e-é"}),
    ("single-char name", {"embedder_name": "x"}),
]


@pytest.mark.parametrize(("label", "value"), BAD_NAMES, ids=[r[0] for r in BAD_NAMES])
def test_a_bad_embedder_name_is_refused_at_construction(label: str, value: Any) -> None:
    with pytest.raises(ValueError, match="embedder_name must be a non-empty string") as exc:
        _result(embedder_name=value)
    # The type, not just a blanket sentence — this is the wording `from_dict`
    # already used, and six call sites' worth of assertions depend on it.
    assert type(value).__name__ in str(exc.value), f"{label}: message omits the type"


@pytest.mark.parametrize(("label", "value", "arm"), BAD_KEYS, ids=[r[0] for r in BAD_KEYS])
def test_a_bad_recall_key_is_refused_at_construction(
    label: str, value: dict[Any, Any], arm: str
) -> None:
    with pytest.raises(ValueError, match="recall_at_k") as exc:
        _result(recall_at_k=value)
    # WHICH arm answers is asserted, not merely that something raised. The type
    # arm has to run before the range arm, because `sorted()` over mixed key
    # types raises a raw TypeError.
    assert arm in str(exc.value), f"{label}: expected the {arm!r} arm, got {exc.value}"


@pytest.mark.parametrize(("label", "overrides"), GOOD, ids=[r[0] for r in GOOD])
def test_a_valid_result_round_trips_identically(label: str, overrides: dict[str, Any]) -> None:
    result = _result(**overrides)
    back = SweepResult.from_dict(result.to_dict())
    # By VALUE, on both fields. `{'1': 0.5}` round-tripped without raising and
    # was still wrong, so "did not raise" is not the assertion here.
    assert back.embedder_name == result.embedder_name, label
    assert back.recall_at_k == result.recall_at_k, label
    assert back.to_dict() == result.to_dict(), label


def test_the_empty_name_was_the_silent_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """`''` is why this is worth a guard rather than six tracebacks.

    The other six bad values crash `aggregate_markdown`. An empty string does
    not: it constructs, renders an empty embedder column into the markdown the
    README publishes, and writes a result file. Pinned by building the table
    from a *valid* result and asserting the cell is non-empty, so the rule this
    test protects is "the published table always names its embedder".
    """
    with pytest.raises(ValueError, match="non-empty string"):
        _result(embedder_name="")

    # And the positive half: a valid name still reaches the cell.
    table = aggregate_markdown([_result()])
    data_row = table.splitlines()[2]
    first_cell = data_row.split("|")[1].strip()
    assert first_cell == "openai-3-small", f"embedder cell rendered {first_cell!r}"
    assert first_cell != "", "the embedder column is empty in the published table"


def test_the_two_harms_from_dicts_comment_names_are_now_unreachable() -> None:
    """`from_dict`'s guard comment names an `AttributeError` from
    `aggregate_markdown` and a `TypeError` from sorting mixed names. Both were
    reachable through the constructor; assert the constructor now closes the
    road rather than that the downstream calls happen to cope."""
    # Sorting a batch: every name is a str by construction, so this cannot
    # raise regardless of what the caller passed.
    names = ["b", "a", "c"]
    results = [_result(embedder_name=n) for n in names]
    assert [r.embedder_name for r in sorted(results, key=lambda r: r.embedder_name)] == [
        "a",
        "b",
        "c",
    ]
    # And the .replace("|", ...) road: a non-str can no longer get here.
    with pytest.raises(ValueError, match="embedder_name must be a non-empty string"):
        _result(embedder_name=123)


def test_from_dicts_own_guards_are_not_redundant() -> None:
    """RUN the claim rather than restate #133's argument for it.

    `from_dict` coerces before the constructor sees anything:
    `_coerce_recall_keys` turns `"1"` into `1`, and `int()`/`float()` reshape
    the scalars. If its guards were removed on the reasoning that the
    constructor now covers them, a JSON `"05"` key would arrive as a perfectly
    good `int` 5 — colliding with a real `"5"` and dropping a measurement,
    which is exactly #129 — with nothing left for the constructor's rule to
    object to.
    """
    # The laundering, demonstrated on the values themselves.
    assert int("05") == 5, "the coercion the loader's canonical-spelling rule exists for"
    assert is_non_empty_str("5")
    assert is_non_empty_str("05")
    # Both spellings would satisfy the CONSTRUCTOR's rule once coerced.
    assert _result(recall_at_k={5: 0.9}).recall_at_k == {5: 0.9}
    # And the loader still refuses the non-canonical spelling at its own seam.
    payload = _result(recall_at_k={5: 0.9}).to_dict()
    payload["recall_at_k"] = {"05": 0.9}
    with pytest.raises(ValueError, match="canonical spelling"):
        SweepResult.from_dict(payload)


def test_neither_rule_is_respelled_in_a_second_place() -> None:
    """Structural, because a copy passes every behavioural row above.

    #141 built `_argcheck.py` for precisely this reason, and this repo has
    already measured that the shared-vs-copied distinction is invisible to a
    behavioural suite. Counted over AST string literals excluding docstrings,
    so the paragraphs above quoting the messages are not false hits.
    """
    import ast
    from pathlib import Path

    from emb_shootout import sweep as sweep_module

    tree = ast.parse(Path(sweep_module.__file__).read_text(encoding="utf-8"))
    doc_ids: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and body
        ):
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                doc_ids.add(id(first.value))
    literals = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc_ids
    ]
    assert len(literals) > 30, "the literal scan found almost nothing; it is not reading the module"

    # The name rule's message is spelled twice on purpose — once per call site,
    # because each interpolates its own value expression — but the PREDICATE
    # must not be. No `isinstance(..., str)`-and-truthiness pair may appear for
    # embedder_name outside `_argcheck`.
    name_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "is_non_empty_str"
    ]
    assert len(name_calls) == 2, (
        f"expected exactly two callers of the shared name predicate, found {len(name_calls)}; "
        "a third is a new site that should share it, zero is the rule respelled inline"
    )

    # The range rule delegates to `validate_k_values` from BOTH the loader and
    # the constructor, rather than either restating `k >= 1`.
    k_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_k_values"
    ]
    assert len(k_calls) >= 3, (
        f"expected validate_k_values to be called from run_sweep, the loader and the "
        f"constructor; found {len(k_calls)}"
    )
