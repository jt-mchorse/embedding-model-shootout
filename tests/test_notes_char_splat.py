"""A string is a sequence of strings, so the check must run before the copy (#153).

`SweepResult.__post_init__` copies `notes` with `list(...)` (#133) and then
validates that every element is a `str` (#65). Both halves are right and the
order defeated them: `list("a note")` does not raise — it splats into one entry
per character, and every one of those *is* a `str`, so the element loop inspected
the splatted list and passed.

Measured end to end before any edit:

    run_sweep(..., notes="recall looks low on the small corpus")
      -> SweepResult.notes  == 36 single-character strings
      -> to_dict()["notes"] == the same 36, written to results/*.json
      -> from_dict(that)    == the same 36 back

**`run_sweep` is the sharp site because it is type-clean.** Its parameter is
`notes: Sequence[str] = ()`, and a `str` *is* a `Sequence[str]` — mypy accepts
`run_sweep(notes="...")` without a murmur while flagging the
`SweepResult(notes="...")` spelling, whose field is declared `list[str]`. The
annotation that looks like it excludes this input is the one that admits it.

Why the guard is a runtime check and not a narrowed annotation
--------------------------------------------------------------

Narrowing `run_sweep`'s parameter to `list[str]` would let a type checker reject
the caller — but this repo runs **no mypy in CI** (`.github/workflows/ci.yml`
runs `ruff check`, `ruff format --check` and `pytest`), so it would buy nothing
at the gate. It would also reject this parameter's own `()` default and break
the symmetry with `corpus`, `queries` and `k_values`, all `Sequence[...]` on
purpose. Checked rather than assumed; see `test_ci_has_no_type_gate_so_the_check_must_be_at_runtime`.

The empty string is why this survived
--------------------------------------

`tests/test_plot_drawn_axes.py` passed `notes=""` and had done so for months.
`list("")` is `[]`, so the coercion was harmless *by accident* on exactly that
one input — the only member of the offending class that produces the right
answer. Fixed to `notes=[]` in this change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from emb_shootout._argcheck import require_mapping, require_str_sequence
from emb_shootout.sweep import SweepResult

_ROOT = Path(__file__).resolve().parents[1]


def _result(**overrides: Any) -> SweepResult:
    kwargs: dict[str, Any] = {
        "embedder_name": "hash-256",
        "embedder_dim": 64,
        "cost_per_million_tokens": 1.0,
        "n_corpus": 10,
        "n_queries": 5,
        "recall_at_k": {1: 0.5, 5: 0.8, 10: 0.9},
        "ndcg_at_10": 0.7,
        "embed_latency_ms": {"corpus_total": 12.0, "query_p50": 1.0, "query_p95": 2.0},
        "notes": ["one real note"],
    }
    kwargs.update(overrides)
    return SweepResult(**kwargs)


# --------------------------------------------------------------------------
# The splat, at each of the three sites
# --------------------------------------------------------------------------

_NOTE = "recall looks low on the small corpus"


def test_the_constructor_refuses_a_string_notes() -> None:
    with pytest.raises(ValueError, match="notes must be a sequence of strings, not a str"):
        _result(notes=_NOTE)


def test_from_dict_refuses_a_string_notes() -> None:
    """The read path has no element guard of its own.

    It relies entirely on `__post_init__`, which — before this change — was
    handed an already-splatted list. A results JSON carrying `"notes": "a note"`
    is the reachable input, and this repo's `results/*.json` are hand-editable
    committed artifacts.
    """
    payload = json.loads(json.dumps(_result().to_dict()))
    payload["notes"] = _NOTE
    with pytest.raises(ValueError, match="notes must be a sequence of strings, not a str"):
        SweepResult.from_dict(payload)


def test_run_sweep_refuses_a_string_notes() -> None:
    """The type-clean site. `Sequence[str]` is satisfied by `str`.

    Reached through the real `run_sweep` rather than by calling the helper, so
    the arm reads the path a caller actually takes.
    """
    from emb_shootout.providers.hash_embedder import HashEmbedderProvider
    from emb_shootout.sweep import run_sweep

    corpus, queries = _hermetic_corpus_and_queries()
    with pytest.raises(ValueError, match="notes must be a sequence of strings, not a str"):
        run_sweep(
            corpus,
            queries,
            embedder=HashEmbedderProvider(dim=64),
            k_values=(1,),
            notes=_NOTE,
        )


def _hermetic_corpus_and_queries() -> tuple[list[Any], list[Any]]:
    from emb_shootout.corpus import Chunk
    from emb_shootout.queries import Query

    corpus = [
        Chunk(
            chunk_id=f"c{i}",
            text=f"alpha beta gamma delta chunk number {i}",
            module="m",
            qualname=f"m.f{i}",
            kind="function",
            source=f"def f{i}(): pass",
        )
        for i in range(3)
    ]
    queries = [Query(query_id="q0", text="alpha beta", expected_chunk_id="c0")]
    return corpus, queries


def test_the_splat_really_was_the_old_behaviour() -> None:
    """Pin what the coercion did, so the arms above are not tautologies.

    If `list(...)` ever stopped splatting a string, the guard would be defending
    against nothing and these arms would still pass.
    """
    assert len(list(_NOTE)) == 36
    assert list(_NOTE)[:3] == ["r", "e", "c"]
    assert all(isinstance(c, str) for c in list(_NOTE)), (
        "every splatted character is a str, which is exactly why an element-type "
        "loop after the copy cannot catch this"
    )


def test_the_empty_string_was_the_harmless_member_and_is_refused_too() -> None:
    """`list("")` is `[]`, so `notes=""` produced the right answer by accident.

    That is why one lived in `tests/test_plot_drawn_axes.py` unnoticed. Refusing
    it is deliberate: a rule that made an exception for the empty string would be
    a rule about *length*, and the defect is about *type*.
    """
    assert list("") == []
    with pytest.raises(ValueError, match="not a str"):
        _result(notes="")


# --------------------------------------------------------------------------
# The ordering — the thing #133's own arms cannot see
# --------------------------------------------------------------------------


def test_the_check_runs_before_the_copy_not_after() -> None:
    """A check placed after `list(...)` is satisfied by the splatted list.

    This is the arm that separates the fix from the neighbour that adds an
    element-type check in the wrong place. Applying `require_str_sequence` to the
    *result* of the copy passes on the very input the guard exists to refuse.
    """
    already_copied = list(_NOTE)
    require_str_sequence("notes", already_copied)  # no raise: it is a list of str
    with pytest.raises(ValueError, match="not a str"):
        require_str_sequence("notes", _NOTE)


def test_a_generator_is_refused_rather_than_consumed() -> None:
    """The mirror of the bug: single-use iterables.

    Validating a generator's elements would consume it and leave the copy empty.
    Refused with its type named rather than silently producing zero notes.
    """
    with pytest.raises(ValueError, match="ordered sequence"):
        _result(notes=(n for n in ["a", "b"]))


def test_a_set_is_refused_because_notes_are_published_in_order() -> None:
    with pytest.raises(ValueError, match="ordered sequence"):
        _result(notes={"a", "b"})


def test_bytes_are_refused_by_type_rather_than_by_their_elements() -> None:
    """`bytes` is a sequence whose elements are `int`, so the element loop *would*
    catch it — with a message about `notes[0]` being an `int` rather than about
    the value being bytes. Named at the right level."""
    with pytest.raises(ValueError, match="not a bytes"):
        _result(notes=b"ab")
    with pytest.raises(ValueError, match="not a bytearray"):
        _result(notes=bytearray(b"ab"))


def test_a_tuple_still_passes_because_run_sweeps_own_default_is_one() -> None:
    """A rule demanding a `list` would reject `notes: Sequence[str] = ()`."""
    assert _result(notes=("a", "b")).notes == ["a", "b"]
    assert _result(notes=()).notes == []


# --------------------------------------------------------------------------
# The quieter half: the two dict coercions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "pairs"),
    [("recall_at_k", [(5, 1.0)]), ("embed_latency_ms", [("corpus_total", 1.0)])],
)
def test_a_list_of_pairs_is_refused_rather_than_coerced(field: str, pairs: Any) -> None:
    """`dict([(5, 1.0)])` builds a mapping without raising, so both fields
    accepted a list of pairs and the validation loop then found well-formed
    numeric entries. Less harmful than the splat — the mapping is what such a
    caller probably meant — but the same hole, on fields declared
    `dict[int, float]` and `dict[str, float]`."""
    assert dict(pairs), "the coercion this refuses still succeeds in Python"
    with pytest.raises(ValueError, match=f"{field} must be a mapping"):
        _result(**{field: pairs})


def test_require_mapping_accepts_any_mapping_not_only_dict() -> None:
    from collections import OrderedDict
    from types import MappingProxyType

    require_mapping("recall_at_k", OrderedDict({1: 0.5}))
    require_mapping("recall_at_k", MappingProxyType({1: 0.5}))
    assert _result(recall_at_k=OrderedDict({1: 0.5, 5: 0.8, 10: 0.9})).recall_at_k == {
        1: 0.5,
        5: 0.8,
        10: 0.9,
    }


# --------------------------------------------------------------------------
# The fourth site, which the population walk found and the issue did not
# --------------------------------------------------------------------------


def test_k_values_bytes_used_to_be_accepted_outright() -> None:
    """`list(b"\x05")` is `[5]`, and 5 is a valid k.

    The sharper of the two `k_values` harms: not a bad message, an accepted
    sweep. A caller passing a byte string got a run at whatever k the bytes
    happened to index to, and nothing anywhere said so.
    """
    from emb_shootout.sweep import validate_k_values

    assert list(b"\x05") == [5], "the coercion this refuses still succeeds"
    with pytest.raises(ValueError, match="k_values must be a sequence of ints, not a bytes"):
        validate_k_values(b"\x05")


def test_k_values_string_is_refused_by_shape_not_by_its_splatted_elements() -> None:
    """It *was* refused — and the message named `['1', '5']`, a list the caller
    never passed. A correct verdict reached through the wrong object is still a
    diagnostic a reader cannot act on."""
    from emb_shootout.sweep import validate_k_values

    with pytest.raises(ValueError, match="k_values") as excinfo:
        validate_k_values("15")
    message = str(excinfo.value)
    assert "not a str" in message
    assert "['1', '5']" not in message, (
        "the message still reports the splatted list rather than the value the caller passed"
    )


def test_k_values_still_accepts_the_shapes_run_sweep_uses() -> None:
    from emb_shootout.sweep import validate_k_values

    validate_k_values((1, 5, 10))
    validate_k_values([1, 5, 10])


# --------------------------------------------------------------------------
# The premise behind choosing a runtime check over a narrowed annotation
# --------------------------------------------------------------------------


def test_ci_has_no_type_gate_so_the_check_must_be_at_runtime() -> None:
    """The reason `run_sweep`'s annotation was not narrowed to `list[str]`.

    Narrowing would let a type checker reject the caller — but only if one ran.
    This repo's CI runs `ruff check`, `ruff format --check` and `pytest`, and no
    mypy. Asserted rather than remembered, because adding a type gate later is
    exactly the change that would make the annotation argument valid and this
    comment stale.
    """
    ci = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "pytest" in ci, "the CI file moved or stopped running tests"
    assert "mypy" not in ci, (
        "CI now runs a type checker. Narrowing `run_sweep`'s `notes` annotation "
        "to `list[str]` would now buy something at the gate — revisit D-013's "
        "reasoning rather than leaving this arm red. The runtime check should "
        "stay regardless: `SweepResult` and `run_sweep` are both exported and a "
        "type checker is not a runtime guard (#143's argument)."
    )


# --------------------------------------------------------------------------
# The population
# --------------------------------------------------------------------------


def test_every_container_copy_in_sweep_is_guarded_first() -> None:
    """Discover the sites; do not trust the three the issue listed.

    Keyed on the shape — a `list(...)` or `dict(...)` over a value this module
    did not itself build — rather than on the three line numbers, because a
    fourth is the obvious next change to this file. The check walks the source
    for copy calls on a *caller-supplied* name and requires a guard to appear
    before each one in the same function.
    """
    import ast

    source = (_ROOT / "emb_shootout" / "sweep.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    guarded_names = {
        "require_str_sequence",
        "require_int_sequence",
        "require_mapping",
        "_checked_notes",
    }

    offenders: list[str] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body_src = ast.unparse(func)
        for call in ast.walk(func):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                continue
            if call.func.id not in {"list", "dict"} or not call.args:
                continue
            arg = ast.unparse(call.args[0])
            # `self.<field>` is this class's own field, already validated and
            # copied at construction — `to_dict`'s copies are the *outbound*
            # half #133 documents and are not a caller-supplied value.
            if arg.startswith("self."):
                continue
            # A comprehension or a value literally built on the spot is this
            # module's own data.
            if any(ch in arg for ch in "[{("):
                continue
            if not any(g in body_src for g in guarded_names):
                offenders.append(f"{func.name}: {ast.unparse(call)}")
    assert not offenders, (
        f"these copies run a coercing constructor over a caller-supplied value "
        f"with no guard in the same function: {offenders}. `list(...)` splats a "
        f"string and `dict(...)` accepts a list of pairs, and an element check "
        f"after the copy inspects the coerced value (#153)."
    )


def test_the_population_arm_is_not_vacuous() -> None:
    """The walk finds the real copy calls, so a pass means something."""
    import ast

    source = (_ROOT / "emb_shootout" / "sweep.py").read_text(encoding="utf-8")
    copies = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"list", "dict"}
    ]
    assert len(copies) >= 5, (
        f"the walk found only {len(copies)} list()/dict() calls in sweep.py; "
        f"#153 touched five, so the discovery has stopped discovering."
    )
