"""Every JSON reader treats input `json.loads` cannot decode as bad input (#204).

Each reader caught `JSONDecodeError` only, and `json.loads` raises two other
things on bad input: `RecursionError` on deep nesting (about 100,000 levels on
3.11/3.12, the interpreters CI runs) and a plain `ValueError` for an integer
past CPython's 4300-digit limit. Measured on main with 3.11, both made
`emb-shootout corpus validate` crash at exit 1, the "findings" code, with no
finding printed. `io_utils.loads_json` turns both into a `JSONDecodeError`, the
llm-eval-harness#333 shape.

3.14's `json` survives the nesting, so the real deep input only bites on CI.
`test_a_recursion_error_is_translated_on_any_interpreter` forces the error so the
translation is pinned everywhere.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from emb_shootout import io_utils
from emb_shootout.cli import main
from emb_shootout.validate import validate_corpus

_PACKAGE = Path(__file__).resolve().parent.parent / "emb_shootout"
_DEEP = "[" * 100_000 + "]" * 100_000
_LONG_INT = '{"chunk_id": ' + "1" * 5000 + ', "text": "x"}'
_GOOD = json.dumps({"chunk_id": "c1", "text": "a valid row after the bad one"})


def _decodes(text: str) -> bool:
    try:
        json.loads(text)
    except (RecursionError, ValueError):
        return False
    return True


# 3.14 decodes the deep input (a nested list, so `not_an_object`); 3.11/3.12
# cannot. Either way it is a finding, never a traceback.
_CODE = {
    "deep": "not_an_object" if _decodes(_DEEP) else "malformed_json",
    "long-int": "malformed_json",
}


@pytest.mark.parametrize(("kind", "bad"), [("deep", _DEEP), ("long-int", _LONG_INT)])
def test_validate_reports_a_finding_and_keeps_going(tmp_path: Path, kind: str, bad: str) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(bad + "\n" + _GOOD + "\n", encoding="utf-8")
    report = validate_corpus(path)
    assert [f.code for f in report.findings] == [_CODE[kind]]
    assert report.findings[0].line_no == 1
    assert report.n_rows == 2  # the row after the bad one was still read


@pytest.mark.parametrize(("kind", "bad"), [("deep", _DEEP), ("long-int", _LONG_INT)])
def test_cli_validate_exits_one_with_a_finding_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str, bad: str
) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(bad + "\n", encoding="utf-8")
    rc = main(["corpus", "validate", str(path)])
    out = capsys.readouterr()
    assert rc == 1
    assert f"[{_CODE[kind]}]" in out.err  # findings go to stderr, the summary to stdout
    assert "findings=1" in out.out
    assert "Traceback" not in out.err


@pytest.mark.parametrize("bad", [_DEEP, "1" * 5000], ids=["deep", "long-int"])
@pytest.mark.parametrize("command", ["aggregate", "plot"])
def test_a_bad_result_file_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], bad: str, command: str
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "bad.json").write_text(bad, encoding="utf-8")
    out_flag = (
        ["--out", str(tmp_path / "agg.md")]
        if command == "aggregate"
        else [
            "--out-png",
            str(tmp_path / "p.png"),
        ]
    )
    rc = main(["sweep", command, "--results-dir", str(results), *out_flag])
    err = capsys.readouterr().err
    assert rc == 2
    assert "bad.json" in err
    assert "Traceback" not in err


def test_a_recursion_error_is_translated_on_any_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deep(_text: str) -> object:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(io_utils.json, "loads", deep)
    with pytest.raises(json.JSONDecodeError, match="nested too deeply"):
        io_utils.loads_json("[[[]]]")


def test_no_module_but_io_utils_calls_json_loads() -> None:
    # The point of one helper: the next reader cannot catch one of the three
    # failures and miss the others.
    offenders = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        if path.name == "io_utils.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"loads", "load"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "json"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"call io_utils.loads_json instead: {offenders}"
