"""A default captured from the host is a placeholder in the corpus (#259).

`platform.architecture(executable=sys.executable)` put the venv's interpreter
path into corpus text. Measured on main 835ae11: two venvs on one Python
version, the README's `corpus build` -> exactly one line differed, chunk
`platform.architecture`, so the two corpus fingerprints differed and `sweep
aggregate` refused results that were otherwise identical.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
import urllib.parse
from pathlib import Path

import pytest

from emb_shootout.corpus import _safe_signature, build_corpus

_BUILD_UNDER = (
    "import sys; sys.executable = sys.argv[1]; del sys.argv[1]\n"
    "from emb_shootout.cli import main\n"
    "raise SystemExit(main(sys.argv[1:]))\n"
)


def _build_platform(out: Path, executable: str) -> bytes:
    # `sys.executable` is set before `platform` is imported, so its default
    # captures this value -- what a second venv at another path does.
    subprocess.run(
        [sys.executable, "-c", _BUILD_UNDER, executable, "corpus", "build"]
        + ["--out", str(out), "--module", "platform"],
        check=True,
        capture_output=True,
    )
    return out.read_bytes()


def test_builds_under_two_interpreter_paths_are_byte_identical(tmp_path: Path) -> None:
    a = _build_platform(tmp_path / "a.jsonl", "/venvs/a/bin/python3")
    b = _build_platform(tmp_path / "b.jsonl", "/elsewhere/b/bin/python3")
    assert a == b
    assert b"/venvs/a" not in a
    assert b"platform.architecture(executable=<sys.executable>, bits='', linkage='')" in a


def _module_with_default(name: str, default: str) -> types.ModuleType:
    module = types.ModuleType(name)

    def run(path=default):  # noqa: ANN001, ANN202
        """Run against *path*."""
        return path

    run.__module__ = name
    module.run = run  # type: ignore[attr-defined]
    return module


def _run_header(monkeypatch: pytest.MonkeyPatch, default: str) -> str:
    monkeypatch.setitem(sys.modules, "ems_259_mod", _module_with_default("ems_259_mod", default))
    (chunk,) = build_corpus(["ems_259_mod"])
    return chunk.text.split("\n", 1)[0]


@pytest.mark.parametrize(
    ("attr", "placeholder"),
    [
        ("executable", "<sys.executable>"),
        ("prefix", "<sys.prefix>"),
        ("base_prefix", "<sys.base_prefix>"),
        ("exec_prefix", "<sys.exec_prefix>"),
        ("base_exec_prefix", "<sys.base_exec_prefix>"),
    ],
)
def test_each_interpreter_path_default_is_a_placeholder(
    monkeypatch: pytest.MonkeyPatch, attr: str, placeholder: str
) -> None:
    # Distinct values, so each placeholder is only reachable by its own entry.
    for i, name in enumerate(("executable", "prefix", "base_prefix", "exec_prefix")):
        monkeypatch.setattr(sys, name, f"/host{i}/{name}")
    monkeypatch.setattr(sys, "base_exec_prefix", "/host9/base_exec_prefix")
    monkeypatch.setattr(sys, "_base_executable", "/host8/base_exe", raising=False)
    header = _run_header(monkeypatch, getattr(sys, attr))
    assert header == f"ems_259_mod.run(path={placeholder})"


def test_the_same_function_reads_the_same_under_two_executables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = []
    for exe in ("/venvs/a/bin/python3", "/elsewhere/b/bin/python3"):
        monkeypatch.setattr(sys, "executable", exe)
        headers.append(_run_header(monkeypatch, exe))
    assert headers[0] == headers[1] == "ems_259_mod.run(path=<sys.executable>)"


def test_a_path_under_the_prefix_keeps_only_its_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "prefix", "/venvs/a")
    header = _run_header(monkeypatch, os.path.join("/venvs/a", "lib", "site"))
    assert header == f"ems_259_mod.run(path=<sys.prefix>{os.sep}lib{os.sep}site)"


def test_the_working_directory_and_home_are_placeholders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    home = tmp_path / "home"
    work.mkdir()
    home.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("HOME", str(home))
    assert _run_header(monkeypatch, os.getcwd()) == "ems_259_mod.run(path=<cwd>)"
    assert _run_header(monkeypatch, str(home)) == "ems_259_mod.run(path=<home>)"


def test_a_literal_default_still_renders_literally(monkeypatch: pytest.MonkeyPatch) -> None:
    # Controls: a sibling path that only shares a string prefix, and a default
    # that happens to be a filesystem root while the build runs from that root.
    monkeypatch.setattr(sys, "prefix", "/venvs/a")
    assert _run_header(monkeypatch, "/venvs/ab") == "ems_259_mod.run(path='/venvs/ab')"
    monkeypatch.chdir("/")
    assert _safe_signature(urllib.parse.quote) == "(string, safe='/', encoding=None, errors=None)"
    assert _run_header(monkeypatch, "C") == "ems_259_mod.run(path='C')"
