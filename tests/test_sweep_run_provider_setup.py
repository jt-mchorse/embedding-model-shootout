"""A provider that cannot be built is `error:` + exit 2, not a traceback at exit 1 (#202).

`sweep run` constructed the provider inside a `try` that caught only
`ValueError`. Measured on main with no optional extras installed: openai,
voyage, cohere, bge and nomic each raised `ImportError: ... requires the
optional ... extra` as a traceback, rc=1. A missing API key with the extra
installed raised the SDK's own error the same way.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from emb_shootout import providers
from emb_shootout.cli import main


def _corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    rows = [
        {"chunk_id": f"c{i:03d}", "text": f"document number {i} discusses topic {i % 3}"}
        for i in range(20)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _sweep(tmp_path: Path, provider: str) -> int:
    return main(
        [
            "sweep",
            "run",
            "--provider",
            provider,
            "--corpus",
            str(_corpus(tmp_path)),
            "--queries",
            "5",
            "--out",
            str(tmp_path / "r.json"),
        ]
    )


class _MissingCredentials(Exception):
    """Stands in for an SDK's own credential error (e.g. `openai.OpenAIError`)."""


@pytest.mark.parametrize(
    "exc",
    [
        ImportError("FakeProvider requires the optional 'fake' extra"),
        _MissingCredentials("Missing credentials. Please pass an `api_key`"),
        OSError("model 'BAAI/bge-small' could not be loaded"),
    ],
    ids=["missing-extra", "missing-key", "model-load"],
)
def test_a_provider_that_cannot_be_built_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    exc: Exception,
) -> None:
    def _broken() -> object:
        raise exc

    monkeypatch.setitem(providers.PROVIDER_REGISTRY, "fake", _broken)
    assert _sweep(tmp_path, "fake") == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert err.startswith("error: could not set up provider 'fake': ")
    assert type(exc).__name__ in err
    assert str(exc) in err
    assert not (tmp_path / "r.json").exists()


@pytest.mark.skipif(
    importlib.util.find_spec("openai") is not None, reason="the openai extra is installed"
)
def test_the_real_missing_extra_path_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _sweep(tmp_path, "openai") == 2
    assert "requires the optional 'openai' extra" in capsys.readouterr().err


def test_hash_still_runs(tmp_path: Path) -> None:
    assert _sweep(tmp_path, "hash") == 0
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["embedder_name"]
