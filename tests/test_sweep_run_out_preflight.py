"""`sweep run` refuses an unwritable --out before embedding anything (#200).

`--out` was touched only after `run_sweep` had embedded the whole corpus and
every query. Measured on main with a 12,020-chunk corpus and `--queries 20`:
an existing directory or a path under a file exited 2 only after 12,040 texts
had gone to the provider -- a full paid sweep, discarded.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from emb_shootout import providers
from emb_shootout.cli import main
from emb_shootout.providers.hash_embedder import HashEmbedderProvider

SENT: list[int] = []


class _CountingHash(HashEmbedderProvider):
    def embed(self, texts):  # type: ignore[no-untyped-def]
        SENT.append(len(texts))
        return super().embed(texts)


@pytest.fixture(autouse=True)
def _count(monkeypatch: pytest.MonkeyPatch) -> None:
    SENT.clear()
    monkeypatch.setitem(providers.PROVIDER_REGISTRY, "hash", _CountingHash)


def _corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    rows = [
        {"chunk_id": f"c{i:03d}", "text": f"document number {i} discusses topic {i % 3}"}
        for i in range(20)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _sweep(tmp_path: Path, out: Path) -> int:
    return main(
        [
            "sweep",
            "run",
            "--provider",
            "hash",
            "--corpus",
            str(_corpus(tmp_path)),
            "--queries",
            "5",
            "--out",
            str(out),
        ]
    )


def test_an_existing_directory_is_refused_with_nothing_embedded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "results"
    out.mkdir()
    assert _sweep(tmp_path, out) == 2
    assert SENT == []
    assert f"failed to write {out}" in capsys.readouterr().err


def test_a_path_under_a_file_is_refused_with_nothing_embedded(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    assert _sweep(tmp_path, blocker / "r.json") == 2
    assert SENT == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_a_read_only_directory_is_refused_with_nothing_embedded(tmp_path: Path) -> None:
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        assert _sweep(tmp_path, ro / "r.json") == 2
        assert SENT == []
    finally:
        ro.chmod(stat.S_IRWXU)


def test_a_good_path_still_embeds_and_writes(tmp_path: Path) -> None:
    out = tmp_path / "new" / "r.json"
    assert _sweep(tmp_path, out) == 0
    assert sum(SENT) > 0
    assert json.loads(out.read_text(encoding="utf-8"))["embedder_name"]


def test_a_symlink_into_a_writable_directory_passes(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    link = tmp_path / "link.json"
    link.symlink_to(real)
    assert _sweep(tmp_path, link) == 0
    assert link.is_symlink()
    assert json.loads(real.read_text(encoding="utf-8"))["embedder_name"]


def test_check_writable_leaves_nothing_behind(tmp_path: Path) -> None:
    from emb_shootout.io_utils import check_writable

    target = tmp_path / "sub" / "r.json"
    check_writable(target)
    assert not target.exists()
    assert sorted(p.name for p in (tmp_path / "sub").iterdir()) == []


def test_check_writable_raises_the_writers_error_for_a_directory(tmp_path: Path) -> None:
    from emb_shootout.io_utils import check_writable

    with pytest.raises(IsADirectoryError):
        check_writable(tmp_path)
