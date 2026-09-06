import errno
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")

from farkle.utils import artifact_contract
from farkle.utils.writer import ParquetShardWriter, atomic_path


def test_parquet_shard_writer_atomic(tmp_path):
    if getattr(pa, "__version__", "0.0.0") == "0.0.0":
        pytest.skip("pyarrow stub active")
    out_path = tmp_path / "sample.parquet"
    table = pa.Table.from_pydict({"winner": ["P1"], "score": [100]})

    with ParquetShardWriter(str(out_path)) as writer:
        writer.write_batch(table)

    assert out_path.exists()
    assert writer.rows_written == 1
    assert not any(tmp_path.glob("._tmp_*"))


def test_atomic_path_stages_in_configured_scratch_and_cleans(tmp_path: Path) -> None:
    results = tmp_path / "results"
    scratch = tmp_path / "scratch"
    destination = results / "value.json"

    with atomic_path(str(destination), temp_dir=scratch) as staged_text:
        staged = Path(staged_text)
        assert staged.parent == scratch
        assert not destination.exists()
        staged.write_text("new", encoding="utf-8")

    assert destination.read_text(encoding="utf-8") == "new"
    assert list(scratch.iterdir()) == []


def test_atomic_path_failure_preserves_destination_and_cleans_scratch(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    destination = tmp_path / "result.txt"
    destination.write_text("old", encoding="utf-8")

    with (
        pytest.raises(RuntimeError, match="writer failed"),
        atomic_path(str(destination), temp_dir=scratch) as staged_text,
    ):
        Path(staged_text).write_text("partial", encoding="utf-8")
        raise RuntimeError("writer failed")

    assert destination.read_text(encoding="utf-8") == "old"
    assert list(scratch.iterdir()) == []


def test_cross_filesystem_publish_uses_bounded_destination_local_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "scratch"
    destination = tmp_path / "results" / "result.bin"
    scratch.mkdir()
    staged = scratch / "staged.bin"
    staged.write_bytes(b"payload")
    real_replace = artifact_contract.replace_file_atomic
    calls: list[tuple[Path, Path]] = []
    copy_lengths: list[int] = []

    def replace_once(source, target):
        source_path, target_path = Path(source), Path(target)
        calls.append((source_path, target_path))
        if len(calls) == 1:
            raise OSError(errno.EXDEV, "cross-device link")
        real_replace(source_path, target_path)

    real_copy = artifact_contract.shutil.copyfileobj

    def record_copy(source, target, length=0):
        copy_lengths.append(length)
        real_copy(source, target, length=length)

    monkeypatch.setattr(artifact_contract, "replace_file_atomic", replace_once)
    monkeypatch.setattr(artifact_contract.shutil, "copyfileobj", record_copy)

    artifact_contract.publish_staged_file_atomic(staged, destination, copy_buffer_bytes=4)

    assert destination.read_bytes() == b"payload"
    assert copy_lengths == [4]
    assert calls[1][0].parent == destination.parent
    assert not staged.exists()
    assert not any(destination.parent.glob("._publish_*"))
