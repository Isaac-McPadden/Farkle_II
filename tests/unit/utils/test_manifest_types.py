import pytest

from farkle.utils.manifest import append_manifest_line, iter_manifest
from farkle.utils.types import normalize_compression


def test_normalize_compression_invalid_value_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported parquet compression"):
        normalize_compression("zip")


def test_append_manifest_ensure_dir_false_on_missing_parent_raises(tmp_path) -> None:
    manifest_path = tmp_path / "missing" / "manifest.ndjson"

    with pytest.raises(FileNotFoundError):
        append_manifest_line(
            manifest_path,
            {"path": "row.parquet"},
            add_timestamp=False,
            ensure_dir=False,
        )


def test_iter_manifest_skips_blank_lines_between_records(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.ndjson"
    manifest_path.write_text('{"path":"a"}\n\n   \n{"path":"b"}\n', encoding="utf-8")

    rows = list(iter_manifest(manifest_path))

    assert [row["path"] for row in rows] == ["a", "b"]
