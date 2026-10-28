from pathlib import Path

import pytest

from ingestion import extract_text, validate_document_size


def test_text_extraction_preserves_content(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("Hello\nknowledge base", encoding="utf-8")
    assert extract_text(path) == [(None, "Hello\nknowledge base")]


def test_csv_extraction_keeps_headers_and_separates_rows(tmp_path: Path):
    path = tmp_path / "employees.csv"
    path.write_text("name,team\nAsha,Data\nSam,IT\n", encoding="utf-8")
    segments = extract_text(path)
    assert len(segments) == 2
    assert segments[0][0] == 2
    assert "name: Asha" in segments[0][1]
    assert "team: Data" in segments[0][1]
    assert segments[1][0] == 3


def test_unsupported_file_returns_no_segments(tmp_path: Path):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"data")
    assert extract_text(path) == []


def test_size_validation_rejects_oversized_document(tmp_path: Path):
    path = tmp_path / "large.txt"
    path.write_text("12345", encoding="utf-8")
    with pytest.raises(ValueError, match="configured limit"):
        validate_document_size(path, 4)


def test_size_validation_requires_positive_limit(tmp_path: Path):
    path = tmp_path / "small.txt"
    path.write_text("a", encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        validate_document_size(path, 0)
