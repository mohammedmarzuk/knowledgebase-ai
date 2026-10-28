from upload_utils import content_sha256, unique_destination


def test_content_sha256_is_stable():
    assert content_sha256(b"abc") == content_sha256(b"abc")
    assert content_sha256(b"abc") != content_sha256(b"abd")


def test_unique_destination_deduplicates_content(tmp_path):
    (tmp_path / "original.txt").write_bytes(b"same")
    path, duplicate = unique_destination(tmp_path, "copy.txt", b"same")
    assert duplicate is True
    assert path.name == "original.txt"


def test_unique_destination_avoids_overwrite(tmp_path):
    (tmp_path / "report.txt").write_bytes(b"old")
    path, duplicate = unique_destination(tmp_path, "report.txt", b"new")
    assert duplicate is False
    assert path.name == "report_2.txt"
