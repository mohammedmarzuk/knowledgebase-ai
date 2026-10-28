from pathlib import Path

from document_intelligence import append_job_event, file_sha256, load_job_history, save_summary, load_cached_summary


def test_file_hash_changes_when_content_changes(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("first", encoding="utf-8")
    first = file_sha256(p)
    p.write_text("second", encoding="utf-8")
    assert file_sha256(p) != first


def test_summary_cache_is_content_addressed(tmp_path, monkeypatch):
    import document_intelligence as di
    monkeypatch.setattr(di, "INSIGHTS_DIR", tmp_path / "insights")
    p = tmp_path / "doc.txt"
    p.write_text("version one", encoding="utf-8")
    saved = save_summary(p, "A useful summary", "test-model", 11)
    assert saved["summary"] == "A useful summary"
    assert load_cached_summary(p)["sha256"] == saved["sha256"]
    p.write_text("version two", encoding="utf-8")
    assert load_cached_summary(p) is None


def test_job_history_is_bounded_and_persistent(tmp_path):
    target = tmp_path / "history.json"
    for n in range(4):
        append_job_event({"kind": "test", "number": n}, history_path=target, keep=3)
    events = load_job_history(target)
    assert [item["number"] for item in events] == [1, 2, 3]
    assert all(item.get("timestamp") for item in events)
