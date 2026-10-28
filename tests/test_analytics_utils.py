import json
from pathlib import Path

import pytest

from analytics_utils import append_event, build_answer_event, citation_coverage, load_events, retrieval_metrics


def test_citation_coverage_uses_nonempty_lines():
    assert citation_coverage("Claim [Source 1]\n\nAnother claim") == 0.5
    assert citation_coverage("") == 0.0


def test_answer_event_does_not_store_question_or_answer_text():
    event = build_answer_event(
        model="qwen3:4b", elapsed_seconds=1.23456, retrieval_seconds=0.12345,
        source_count=2, answer="Policy states 21 days. [Source 1]",
    )
    assert event["event"] == "answer"
    assert event["elapsed_seconds"] == 1.2346
    assert event["source_count"] == 2
    assert "question" not in event
    assert "answer" not in event
    assert event["citation_coverage"] == 1.0


def test_jsonl_append_and_load_ignore_malformed_lines(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    append_event(path, {"event": "answer", "elapsed_seconds": 1.2})
    with path.open("a", encoding="utf-8") as handle:
        handle.write("not-json\n")
    assert load_events(path) == [{"event": "answer", "elapsed_seconds": 1.2}]


def test_retrieval_metrics_hit_rate_and_mrr():
    rows = [
        {"expected_source": "policy.txt", "retrieved_sources": ["other.txt", "policy.txt"]},
        {"expected_source": "hr.txt", "retrieved_sources": ["hr.txt", "other.txt"]},
        {"expected_source": "finance.txt", "retrieved_sources": ["other.txt"]},
    ]
    result = retrieval_metrics(rows, k=2)
    assert result["evaluated"] == 3
    assert result["hits"] == 2
    assert result["hit_rate_at_k"] == pytest.approx(2 / 3)
    assert result["mrr_at_k"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)


def test_retrieval_metrics_validates_k():
    with pytest.raises(ValueError):
        retrieval_metrics([], k=0)
