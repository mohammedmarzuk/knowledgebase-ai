from answer_guardrails import inspect_answer


def test_valid_source_labels_have_no_invalid_citations():
    result = inspect_answer("The policy allows 15 days [Source 1].", [{"text": "policy"}])
    assert result["citation_count"] == 1
    assert result["invalid_citations"] == []
    assert result["has_citations"] is True


def test_invalid_source_labels_are_reported():
    result = inspect_answer("The policy allows 15 days [Source 3].", [{"text": "policy"}])
    assert result["invalid_citations"] == [3]
    assert result["warnings"]


def test_uncited_non_abstention_answer_warns():
    result = inspect_answer("Employees get 15 days of leave.", [{"text": "policy"}])
    assert any("no recognizable source labels" in warning for warning in result["warnings"])


def test_abstention_is_detected():
    result = inspect_answer("I could not find sufficient evidence in the indexed documents to answer that.", [])
    assert result["abstained"] is True
    assert not result["warnings"]
