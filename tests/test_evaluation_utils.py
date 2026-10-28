from evaluation_utils import citation_integrity, evaluate_answer, is_abstention, token_overlap_scores


def test_token_overlap_scores_exact_reference():
    scores = token_overlap_scores("RAG means retrieval augmented generation", "Retrieval-augmented generation")
    assert scores["token_recall"] == 1.0
    assert scores["token_f1"] > 0.5


def test_token_overlap_scores_empty_input():
    assert token_overlap_scores("", "some reference") == {
        "token_precision": 0.0, "token_recall": 0.0, "token_f1": 0.0
    }


def test_citation_integrity_detects_invalid_source_labels():
    result = citation_integrity("Fact [Source 1]\nOther [Source 3]", source_count=2)
    assert result["citation_count"] == 2
    assert result["invalid_citations"] == 1
    assert result["citation_coverage"] == 1.0
    assert result["citation_integrity"] == 0


def test_citation_integrity_with_no_citations():
    result = citation_integrity("An unsupported claim", source_count=2)
    assert result["citation_integrity"] == 0
    assert result["citation_coverage"] == 0.0


def test_abstention_detection():
    assert is_abstention("I could not find sufficient evidence in the indexed documents.")
    assert not is_abstention("Annual leave is 21 days. [Source 1]")


def test_evaluate_answer_combines_signals():
    result = evaluate_answer("RAG means retrieval-augmented generation. [Source 1]", "Retrieval-augmented generation", 1)
    assert result["invalid_citations"] == 0
    assert result["citation_integrity"] == 1
    assert result["token_recall"] == 1.0
    assert result["abstained"] is False
