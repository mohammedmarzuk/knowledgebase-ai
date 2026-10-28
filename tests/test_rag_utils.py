from pathlib import Path

import pytest

from rag_utils import chunk_text, document_fingerprint


def test_chunk_text_returns_empty_for_empty_input() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\t") == []


def test_chunk_text_splits_long_text_and_preserves_all_words() -> None:
    words = [f"word{i}" for i in range(11)]
    chunks = chunk_text(" ".join(words), size=4, overlap=1)

    assert [len(chunk.split()) for chunk in chunks] == [4, 4, 4, 2]
    reconstructed = chunks[0].split() + [word for chunk in chunks[1:] for word in chunk.split()[1:]]
    assert reconstructed == words


def test_chunk_text_supports_no_overlap() -> None:
    assert chunk_text("one two three four", size=2, overlap=0) == [
        "one two",
        "three four",
    ]


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (-1, 0), (4, -1), (4, 4), (4, 5)],
)
def test_chunk_text_rejects_invalid_parameters(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("one two three", size=size, overlap=overlap)


def test_document_fingerprint_is_stable_and_32_characters() -> None:
    args = (Path("policy.txt"), 2, 3, "Sample policy chunk")
    first = document_fingerprint(*args)
    second = document_fingerprint(*args)

    assert first == second
    assert len(first) == 32
    assert first != document_fingerprint(Path("other.txt"), 2, 3, "Sample policy chunk")


def test_lexical_rank_prioritizes_exact_keyword_match() -> None:
    from rag_utils import lexical_rank

    documents = [
        "Employees should contact People Operations for leave questions.",
        "The ETL pipeline must record row counts and run status.",
        "Remote work requests require manager approval.",
    ]
    ranking = lexical_rank("ETL pipeline row counts", documents)
    assert ranking[0] == 1


def test_lexical_rank_returns_empty_for_unmatched_query() -> None:
    from rag_utils import lexical_rank

    assert lexical_rank("zzzzunlikelytoken", ["policy information", "employee benefits"]) == []


def test_reciprocal_rank_fusion_combines_rankings() -> None:
    from rag_utils import reciprocal_rank_fusion

    result = reciprocal_rank_fusion([
        ["shared", "semantic-only"],
        ["shared", "lexical-only"],
    ])
    assert result[0] == "shared"
    assert set(result) == {"shared", "semantic-only", "lexical-only"}


def test_reciprocal_rank_fusion_validates_constant() -> None:
    from rag_utils import reciprocal_rank_fusion

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)


def test_plan_index_delta_add_remove_unchanged():
    from rag_utils import plan_index_delta
    delta = plan_index_delta({"keep", "remove"}, {"keep", "add"})
    assert delta == {"added": {"add"}, "removed": {"remove"}, "unchanged": {"keep"}}


def test_plan_index_delta_empty_index():
    from rag_utils import plan_index_delta
    delta = plan_index_delta(set(), {"one", "two"})
    assert delta["added"] == {"one", "two"}
    assert delta["removed"] == set()
    assert delta["unchanged"] == set()
