# KnowledgeBase AI v5 — Answer Quality Evaluation

## Added
- Answer evaluation helper functions for token precision/recall/F1, citation-label validation, and abstention detection.
- An opt-in Streamlit benchmark that evaluates generated answers using the existing `rag_test_questions.csv` reference answers.
- Progress reporting, a per-question results table, and explicit CSV export.
- Six unit tests for answer-evaluation helpers.

## Metrics caveats
- Token overlap is a lexical signal, not a semantic or factuality score.
- Citation-label validation checks that source labels are in range; it does not validate entailment.
- Abstention detection is phrase-based and may miss alternative wording.
- Evaluation calls the configured local Ollama model and may take several minutes.

## Validation
- `pytest -q`: 24 passed.
- Python compilation checks passed for application, utilities, scripts, and tests.
