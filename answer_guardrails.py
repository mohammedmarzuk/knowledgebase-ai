"""Lightweight checks for source-label integrity and evidence-aware answers.

These checks improve transparency but do not prove factuality or entailment.
"""
from __future__ import annotations

import re
from typing import Any

CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)
ABSTENTION_TEXT = (
    "i could not find sufficient evidence",
    "not specified in the indexed documents",
    "not specified in the documents",
    "i don't know based on the provided documents",
)


def inspect_answer(answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(value) for value in CITATION_PATTERN.findall(answer or "")]
    source_count = len(sources)
    invalid = sorted({label for label in labels if label < 1 or label > source_count})
    lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]
    cited_lines = sum(bool(CITATION_PATTERN.search(line)) for line in lines)
    normalized = (answer or "").casefold()
    abstained = any(phrase in normalized for phrase in ABSTENTION_TEXT)
    return {
        "citation_count": len(labels),
        "invalid_citations": invalid,
        "citation_coverage": round(cited_lines / len(lines), 4) if lines else 0.0,
        "has_citations": bool(labels),
        "abstained": abstained,
        "source_count": source_count,
        "warnings": _warnings(labels, invalid, source_count, abstained),
    }


def _warnings(labels: list[int], invalid: list[int], source_count: int, abstained: bool) -> list[str]:
    warnings: list[str] = []
    if invalid:
        warnings.append("One or more source labels refer to passages that were not retrieved. Verify the answer before relying on it.")
    if source_count and not labels and not abstained:
        warnings.append("The answer contains no recognizable source labels. Check the source passages manually.")
    if source_count == 0 and not abstained:
        warnings.append("No evidence passages were available for this answer.")
    return warnings
