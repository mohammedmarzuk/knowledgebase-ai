"""Lightweight, dependency-free answer evaluation helpers.

Reference-token overlap is a lexical proxy, not a semantic correctness or factuality score.
"""
from __future__ import annotations

import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*", re.IGNORECASE)
CITATION_RE = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)
ABSTENTION_PHRASES = (
    "could not find sufficient evidence",
    "not specified in the indexed documents",
    "not specified in the documents",
    "i don't know based on the provided documents",
)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall((text or "").replace("-", " "))]


def token_overlap_scores(answer: str, reference: str) -> dict[str, float]:
    """Return token precision/recall/F1 against a reference answer.

    Repeated tokens are counted, but word order and paraphrase meaning are not
    understood. Treat these values as a rough regression signal only.
    """
    predicted = Counter(_tokens(answer))
    expected = Counter(_tokens(reference))
    if not predicted or not expected:
        return {"token_precision": 0.0, "token_recall": 0.0, "token_f1": 0.0}
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "token_precision": round(precision, 4),
        "token_recall": round(recall, 4),
        "token_f1": round(f1, 4),
    }


def citation_integrity(answer: str, source_count: int) -> dict[str, int | float]:
    """Check citation labels are in range and estimate factual-line citation coverage."""
    labels = [int(number) for number in CITATION_RE.findall(answer or "")]
    invalid = sum(1 for number in labels if number < 1 or number > max(0, source_count))
    lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]
    cited_lines = sum(1 for line in lines if CITATION_RE.search(line))
    return {
        "citation_count": len(labels),
        "invalid_citations": invalid,
        "citation_coverage": round(cited_lines / len(lines), 4) if lines else 0.0,
        "citation_integrity": 1 if labels and invalid == 0 else 0,
    }


def is_abstention(answer: str) -> bool:
    """Detect common explicit abstention phrases; this is a heuristic."""
    normalized = (answer or "").lower()
    return any(phrase in normalized for phrase in ABSTENTION_PHRASES)


def evaluate_answer(answer: str, reference: str, source_count: int) -> dict[str, int | float | bool]:
    """Combine lexical overlap, citation checks, and abstention detection."""
    return {
        **token_overlap_scores(answer, reference),
        **citation_integrity(answer, source_count),
        "abstained": is_abstention(answer),
    }
