"""Privacy-conscious local analytics and retrieval-evaluation helpers."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def citation_coverage(answer: str) -> float:
    """Return fraction of non-empty answer lines containing a [Source N] citation.

    This is a lightweight heuristic, not a factuality or entailment metric.
    """
    lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]
    if not lines:
        return 0.0
    cited = sum("[source " in line.lower() for line in lines)
    return cited / len(lines)


def build_answer_event(*, model: str, elapsed_seconds: float, retrieval_seconds: float,
                       source_count: int, answer: str, feedback: int | None = None) -> dict[str, Any]:
    """Build an analytics event without saving question or answer text."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "answer",
        "model": model,
        "elapsed_seconds": round(max(0.0, float(elapsed_seconds)), 4),
        "retrieval_seconds": round(max(0.0, float(retrieval_seconds)), 4),
        "source_count": max(0, int(source_count)),
        "answer_chars": len(answer or ""),
        "citation_coverage": round(citation_coverage(answer), 4),
        "feedback": feedback,
    }


def append_event(path: Path, event: dict[str, Any]) -> None:
    """Append one JSON object per line, creating the parent folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_events(path: Path) -> list[dict[str, Any]]:
    """Read valid JSONL events; ignore malformed/incomplete trailing lines."""
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    events.append(item)
            except (json.JSONDecodeError, TypeError):
                continue
    return events


def retrieval_metrics(rows: Iterable[dict[str, Any]], k: int = 5) -> dict[str, float | int]:
    """Compute source Hit Rate@K and MRR@K from ranked source-name lists.

    Each row requires ``expected_source`` and ``retrieved_sources`` (best first).
    Source names are compared by basename to tolerate relative/absolute paths.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    evaluated = 0
    hits = 0
    reciprocal_sum = 0.0
    for row in rows:
        expected = Path(str(row.get("expected_source", ""))).name
        sources = row.get("retrieved_sources", []) or []
        if not expected:
            continue
        evaluated += 1
        rank = next((i for i, source in enumerate(sources[:k], start=1)
                     if Path(str(source)).name == expected), None)
        if rank is not None:
            hits += 1
            reciprocal_sum += 1.0 / rank
    return {
        "evaluated": evaluated,
        "hit_rate_at_k": hits / evaluated if evaluated else 0.0,
        "mrr_at_k": reciprocal_sum / evaluated if evaluated else 0.0,
        "hits": hits,
        "k": k,
    }
