"""Persistent, content-addressed document intelligence helpers."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ingestion import extract_text

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", str(APP_DIR / "data")))
INSIGHTS_DIR = Path(os.getenv("RAG_INSIGHTS_DIR", str(APP_DIR / "runtime" / "insights")))
HISTORY_PATH = Path(os.getenv("RAG_JOB_HISTORY_PATH", str(APP_DIR / "runtime" / "job_history.json")))


def file_sha256(path: Path) -> str:
    """Hash a file in bounded-memory blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_document_text(path: Path, max_chars: int = 100_000) -> str:
    """Extract and join document pages/rows, bounded for preview and summarization."""
    pages = extract_text(path)
    parts = []
    total = 0
    for page_number, content in pages:
        content = (content or "").strip()
        if not content:
            continue
        label = f"[Page/row {page_number}] " if page_number is not None else ""
        remaining = max_chars - total
        if remaining <= 0:
            break
        piece = (label + content)[:remaining]
        parts.append(piece)
        total += len(piece)
    return "\\n\\n".join(parts)


def _summary_path(digest: str) -> Path:
    return INSIGHTS_DIR / f"{digest}.json"


def load_cached_summary(path: Path) -> dict[str, Any] | None:
    """Return a summary only if it belongs to the current file contents."""
    digest = file_sha256(path)
    cache = _summary_path(digest)
    if not cache.exists():
        return None
    try:
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if payload.get("sha256") == digest:
            return payload
    except (OSError, ValueError, TypeError):
        return None
    return None


def save_summary(path: Path, summary: str, model: str, source_chars: int) -> dict[str, Any]:
    """Atomically persist a summary keyed to file content, not filename."""
    digest = file_sha256(path)
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "filename": path.name,
        "sha256": digest,
        "summary": summary.strip(),
        "model": model,
        "source_chars": source_chars,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    target = _summary_path(digest)
    fd, temp_name = tempfile.mkstemp(prefix=".summary-", suffix=".tmp", dir=INSIGHTS_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return payload


def append_job_event(event: dict[str, Any], history_path: Path | None = None, keep: int = 100) -> None:
    """Persist bounded local job history with atomic replacement."""
    target = history_path or HISTORY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    events = []
    if target.exists():
        try:
            events = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(events, list):
                events = []
        except (OSError, ValueError):
            events = []
    event = dict(event)
    event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    events.append(event)
    events = events[-keep:]
    fd, temp_name = tempfile.mkstemp(prefix=".history-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(events, handle, ensure_ascii=False, indent=2)
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_job_history(history_path: Path | None = None) -> list[dict[str, Any]]:
    target = history_path or HISTORY_PATH
    if not target.exists():
        return []
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
        return value[-100:] if isinstance(value, list) else []
    except (OSError, ValueError):
        return []
