"""Environment-backed configuration for KnowledgeBase AI.

All defaults are safe for a local, single-user development workspace. Set values
in the shell or copy .env.example to .env and export them with your shell tooling.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load optional project-local settings before reading any environment defaults.
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if minimum <= value <= maximum else default


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:4b")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    default_top_k: int = _int_env("RAG_TOP_K", 3, minimum=1, maximum=12)
    chunk_words: int = _int_env("RAG_CHUNK_WORDS", 180, minimum=40, maximum=1200)
    chunk_overlap: int = _int_env("RAG_CHUNK_OVERLAP", 35, minimum=0, maximum=300)
    candidate_multiplier: int = _int_env("RAG_CANDIDATE_MULTIPLIER", 4, minimum=1, maximum=12)
    max_upload_mb: int = _int_env("MAX_UPLOAD_MB", 500, minimum=1, maximum=500)
    max_context_chars: int = _int_env("RAG_MAX_CONTEXT_CHARS", 18000, minimum=2000, maximum=100000)
    reranker_model: str = os.getenv("RAG_RERANKER_MODEL", "").strip()
    streaming_default: bool = _bool_env("RAG_STREAMING_DEFAULT", True)
    analytics_enabled: bool = _bool_env("RAG_ANALYTICS_ENABLED", True)


settings = Settings()

if settings.chunk_overlap >= settings.chunk_words:
    # Avoid invalid chunking configuration if the environment overrides defaults.
    settings = Settings(chunk_overlap=max(0, settings.chunk_words // 5))
