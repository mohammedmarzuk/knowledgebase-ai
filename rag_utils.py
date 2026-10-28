"""Reusable RAG utilities: chunking, stable IDs, lexical scoring, and rank fusion."""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from collections import Counter
from pathlib import Path

CHUNK_WORDS = 180
CHUNK_OVERLAP = 35
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_'-]*")


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word chunks, validating configuration."""
    if size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    if overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")
    if overlap >= size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks


def document_fingerprint(path: Path, page: int | None, chunk_no: int, chunk: str) -> str:
    """Return a deterministic ID that changes when the chunk content changes."""
    raw = f"{path.name}|{page}|{chunk_no}|{chunk}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]



def plan_index_delta(existing_ids: set[str], desired_ids: set[str]) -> dict[str, set[str]]:
    """Return stable chunk IDs to add, remove, or reuse during incremental indexing."""
    return {
        "added": desired_ids - existing_ids,
        "removed": existing_ids - desired_ids,
        "unchanged": existing_ids & desired_ids,
    }

@lru_cache(maxsize=100_000)
def tokenize(text: str) -> list[str]:
    """Normalize and cache text tokens to speed repeated lexical retrieval."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def lexical_rank(query: str, documents: list[str]) -> list[int]:
    """Rank document indices using a small BM25-inspired scorer.

    This dependency-free scorer is designed as a complementary signal to vector
    search, not as a replacement for a production BM25 index.
    """
    query_terms = set(tokenize(query))
    if not query_terms or not documents:
        return []
    tokenized = [tokenize(doc) for doc in documents]
    token_sets = [set(tokens) for tokens in tokenized]
    avg_len = sum(map(len, tokenized)) / max(len(tokenized), 1)
    doc_freq = Counter(term for term in query_terms for terms in token_sets if term in terms)
    n_docs = len(tokenized)
    scores: list[tuple[float, int]] = []
    for index, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        length_norm = 1.5 * (1 - 0.75 + 0.75 * len(tokens) / max(avg_len, 1))
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            if not frequency:
                continue
            idf = math.log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            score += idf * (frequency * 2.5) / (frequency + length_norm)
        if score > 0:
            scores.append((score, index))
    scores.sort(key=lambda pair: (-pair[0], pair[1]))
    return [index for _, index in scores]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Combine ranked ID lists without requiring scores to share a scale."""
    if k <= 0:
        raise ValueError("Fusion constant k must be positive.")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item_id: (-scores[item_id], item_id))


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def rerank_with_cross_encoder(query: str, items: list[dict], model_name: str) -> list[dict]:
    """Optionally rerank retrieved passages with a CrossEncoder.

    sentence-transformers is an optional dependency. The base app remains lightweight
    and works without it; install requirements-advanced.txt to enable this feature.
    """
    if not items:
        return []
    try:
        model = _load_cross_encoder(model_name)
    except ImportError as exc:
        raise RuntimeError(
            "Cross-encoder reranking requires sentence-transformers. "
            "Install it with: pip install -r requirements-advanced.txt"
        ) from exc

    pairs = [(query, item["text"]) for item in items]
    scores = model.predict(pairs, show_progress_bar=False)
    reranked = []
    for item, score in zip(items, scores):
        copy = dict(item)
        copy["rerank_score"] = float(score)
        copy["retrieval_method"] = f"hybrid + cross-encoder ({model_name})"
        reranked.append(copy)
    return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)
