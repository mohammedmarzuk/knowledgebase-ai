"""Evaluate retrieval against rag_test_questions.csv using source-document labels.

Run from the repository root after building the Chroma index:
    python scripts/evaluate_retrieval.py --k 5

This evaluates retrieval only; it does not claim to measure answer correctness.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import chromadb

from config import settings  # Load project-local .env before reading path overrides.
from rag_utils import lexical_rank, reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = Path(os.getenv("RAG_DB_DIR", str(ROOT / "chroma_db")))
DEFAULT_QUESTIONS = ROOT / "rag_test_questions.csv"
COLLECTION_NAME = "knowledge_base"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="Number of top chunks to evaluate")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")
    if not args.questions.exists():
        parser.error(f"Questions file not found: {args.questions}")

    client = chromadb.PersistentClient(path=str(args.db))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print("Knowledge base index not found. Start the app and build the index first.")
        return 2
    records = collection.get(include=["documents", "metadatas"])
    ids = records.get("ids") or []
    documents = records.get("documents") or []
    metadatas = records.get("metadatas") or []
    if not documents:
        print("Knowledge base index is empty. Build the index first.")
        return 2

    hits = 0
    reciprocal_ranks: list[float] = []
    evaluated = 0
    with args.questions.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        question = (row.get("question") or "").strip()
        expected_source = Path(row.get("source_document") or "").name
        if not question or not expected_source:
            continue
        evaluated += 1
        semantic = collection.query(query_texts=[question], n_results=min(args.k * 3, len(documents)))
        semantic_ids = (semantic.get("ids") or [[]])[0]
        lexical_order = lexical_rank(question, documents)
        lexical_ids = [ids[i] for i in lexical_order]
        fused = reciprocal_rank_fusion([semantic_ids, lexical_ids])[:args.k]
        source_by_id = {item_id: (meta or {}).get("source", "") for item_id, meta in zip(ids, metadatas)}
        relevant_ranks = [rank for rank, item_id in enumerate(fused, start=1)
                          if Path(source_by_id.get(item_id, "")).name == expected_source]
        if relevant_ranks:
            hits += 1
            reciprocal_ranks.append(1.0 / relevant_ranks[0])
        else:
            reciprocal_ranks.append(0.0)

    if evaluated == 0:
        print("No valid question/source_document rows found.")
        return 2
    print(f"Evaluated questions: {evaluated}")
    print(f"Source Hit Rate@{args.k}: {hits / evaluated:.3f} ({hits}/{evaluated})")
    print(f"Source MRR@{args.k}: {sum(reciprocal_ranks) / evaluated:.3f}")
    print("Note: these metrics measure source retrieval, not the factual correctness of generated answers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
