"""Shared read-only helpers for the KnowledgeBase AI multi-page workspace."""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from config import settings  # Loads .env before the Ollama client is imported.

import chromadb
import ollama
import streamlit as st

from analytics_utils import load_events
from rag_utils import lexical_rank, reciprocal_rank_fusion, rerank_with_cross_encoder

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", str(APP_DIR / "data")))
DB_DIR = Path(os.getenv("RAG_DB_DIR", str(APP_DIR / "chroma_db")))
ANALYTICS_PATH = Path(os.getenv("RAG_ANALYTICS_PATH", str(APP_DIR / "analytics_events.jsonl")))
COLLECTION_NAME = "knowledge_base"
DEFAULT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:4b")
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".csv"}


def get_collection():
    """Return the existing collection without creating an empty index."""
    if not DB_DIR.exists():
        return None
    try:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return None



@st.cache_data(ttl=300, show_spinner=False)
def _cached_corpus_records(collection_count: int) -> tuple[list[str], list[str], list[dict]]:
    """Cache corpus text and metadata for repeated lexical searches."""
    collection = get_collection()
    if collection is None or collection.count() != collection_count:
        return [], [], []
    records = collection.get(include=["documents", "metadatas"])
    return records.get("ids") or [], records.get("documents") or [], records.get("metadatas") or []

def list_documents() -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def get_installed_models() -> tuple[list[str], str | None]:
    try:
        result = ollama.list()
        names = []
        for model in result.models:
            name = getattr(model, "model", None) or getattr(model, "name", None)
            if name:
                names.append(name)
        return sorted(set(names)), None
    except Exception as exc:
        return [], str(exc)


def retrieve(question: str, top_k: int = 5, use_reranker: bool = False) -> list[dict[str, Any]]:
    """Run hybrid retrieval against the existing Chroma collection."""
    question = question.strip()
    if not question:
        raise ValueError("Enter a search query first.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    collection = get_collection()
    if collection is None or collection.count() == 0:
        raise ValueError("The knowledge base is empty. Open the Chat page and build the index first.")
    count = collection.count()
    semantic = collection.query(query_texts=[question], n_results=min(count, max(top_k * settings.candidate_multiplier, top_k)), include=["documents", "metadatas", "distances"])
    sem_ids = (semantic.get("ids") or [[]])[0]
    sem_docs = (semantic.get("documents") or [[]])[0]
    sem_meta = (semantic.get("metadatas") or [[]])[0]
    sem_dist = (semantic.get("distances") or [[]])[0]
    all_ids, all_docs, all_meta = _cached_corpus_records(count)
    lexical_indices = lexical_rank(question, all_docs)
    lexical_ids = [all_ids[i] for i in lexical_indices]
    fused = reciprocal_rank_fusion([sem_ids, lexical_ids])
    lookup = {i: {"text": d, "metadata": m or {}, "distance": None} for i, d, m in zip(all_ids, all_docs, all_meta)}
    lookup.update({i: {"text": d, "metadata": m or {}, "distance": dist} for i, d, m, dist in zip(sem_ids, sem_docs, sem_meta, sem_dist)})
    candidates = []
    for rank, item_id in enumerate(fused[: min(len(fused), top_k * settings.candidate_multiplier)], 1):
        item = dict(lookup[item_id])
        item["id"] = item_id
        item["rank"] = rank
        item["retrieval_method"] = "hybrid (semantic + lexical)"
        candidates.append(item)
    if use_reranker:
        model_name = os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        candidates = rerank_with_cross_encoder(question, candidates, model_name)
    return candidates[:top_k]


def load_benchmark(path: Path | None = None) -> list[dict[str, str]]:
    path = path or APP_DIR / "rag_test_questions.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_analytics() -> list[dict[str, Any]]:
    return load_events(ANALYTICS_PATH)


def page_style() -> None:
    """Shared visual language for the secondary workspace pages."""
    import streamlit as st
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    .stApp{background:#f6f8fc;color:#172033;font-family:'DM Sans',sans-serif}
    [data-testid="stSidebar"]{background:#fff;border-right:1px solid #e7eaf0}
    .block-container{max-width:1400px;padding-top:1.8rem;padding-bottom:3rem}
    h1,h2,h3{font-family:'Manrope',sans-serif!important;letter-spacing:-.035em;color:#172033}
    h1{font-weight:800!important}
    .page-hero{padding:27px 30px;border-radius:23px;background:linear-gradient(115deg,#172554,#293c92 55%,#5146d8);color:white;margin-bottom:22px;box-shadow:0 14px 32px #1e2b6517}
    .page-kicker{font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:#c7d2fe;font-weight:800;margin-bottom:9px}
    .page-title{font:800 1.8rem 'Manrope',sans-serif;color:white;letter-spacing:-.04em}
    .page-copy{color:#dbe4ff;margin-top:8px;max-width:800px;line-height:1.6}
    .metric-card{background:white;border:1px solid #e7eaf0;border-radius:17px;padding:17px 18px;min-height:105px;box-shadow:0 3px 10px #10182806}
    .metric-label{font-size:.78rem;color:#667085;font-weight:700}.metric-value{font:800 1.6rem 'Manrope',sans-serif;color:#172033;margin-top:8px}.metric-note{font-size:.73rem;color:#98a2b3;margin-top:5px}
    .content-card{background:white;border:1px solid #e7eaf0;border-radius:17px;padding:19px 21px;margin:10px 0;box-shadow:0 3px 10px #10182804}
    .stButton>button{border-radius:11px;font-weight:700;min-height:40px}
    [data-testid="stFileUploader"]{border:1px dashed #cfd6e4;border-radius:14px;background:#fff}
    </style>""", unsafe_allow_html=True)


def hero(title: str, subtitle: str, kicker: str = "KNOWLEDGEBASE AI WORKSPACE") -> None:
    import streamlit as st
    st.markdown(f'<div class="page-hero"><div class="page-kicker">{kicker}</div><div class="page-title">{title}</div><div class="page-copy">{subtitle}</div></div>', unsafe_allow_html=True)


def metric(label: str, value: str, note: str = "") -> None:
    import streamlit as st
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)
