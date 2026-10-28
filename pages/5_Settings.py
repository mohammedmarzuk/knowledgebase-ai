import os
import streamlit as st
from config import settings
from workspace_core import get_collection, get_installed_models, page_style, hero, metric

st.set_page_config(page_title="Settings · KnowledgeBase AI", page_icon="⚙️", layout="wide")
page_style()
hero("Workspace settings", "Review local configuration, check dependencies, and understand the performance and privacy trade-offs of your RAG workspace.", "WORKSPACE · CONFIGURATION")

collection = get_collection()
models, error = get_installed_models()
cols = st.columns(3)
with cols[0]: metric("Ollama host", settings.ollama_host.replace("http://", "").replace("https://", ""), "Configured endpoint")
with cols[1]: metric("Default model", settings.chat_model, "Set by OLLAMA_CHAT_MODEL")
with cols[2]: metric("Upload limit", f"{settings.max_upload_mb} MB", "Per uploaded file")

st.subheader("Retrieval configuration")
config_rows = [
    {"Setting": "Default retrieved passages", "Value": settings.default_top_k, "Environment variable": "RAG_TOP_K"},
    {"Setting": "Chunk size (words)", "Value": settings.chunk_words, "Environment variable": "RAG_CHUNK_WORDS"},
    {"Setting": "Chunk overlap (words)", "Value": settings.chunk_overlap, "Environment variable": "RAG_CHUNK_OVERLAP"},
    {"Setting": "Candidate multiplier", "Value": settings.candidate_multiplier, "Environment variable": "RAG_CANDIDATE_MULTIPLIER"},
    {"Setting": "Maximum context characters", "Value": settings.max_context_chars, "Environment variable": "RAG_MAX_CONTEXT_CHARS"},
    {"Setting": "Reranker model", "Value": settings.reranker_model or "Disabled", "Environment variable": "RAG_RERANKER_MODEL"},
    {"Setting": "Streaming default", "Value": settings.streaming_default, "Environment variable": "RAG_STREAMING_DEFAULT"},
    {"Setting": "Analytics enabled", "Value": settings.analytics_enabled, "Environment variable": "RAG_ANALYTICS_ENABLED"},
]
st.dataframe(config_rows, use_container_width=True, hide_index=True)
st.caption("Change environment variables before launching Streamlit to alter these settings. The page is read-only so configuration changes remain explicit and reproducible.")

st.subheader("Runtime health")
if error:
    st.error(f"Ollama is not responding at the configured endpoint: {error}")
else:
    st.success(f"Ollama is connected. {len(models)} model(s) detected.")
    if models:
        st.write("Installed models: " + " · ".join(f"`{name}`" for name in models))
if collection is None:
    st.info("No Chroma collection found yet. Add documents and build the knowledge base from the Chat page.")
else:
    st.write(f"Indexed chunks: **{collection.count():,}**")

with st.expander("Performance guidance"):
    st.markdown("""
    - Smaller `RAG_TOP_K` and `RAG_MAX_CONTEXT_CHARS` values usually reduce local generation latency, but may omit evidence.
    - Incremental indexing reuses embeddings for stable chunk IDs; changing chunk size or overlap changes chunk IDs and can require a full re-embedding.
    - Cross-encoder reranking can improve ordering but adds CPU/RAM cost and a one-time model download.
    - Streaming improves perceived speed, not necessarily total generation time.
    - For large corpora, replace in-memory lexical scoring with a persistent BM25 index.
    """)

with st.expander("Privacy and deployment boundary"):
    st.warning("This project is designed for a trusted, local, single-user environment. It does not provide authentication, per-user document permissions, encrypted storage, or multi-tenant isolation. Do not expose it directly to the public internet without adding those controls.")
    st.write("Document contents and the Chroma index are stored locally. Ollama generates answers using the configured local or remote Ollama endpoint. Package and model downloads may use the internet.")
