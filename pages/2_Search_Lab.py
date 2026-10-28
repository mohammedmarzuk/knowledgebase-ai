import html
import time
import streamlit as st
from workspace_core import get_collection, page_style, hero, retrieve

st.set_page_config(page_title="Search Lab · KnowledgeBase AI", page_icon="🔎", layout="wide")
page_style()
hero("Search Lab", "Inspect the retrieval layer independently from answer generation. Compare hybrid search with optional cross-encoder reranking and see exactly which passages rank highest.", "WORKSPACE · RETRIEVAL ENGINEERING")
collection = get_collection()
if collection is None or collection.count() == 0:
    st.warning("The knowledge base is empty. Open the Chat page and build the index first.")
    st.stop()

left, right = st.columns([3, 1])
with left:
    query = st.text_input("Search query", placeholder="e.g. What is the data retention policy?", key="search_query")
with right:
    top_k = st.slider("Results", 3, 12, 5)
use_reranker = st.toggle("Enable cross-encoder reranking", value=False, help="Requires the optional sentence-transformers dependency. The first run may download model weights.")
if use_reranker:
    st.caption("Reranking can improve ordering but increases latency and memory use on a local machine.")
if st.button("Run search", type="primary", disabled=not query.strip()):
    try:
        start = time.perf_counter()
        results = retrieve(query, top_k=top_k, use_reranker=use_reranker)
        elapsed = time.perf_counter() - start
        st.session_state["search_lab_results"] = results
        st.session_state["search_lab_elapsed"] = elapsed
        st.session_state["search_lab_query"] = query
    except Exception as exc:
        st.error(str(exc))

results = st.session_state.get("search_lab_results", [])
if results:
    st.markdown(f"**{len(results)} result(s)** for _{st.session_state.get('search_lab_query', '')}_ · {st.session_state.get('search_lab_elapsed', 0):.3f}s")
    for i, item in enumerate(results, start=1):
        meta = item.get("metadata") or {}
        source = html.escape(str(meta.get("source", "Unknown source")))
        location = f"Page {meta['page']}" if meta.get("page") is not None else f"Chunk {meta.get('chunk', '?')}"
        score = item.get("rerank_score")
        with st.container(border=True):
            rank_col, info_col = st.columns([1, 9])
            rank_col.markdown(f"### #{i}")
            info_col.markdown(f"**{source}** · {location}")
            info_col.caption(f"Method: {item.get('retrieval_method', 'hybrid')}" + (f" · Rerank score: {score:.4f}" if score is not None else "") + (f" · Vector distance: {item['distance']:.4f}" if item.get("distance") is not None else ""))
            st.write(item.get("text", ""))
else:
    st.info("Enter a query and run search to inspect retrieved passages.")
