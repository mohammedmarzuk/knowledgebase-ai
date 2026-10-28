import time
import pandas as pd
import streamlit as st
from workspace_core import load_benchmark, page_style, hero, retrieve

st.set_page_config(page_title="Evaluation Lab · KnowledgeBase AI", page_icon="🧪", layout="wide")
page_style()
hero("Evaluation Lab", "Benchmark retrieval against expected source documents. Use this lab to compare settings and catch regressions instead of relying on subjective examples.", "WORKSPACE · QUALITY ENGINEERING")
rows = load_benchmark()
if not rows:
    st.error("Benchmark file rag_test_questions.csv is missing or empty.")
    st.stop()
st.caption(f"Loaded {len(rows)} benchmark questions from rag_test_questions.csv. This evaluates retrieval quality, not the factual correctness of generated answers.")
col1, col2 = st.columns(2)
with col1: limit = st.slider("Questions to run", min_value=1, max_value=len(rows), value=min(10, len(rows)))
with col2: k = st.slider("Top-K cutoff", min_value=1, max_value=10, value=5)
if st.button("Run retrieval benchmark", type="primary"):
    records = []
    progress = st.progress(0, text="Preparing benchmark…")
    started = time.perf_counter()
    for idx, row in enumerate(rows[:limit]):
        question = row.get("question", "")
        expected = row.get("source_document", "").strip().casefold()
        try:
            results = retrieve(question, top_k=k)
            ranks = [n for n, result in enumerate(results, start=1) if str((result.get("metadata") or {}).get("source", "")).casefold() == expected]
            rank = ranks[0] if ranks else None
            records.append({"question": question, "expected_source": row.get("source_document", ""), "found": rank is not None, "rank": rank or "—", "reciprocal_rank": 1 / rank if rank else 0, "top_sources": ", ".join(str((x.get("metadata") or {}).get("source", "")) for x in results)})
        except Exception as exc:
            records.append({"question": question, "expected_source": row.get("source_document", ""), "found": False, "rank": "Error", "reciprocal_rank": 0, "top_sources": str(exc)})
        progress.progress((idx + 1) / min(limit, len(rows)), text=f"Evaluating {idx + 1}/{min(limit, len(rows))}")
    duration = time.perf_counter() - started
    st.session_state["retrieval_benchmark_results"] = records
    st.session_state["retrieval_benchmark_duration"] = duration

records = st.session_state.get("retrieval_benchmark_results", [])
if records:
    hit_rate = sum(bool(r["found"]) for r in records) / len(records)
    mrr = sum(float(r["reciprocal_rank"]) for r in records) / len(records)
    a, b, c = st.columns(3)
    a.metric(f"Hit Rate@{k}", f"{hit_rate:.1%}")
    b.metric(f"MRR@{k}", f"{mrr:.3f}")
    c.metric("Benchmark duration", f"{st.session_state.get('retrieval_benchmark_duration', 0):.1f}s")
    st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
    st.download_button("Download benchmark CSV", pd.DataFrame(records).to_csv(index=False).encode("utf-8"), file_name="retrieval_benchmark_results.csv", mime="text/csv")
    st.caption("Hit Rate@K is the share of questions whose expected source appears in the top K results. MRR@K rewards higher ranks. Scores depend on the benchmark's expected-source labels.")
