import json
import pandas as pd
import streamlit as st
from workspace_core import ANALYTICS_PATH, get_collection, get_installed_models, load_analytics, page_style, hero, metric

st.set_page_config(page_title="Performance · KnowledgeBase AI", page_icon="⚡", layout="wide")
page_style()
hero("Performance center", "Monitor local model availability, index size, response latency, and user feedback. Telemetry is read from the local analytics log.", "WORKSPACE · OBSERVABILITY")
collection = get_collection()
count = collection.count() if collection is not None else 0
models, model_error = get_installed_models()
events = load_analytics()
answers = [e for e in events if e.get("event") == "answer"]
feedback = [e for e in events if e.get("event") == "feedback"]
avg_total = sum(float(e.get("elapsed_seconds", 0)) for e in answers) / len(answers) if answers else 0
avg_retrieval = sum(float(e.get("retrieval_seconds", 0)) for e in answers) / len(answers) if answers else 0
positive = sum(1 for e in feedback if int(e.get("rating", 0)) > 0)
negative = sum(1 for e in feedback if int(e.get("rating", 0)) < 0)
cols = st.columns(4)
with cols[0]: metric("Indexed passages", f"{count:,}", "Current Chroma collection")
with cols[1]: metric("Answered questions", f"{len(answers):,}", "Locally recorded events")
with cols[2]: metric("Average response", f"{avg_total:.2f}s", f"Retrieval average {avg_retrieval:.2f}s")
with cols[3]: metric("Feedback", f"{positive} 👍 / {negative} 👎", "Ratings saved")
st.subheader("Local model health")
if model_error:
    st.warning("Ollama is not responding. Start Ollama and refresh this page.")
else:
    st.success(f"Ollama responded. {len(models)} installed model(s) detected.")
    st.write(" · ".join(f"`{name}`" for name in models) if models else "No models detected.")
if answers:
    st.subheader("Latency trend")
    df = pd.DataFrame([{"request": i + 1, "Total response (s)": float(e.get("elapsed_seconds", 0)), "Retrieval (s)": float(e.get("retrieval_seconds", 0)), "Model": e.get("model", "unknown")} for i, e in enumerate(answers[-50:])])
    st.line_chart(df.set_index("request")[["Total response (s)", "Retrieval (s)"]])
    st.subheader("Model usage")
    st.bar_chart(df["Model"].value_counts())
    with st.expander("Recent telemetry"):
        st.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)
else:
    st.info("No answer telemetry yet. Ask questions on the Chat page to populate this dashboard.")
if ANALYTICS_PATH.exists():
    st.download_button("Export local analytics JSONL", ANALYTICS_PATH.read_bytes(), file_name="analytics_events.jsonl", mime="application/x-ndjson")
with st.expander("Telemetry privacy"):
    st.write("The existing analytics logger is designed to store performance metadata rather than question and answer text. The exported file may contain timestamps, model names, timing, source counts, and derived quality indicators. Review it before sharing publicly.")
