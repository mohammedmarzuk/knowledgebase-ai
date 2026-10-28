from pathlib import Path
import time
import streamlit as st
import ollama

from config import settings
from document_intelligence import (
    append_job_event, extract_document_text, file_sha256,
    load_cached_summary, load_job_history, save_summary,
)
from workspace_core import (
    DATA_DIR, SUPPORTED_EXTENSIONS, get_collection, get_installed_models,
    list_documents, page_style, hero, metric,
)

st.set_page_config(page_title="Document Intelligence · KnowledgeBase AI", page_icon="🧠", layout="wide")
page_style()
hero(
    "Document intelligence",
    "Create persistent AI summaries, inspect extracted text, and review recent indexing and analysis events. Summaries are cached against document content, so unchanged files do not need to be summarized again.",
    "WORKSPACE · DOCUMENT INTELLIGENCE",
)

files = list_documents()
collection = get_collection()
chunk_count = collection.count() if collection is not None else 0
history = load_job_history()
summaries = [load_cached_summary(path) for path in files]
summaries = [item for item in summaries if item]
cols = st.columns(4)
with cols[0]: metric("Documents", f"{len(files):,}", "Supported files in library")
with cols[1]: metric("Indexed passages", f"{chunk_count:,}", "Current vector collection")
with cols[2]: metric("Summaries cached", f"{len(summaries):,}", "Content-addressed summaries")
with cols[3]: metric("Recent events", f"{len(history):,}", "Local history, max 100")

if not files:
    st.info("Upload documents from the Documents page before generating summaries.")
    st.stop()

models, model_error = get_installed_models()
default_model = settings.chat_model
if models:
    default_index = next((i for i, name in enumerate(models) if name == default_model or name.startswith(default_model + ":")), 0)
    model = st.selectbox("Summary model", models, index=default_index)
else:
    model = default_model
    st.warning("Ollama models could not be listed. Check that Ollama is running and the selected model is installed.")

st.subheader("Generate document summaries")
st.caption("Summaries use extracted document text and your local Ollama model. Large documents are bounded to 100,000 characters per summary request to limit memory use and latency.")
options = [p.name for p in files]
selected = st.selectbox("Choose a document", options)
selected_path = next(p for p in files if p.name == selected)
cached = load_cached_summary(selected_path)
if cached:
    st.success(f"A saved summary exists for this version of the document · {cached.get('generated_at', '')}")
    with st.expander("View saved summary", expanded=True):
        st.markdown(cached.get("summary", ""))
        st.caption(f"Model: {cached.get('model', 'unknown')} · Source characters: {cached.get('source_chars', 0):,}")
else:
    st.info("No summary exists for this document version yet.")

c1, c2 = st.columns([1, 2])
with c1:
    regenerate = st.button("Generate / refresh summary", type="primary", use_container_width=True)
with c2:
    preview = st.button("Preview extracted text", use_container_width=True)

if preview:
    try:
        text = extract_document_text(selected_path, max_chars=16000)
        if text.strip():
            st.text_area("Extracted text preview (first 16,000 characters)", text, height=280)
        else:
            st.warning("No selectable text was extracted. Scanned PDFs may require optional OCR dependencies and system tools.")
    except Exception as exc:
        st.error(f"Could not extract text: {exc}")

if regenerate:
    started = time.perf_counter()
    progress = st.progress(0, text="Extracting document text…")
    try:
        source_text = extract_document_text(selected_path, max_chars=100_000)
        if not source_text.strip():
            raise ValueError("No text could be extracted from this document. For scanned PDFs, enable OCR first.")
        progress.progress(0.45, text="Generating summary with local Ollama…")
        prompt = (
            "You are a careful document analyst. Summarize ONLY the provided source. "
            "Do not invent facts. Use headings: Executive summary, Key points, Important dates/numbers, "
            "Actions or requirements, and Unclear or missing information. If a heading has no evidence, say so. "
            "Keep the result concise but useful. Distinguish explicit facts from reasonable interpretations.\\n\\n"
            f"DOCUMENT NAME: {selected_path.name}\\nSOURCE TEXT:\\n{source_text}"
        )
        response = ollama.generate(model=model, prompt=prompt, options={"temperature": 0.1})
        summary_text = (response.get("response") or "").strip()
        if not summary_text:
            raise RuntimeError("The model returned an empty summary.")
        payload = save_summary(selected_path, summary_text, model, len(source_text))
        append_job_event({
            "kind": "document_summary", "status": "completed",
            "filename": selected_path.name, "sha256": payload["sha256"],
            "model": model, "elapsed_seconds": round(time.perf_counter() - started, 3),
        })
        progress.progress(1.0, text="Summary saved")
        st.success("Summary generated and saved locally.")
        st.markdown(summary_text)
        st.caption(f"Completed in {time.perf_counter() - started:.2f}s · Model: {model}")
    except Exception as exc:
        append_job_event({
            "kind": "document_summary", "status": "failed",
            "filename": selected_path.name, "model": model,
            "error": str(exc)[:500], "elapsed_seconds": round(time.perf_counter() - started, 3),
        })
        st.error(f"Summary generation failed: {exc}")
    finally:
        progress.empty()

st.divider()
st.subheader("Recent local processing history")
if history:
    for event in reversed(history[-10:]):
        status = event.get("status", "recorded")
        icon = "✅" if status == "completed" else "⚠️" if status == "failed" else "•"
        label = event.get("filename") or event.get("kind") or "Processing event"
        st.markdown(f"{icon} **{label}** — `{status}` · {event.get('timestamp', '')}")
        if event.get("error"):
            st.caption(event["error"])
else:
    st.caption("No document-analysis events have been recorded yet.")
