from pathlib import Path
import streamlit as st
from workspace_core import DATA_DIR, SUPPORTED_EXTENSIONS, get_collection, list_documents, page_style, hero
from config import settings
from upload_utils import unique_destination

st.set_page_config(page_title="Documents · KnowledgeBase AI", page_icon="📚", layout="wide")
page_style()
hero("Document library", "Manage the files that power your private knowledge assistant. Add new sources, inspect the library, and keep the index in sync.", "WORKSPACE · DOCUMENTS")

files = list_documents()
collection = get_collection()
chunk_count = collection.count() if collection is not None else 0
c1, c2, c3 = st.columns(3)
with c1: st.metric("Documents", len(files))
with c2: st.metric("Indexed passages", chunk_count)
with c3: st.metric("Supported formats", len(SUPPORTED_EXTENSIONS))

st.subheader("Add documents")
uploads = st.file_uploader("Upload files", type=[x.lstrip(".") for x in sorted(SUPPORTED_EXTENSIONS)], accept_multiple_files=True, help="TXT, Markdown, PDF, DOCX, and CSV are supported.")
if st.button("Save uploaded files", type="primary", disabled=not uploads):
    saved, skipped = 0, 0
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rejected = []
    for item in uploads or []:
        name = Path(item.name).name
        payload = item.getvalue()
        if not name or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped += 1
            rejected.append(f"{name or 'Unnamed file'}: unsupported file type")
            continue
        if len(payload) > settings.max_upload_mb * 1024 * 1024:
            skipped += 1
            rejected.append(f"{name}: exceeds {settings.max_upload_mb} MB")
            continue
        (DATA_DIR / name).write_bytes(payload)
        saved += 1
    if saved: st.success(f"Saved {saved} file(s). Return to the Chat page and select **Build / refresh knowledge base** to index them.")
    if skipped: st.warning(f"Skipped {skipped} file(s).")
    for reason in rejected: st.caption(reason)
    st.rerun()

st.subheader("Library")
if not files:
    st.info("No documents yet. Upload files above or add them to the project's data folder.")
else:
    query = st.text_input("Filter documents", placeholder="Search by filename…")
    filtered = [p for p in files if query.casefold() in p.name.casefold()]
    for path in filtered:
        with st.container(border=True):
            left, mid, right = st.columns([5, 2, 1])
            left.markdown(f"**📄 {path.name}**")
            left.caption(f"{path.stat().st_size / 1024:.1f} KB · {path.suffix.upper().lstrip('.')} file")
            mid.caption("Stored locally")
            if right.button("Delete", key=f"delete_{path.name}"):
                path.unlink(missing_ok=True)
                st.warning("File deleted. Rebuild the index from the Chat page to sync the search index.")
                st.rerun()
with st.expander("Indexing notes"):
    st.write("The current index may still contain chunks from deleted or modified files until you rebuild it. Scanned PDFs without a text layer may require OCR before they can be searched.")
