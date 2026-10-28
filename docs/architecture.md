# Architecture

KnowledgeBase AI uses a local Retrieval-Augmented Generation (RAG) workflow.

```text
TXT / Markdown / PDF / DOCX
            |
            v
     Text extraction
            |
            v
  Word-based chunking
            |
            v
 ChromaDB vector collection
            |
User question --> semantic retrieval
                         |
                         v
               Retrieved passages
                         |
                         v
                  Ollama chat model
                         |
                         v
              Answer with source context
```

## Components

- `app.py`: Streamlit UI, upload workflow, indexing controls, retrieval orchestration, and answer display.
- `rag_utils.py`: Dependency-light chunking, deterministic chunk IDs, lexical ranking, rank fusion, and optional reranking.
- `ingestion.py`: TXT, Markdown, PDF, DOCX, and row-aware CSV extraction plus document-size validation.
- `config.py`: Environment and project-local `.env` settings.
- `answer_guardrails.py`: Heuristic source-label checks and warnings.
- `workspace_core.py`: Shared retrieval helpers and corpus cache for the multi-page workspace.
- `data/`: Fictional sample documents used as the initial knowledge base.
- `chroma_db/`: Runtime vector index created locally; intentionally excluded from Git.
- `tests/`: Unit tests for chunking and deterministic IDs.

## Indexing behavior

The app extracts text from supported files, splits it into chunks using the configured word size and overlap, and stores the chunks in a persistent ChromaDB collection. ChromaDB's default embedding function is used by this prototype. Ask-time retrieval combines semantic search with lexical ranking and Reciprocal Rank Fusion, then supplies selected passages to the configured Ollama model.

## Known limitations

- Scanned PDFs need OCR before their text can be indexed.
- Incremental synchronization reuses stable chunk IDs; changing chunking settings can invalidate many IDs and require re-embedding.
- Retrieval quality depends on the document text, chunk boundaries, embedding model, and `top_k` setting.
- The application has no authentication or document-level permissions and should not be exposed to untrusted users as-is.
- Generated answers can be incomplete or incorrect even when source passages are displayed.


## Analytics & evaluation

The Streamlit interface records local performance metadata in `analytics_events.jsonl`: model name, retrieval/total latency, number of sources, answer length, and a heuristic citation-coverage ratio. It does not persist question or answer text. User feedback is recorded as a separate rating event. The runtime log is excluded from version control.

The in-app benchmark uses `rag_test_questions.csv` to calculate source Hit Rate@K and MRR@K. These metrics measure retrieval ranking only; they do not establish answer correctness. Citation coverage is a line-level heuristic and is not a substitute for entailment-based evaluation or human review.


## Incremental indexing and runtime safety

Each chunk receives a deterministic ID derived from its source filename, location, chunk position, and text. During synchronization, the indexer computes the desired ID set, adds only new chunks, and removes stale chunks only after additions succeed. This avoids re-embedding unchanged passages during ordinary updates. Changing chunk size or overlap can change many IDs and trigger a larger re-index.

Answer generation supports streaming through Ollama. Context length is bounded by `RAG_MAX_CONTEXT_CHARS`; source labels are inspected after generation, and invalid/missing labels produce warnings. These are operational guardrails, not factual verification or claim-to-evidence entailment.

The Docker Compose setup binds the UI and Ollama API to loopback on the host and persists data in named volumes. It is intended for local development. The application does not implement authentication or per-document authorization, so it must not be exposed directly to untrusted networks.
