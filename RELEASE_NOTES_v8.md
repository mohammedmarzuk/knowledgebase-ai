# Release notes — v8

## Reliability and performance

- Added incremental Chroma synchronization using deterministic chunk IDs.
- Reuses existing embeddings for unchanged chunks; adds new chunks before removing stale chunks.
- Caches corpus text for repeated lexical retrieval on workspace pages.
- Caches tokenization to reduce repeated lexical scoring work.
- Bounds prompt context with `RAG_MAX_CONTEXT_CHARS`.
- Adds streaming answer generation from Ollama.

## Safety and operations

- Adds source-label integrity warnings and citation coverage indicators (heuristic only).
- Adds configurable upload size limits and row-wise CSV extraction.
- Loads optional `.env` configuration with `python-dotenv`.
- Adds `scripts/doctor.py` diagnostics.
- Adds Dockerfile and Docker Compose for a local-only Streamlit + Ollama setup.
- Adds a Settings page with active configuration and privacy boundaries.
- Configures Docker ports to bind to loopback by default.

## Validation

- Python compilation checks pass.
- Unit tests cover chunking, rank fusion, index delta planning, configuration parsing, answer guardrails, document extraction, and upload size validation.
- The test suite was run successfully in the build environment. The full Streamlit/Chroma/Ollama integration could not be exercised there because those runtime services/packages were not installed in that environment.

## Known limitations

- No authentication or document-level access control; intended for local single-user use.
- Source-label checks do not prove a claim is supported by its citation.
- Scanned PDF OCR is not built in.
- Lexical ranking is in-memory and intended for small/medium demo corpora; large deployments should use a persistent BM25 index.
