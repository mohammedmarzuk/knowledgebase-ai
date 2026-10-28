# KnowledgeBase AI

**A local-first Retrieval-Augmented Generation (RAG) assistant for asking questions about your own documents.** Built with Python, Streamlit, ChromaDB, and Ollama.

KnowledgeBase AI ingests documents, indexes their text for semantic search, retrieves relevant passages for a question, and asks a locally running language model to answer using that context. Retrieved source passages are available for verification.

> **Demo data:** The included company handbook and policy documents are fictional sample content for software demonstration. They are not real company policies, legal advice, or authoritative guidance.

## Highlights

- **Local LLM inference** through Ollama.
- **Document ingestion** for TXT, Markdown, PDF, DOCX, and CSV with row-aware tabular extraction.
- **Hybrid retrieval** combining ChromaDB semantic search with lexical keyword ranking using reciprocal rank fusion.
- **Exact-term discovery** for acronyms, policy names, and other keyword-heavy queries.
- **Configurable retrieval depth** and source passage inspection.
- **Source-aware responses** with document names and page/chunk references.
- **Adjustable retrieval** to choose how many passages are provided to the model.
- **Expanded sample corpus** spanning HR, security, data governance, AI usage, engineering, ETL/SQL, ML lifecycle, project management, and more.
- **Optional cross-encoder reranking** to refine hybrid retrieval results.
- **Evidence-first answer prompting** with explicit source labels and refusal guidance when evidence is insufficient.
- **Automated unit tests** and a GitHub Actions test workflow.
- **Incremental indexing** that reuses embeddings for unchanged chunk IDs and removes stale chunks only after new chunks have been indexed.
- **Streaming answers** from Ollama for improved perceived responsiveness.
- **Citation-label guardrails** that flag invalid or missing source labels (heuristic only; not a proof of factuality).
- **Environment-based configuration** for chunking, retrieval depth, context size, upload limits, analytics, and model endpoints.
- **Setup diagnostics** with `python scripts/doctor.py`.
- **Docker Compose** for a local Streamlit + Ollama setup.


## Advanced retrieval (optional)

The default setup uses ChromaDB semantic retrieval plus lexical ranking and Reciprocal Rank Fusion. For a second-stage cross-encoder reranker, install the optional dependencies:

```bash
pip install -r requirements-advanced.txt
```

Then set `RAG_RERANKER_MODEL` in your environment before launching Streamlit:

```bash
export RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
python -m streamlit run app.py
```

The first use downloads the reranker model and may require additional disk space and memory. Leave the variable unset to use the lighter default pipeline. The model is cached in-process after its first load.

CSV files are read as row-wise field/value text to preserve column names in retrieval context. Scanned PDFs without a text layer are not OCR-processed; convert them to searchable PDFs before indexing.

## Analytics dashboard and evaluation

The app includes a local analytics section with response-time trends, model usage, lightweight citation-coverage heuristics, feedback counts, and a downloadable JSONL report. The logger records performance metadata only; it intentionally does not save question or answer text. Runtime analytics are stored in `analytics_events.jsonl`, which is ignored by Git.

Use **Run retrieval benchmark** in the app to evaluate the expected source documents in `rag_test_questions.csv`. It reports Hit Rate@K and MRR@K. These are retrieval metrics, not proof that generated answers are factually correct. Citation coverage is also a heuristic and does not verify that a citation actually supports a claim.

## Screenshots

Run the app locally and add screenshots of the interface here before publishing, for example in `docs/images/`. Screenshots make the repository easier for visitors and recruiters to evaluate.

## Architecture

```text
Documents (TXT / MD / PDF / DOCX / CSV)
                |
                v
          Text extraction
                |
                v
    Chunking (180 words, 35 overlap)
                |
                v
       ChromaDB vector index
                |
Question --> semantic retrieval + lexical ranking
                         |
                  reciprocal rank fusion
                         |
                    top-k passages
                                      |
                                      v
                               Ollama chat model
                                      |
                                      v
                            Answer + inspectable sources
```

See [`docs/architecture.md`](docs/architecture.md) for component responsibilities and limitations.

## Requirements

- macOS, Windows, or Linux
- Python 3.11 or newer recommended
- [Ollama](https://ollama.com/download) installed and running
- Sufficient memory and disk space for the model you choose
- Internet access for initial package, model, and embedding-model downloads

## Quick start (macOS / VS Code)

### 1. Clone your repository

Replace the placeholder with your GitHub username and repository name:

```bash
git clone https://github.com/YOUR-USERNAME/knowledgebase-ai.git
cd knowledgebase-ai
```

If you are testing before publishing, extract the project ZIP and open the extracted project folder in VS Code instead.

### 2. Install and start Ollama

Install Ollama from the official [download page](https://ollama.com/download), then pull the default model:

```bash
ollama pull qwen3:4b
```

If Ollama is not already running, start it in a separate terminal:

```bash
ollama serve
```

You can use another Ollama model, but it must be installed locally first.

### 3. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure and run the app

Copy `.env.example` to `.env` and adjust values if needed. The app loads `.env` automatically; the configured Ollama model must already be installed.

```bash
cp .env.example .env
python scripts/doctor.py
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

### 5. Build the knowledge base and ask a question

1. In the sidebar, click **Build / refresh knowledge base** to index the included sample documents.
2. Choose a suggested question or enter your own.
3. Click **Generate answer**.
4. Expand the retrieved source passages and verify that they support the response.

The first indexing run may download ChromaDB's default embedding model. This can take time depending on your network and machine.

### Retrieval evaluation

After building the local index in the app, run the source-retrieval benchmark from the project root:

```bash
python scripts/evaluate_retrieval.py --k 5
```

The script reports source Hit Rate@K and Mean Reciprocal Rank (MRR@K) using `rag_test_questions.csv`. These metrics assess whether the expected source document appears near the top of retrieval results; they do not independently validate generated-answer correctness.

### Retrieval behavior

The application combines semantic results with a lightweight lexical ranking and merges the ranked lists with reciprocal rank fusion. This helps when a question contains exact terms, acronyms, or policy names while retaining semantic matching for paraphrased questions. The lexical pass currently loads indexed chunks into memory, which is appropriate for the included demo corpus; for large corpora, use a dedicated BM25 index or search service.

## Add your own documents

1. Use **Add documents** in the sidebar to upload TXT, Markdown, PDF, or DOCX files.
2. Click **Save uploaded files**.
3. Click **Build / refresh knowledge base** to re-index all supported files in `data/`.
4. Ask questions about the newly indexed content.

You can also place supported files directly in `data/` and rebuild the index. Scanned/image-only PDFs require OCR before indexing. Only add files you are authorized to process.

## Configuration

The default chat model is `qwen3:4b`. You can override it with an environment variable:

```bash
OLLAMA_CHAT_MODEL=llama3.2 python -m streamlit run app.py
```

The `.env.example` file documents the optional setting, but the app does not automatically load `.env` files. Export the variable in your shell or set it directly in the command as shown above. The model must be installed using `ollama pull <model-name>`.

## Docker Compose (local only)

Docker Compose runs the Streamlit app and Ollama in containers. The ports are bound to `127.0.0.1` so they are not published to your local network by default.

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen3:4b
docker compose up --build -d app
```

Open `http://127.0.0.1:8501`. The first model pull can take time and needs disk space. To stop the services, run `docker compose down`; named volumes retain the Ollama model and Chroma index. To delete persisted state too, use `docker compose down -v` (this is destructive).

## Tests and development

Install development dependencies and run unit tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover chunk splitting, overlap behavior, invalid chunk parameters, and stable chunk identifiers. GitHub Actions runs the unit test suite on pushes and pull requests.

## Repository structure

```text
knowledgebase-ai/
├── .github/workflows/tests.yml
├── data/                   # Fictional sample knowledge-base documents
├── docs/                   # Architecture notes
├── tests/                  # Unit tests
├── .env.example            # Example configuration (no secrets)
├── .gitignore
├── app.py                  # Streamlit application
├── rag_utils.py            # Testable RAG utility functions
├── requirements.txt
├── requirements-dev.txt
├── DATASET_CATALOG.md
├── rag_test_questions.csv
├── CONTRIBUTING.md
├── SECURITY.md
└── LICENSE
```

The local `chroma_db/`, virtual environment, and secret configuration files are intentionally excluded from version control. The vector index is generated when you build the knowledge base.

## Privacy, safety, and limitations

- In this local setup, documents and vector index are stored on your machine and answer generation runs through local Ollama. Initial dependency/model downloads require internet access.
- The app does not encrypt local files or the vector database and does not include authentication or document-level access controls. Keep it on a trusted machine and do not expose it directly to the public internet.
- Prompt-injection resistance is best-effort; retrieved content must be treated as untrusted.
- RAG can return incomplete or incorrect answers. Always inspect the source passages before relying on a response.
- Do not use confidential, personal, regulated, or proprietary data unless you understand and are authorized to accept the storage and security risks.
- This project is a learning/demo prototype, not a production-ready enterprise knowledge system.

## Roadmap ideas

- Add evaluation metrics for retrieval precision and groundedness.
- Support incremental indexing and document-change detection.
- Add configurable embedding models and chunking strategies.
- Add authentication, access controls, and safer multi-user deployment.
- Add OCR for scanned documents.
- Add authentication and document-level authorization before any multi-user deployment.
- Add a persistent BM25 index for large corpora instead of in-memory lexical scoring.
- Add semantic groundedness evaluation with human-reviewed examples; source-label checks alone are insufficient.

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE).

## Performance improvements

The Streamlit app caches corpus records and tokenization for repeated lexical searches, and caches identical non-streaming answer requests for a limited time. Incremental indexing preserves existing chunk embeddings, adds new chunks first, and removes stale chunks after additions succeed. Context length, chunking, candidate multiplier, upload size, model endpoint, streaming, and analytics can be configured through `.env`. The default retrieval context is three passages to reduce prompt size and local model latency; increase the retrieval slider when answer coverage matters more than speed.

The first run can still be slower because Chroma may need to download or initialize its default embedding model and Ollama may need to load the language model into memory.

## v5 answer-quality evaluation

The Analytics & Evaluation section now includes an opt-in **Evaluate generated answer quality** benchmark. It runs questions from `rag_test_questions.csv` through retrieval and the selected local Ollama model, then reports:

- **Reference token F1 / recall**: lexical overlap between the generated answer and the reference answer. This is a lightweight regression signal and may penalize valid paraphrases.
- **Expected source hit**: whether the expected source document was retrieved.
- **Citation-label validity**: whether generated `[Source N]` labels point to one of the passages returned for that question. This does not prove that the cited passage supports the claim.
- **Abstention accuracy**: whether the model abstains on benchmark questions whose reference answer indicates the information is unspecified.

The benchmark can take several minutes because it calls the local language model once per question. Results are displayed in the UI and can be explicitly exported as CSV. The normal analytics logger still does not store question or answer text.

Run the tests with:

```bash
pytest -q
```

The evaluation helpers are dependency-free and have unit tests in `tests/test_evaluation_utils.py`.


## v6 UI and performance upgrade

- **Incremental indexing:** unchanged chunks are retained, new or changed chunks are added, and stale chunks are removed after successful additions.
- **Live streaming mode:** optionally streams Ollama tokens to the interface as they arrive. Fast/cached mode remains available.
- **UI refresh:** feature tiles, clearer workspace badges, metric-card hover states, and improved focus styling.
- **Index sync summary:** shows added, removed, and unchanged chunks after indexing.

For full details, see `RELEASE_NOTES_v8.md`. Streaming reduces perceived waiting time; actual completion speed depends on the local model and hardware.

## Multi-page workspace (v7 upgrade)

Streamlit automatically discovers the `pages/` directory. The application now includes dedicated workspace pages:

- **Documents** — upload, filter, inspect, and delete local source files. Rebuild the index from the Chat page after changing the library.
- **Search Lab** — inspect hybrid retrieval results and optionally compare cross-encoder reranking.
- **Evaluation Lab** — run the source-document retrieval benchmark and export results as CSV.
- **Performance** — review local latency telemetry, installed Ollama models, and feedback metrics.
- **Settings** — inspect environment-backed configuration, runtime health, and local-only deployment limitations.

Run the app with `python -m streamlit run app.py`. The pages share the same local Chroma collection and analytics log. The reranker is optional and can download model weights on first use. Benchmark scores depend on the quality of expected-source labels and do not prove generated answers are factually correct.


## v8 production-readiness upgrade

Version 8 focuses on reliability and operations rather than adding UI features alone:

- Incremental indexing reuses unchanged chunk embeddings and synchronizes deletions after successful additions.
- Ollama answer generation can stream tokens, with a bounded context window.
- CSV rows are indexed separately with header/value pairs; file sizes are validated before parsing.
- Source-label warnings flag invalid or absent citations. This is not a factuality or entailment check.
- `.env` configuration, setup diagnostics, Dockerfile, and Docker Compose are included.
- The Settings page documents runtime configuration and privacy/security boundaries.

See `RELEASE_NOTES_v8.md` for the changes and known limitations. Run `python scripts/doctor.py` to check the local setup and `pytest -q` to run the automated tests. The app remains a local single-user project: it has no authentication or per-document authorization and should not be exposed directly to the public internet.


## Large document uploads and local resource limits

The default configuration permits individual files up to **500 MB**. Streamlit's
request limits are configured in `.streamlit/config.toml` (`maxUploadSize = 500`
and `maxMessageSize = 550`). You can lower the per-file limit using
`MAX_UPLOAD_MB` in `.env` (maximum supported by this app: 500 MB).

Large uploads require sufficient free disk space and RAM because Streamlit receives
the upload in memory before the app saves it. For very large files or many files at
once, upload in smaller batches. Do not raise limits on a public deployment without
authentication, quotas, malware scanning, and reverse-proxy/server resource limits.
Scanned PDFs need OCR; large PDFs and DOCX files may take longer to parse and embed.
Files with duplicate names are saved with a numbered suffix rather than silently
overwriting the existing document.


## Large-file processing and optional OCR

The index builder displays progress through extraction and embedding/indexing batches.
Unchanged chunks reuse their stored embeddings; only new chunks are embedded.

For scanned/image-only PDFs, install the optional Python OCR packages with
`pip install -r requirements-advanced.txt`, then install the **Tesseract OCR**
and **Poppler** system tools for your operating system. OCR is attempted only when
a PDF has no extractable text layer. OCR quality depends on scan resolution,
language packs, and document quality. OCR is intentionally optional because these
system tools are not bundled with Python dependencies.


## v11 — Document intelligence and persistent processing history

The **Document Intelligence** page adds on-demand local Ollama summaries, extracted-text previews,
content-addressed summary caching, and a bounded local processing history. Summary cache keys are
SHA-256 hashes of file contents, so editing a document automatically invalidates its old summary.
The local history records indexing syncs and summary success/failure events, retaining at most 100
events by default. These records remain on the local machine under `runtime/`.

To use it, upload documents in **Documents**, build or refresh the index, then open **Document
Intelligence** from Streamlit's page navigation. Choose a document and select **Generate / refresh
summary**. The summary is generated from extracted text using the selected local Ollama model.
Only the first 100,000 extracted characters are sent to the model per summary request to bound
resource use. Verify important facts against the original document and source citations.

## Deploy to a VPS with HTTPS

The repository includes a Docker deployment and an optional Caddy reverse proxy with
automatic TLS and HTTP Basic Authentication. Read [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)
before exposing the application publicly. Do not expose Streamlit port 8501 or Ollama
port 11434 directly to the internet. The default setup is intended for a single trusted
user; multi-user authorization and tenant isolation are not included.
