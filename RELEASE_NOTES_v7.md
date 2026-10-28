# Release notes — v7 Multi-page workspace

## Added
- Dedicated Documents page for local file library management.
- Search Lab page to inspect hybrid retrieval results and optionally run cross-encoder reranking.
- Evaluation Lab page with Hit Rate@K, MRR@K, progress indication, and CSV export.
- Performance Center page for local telemetry, latency charts, model availability, and feedback summary.
- Shared `workspace_core.py` helpers for read-only access to the existing Chroma index, retrieval, benchmark data, analytics, and consistent page styling.
- New page-level visual system with gradient hero panels, metric cards, and responsive layout defaults.

## Notes and limitations
- The existing Chat page remains the primary place to build or refresh the index and generate answers.
- Deleting or modifying source files does not automatically update the index; rebuild it from the Chat page.
- Cross-encoder reranking is optional and can require model downloads and additional memory.
- Retrieval metrics evaluate expected source ranking, not factual correctness of generated answers.
- The existing test suite passes: 24 tests.
