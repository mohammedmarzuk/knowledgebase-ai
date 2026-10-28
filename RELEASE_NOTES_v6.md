# Release notes — v6

## UI and performance upgrade

- Added incremental indexing: stable content fingerprints mean unchanged chunks are retained, while only new chunks are embedded and stale chunks removed.
- Added optional live streaming answers from Ollama to reduce perceived wait time. Fast/cached mode remains the default for repeated identical questions.
- Updated the visual design with richer hero treatment, feature tiles, metric-card hover states, clearer focus states, and workspace badges.
- Index sync now reports added, removed, and unchanged chunk counts.

## Notes

- Streaming improves perceived latency, not necessarily total generation time.
- First-time indexing still processes every document. Incremental runs are faster when only a subset of files changes.
- ChromaDB and Ollama remain local; the first model use may still be slower due to model initialization.
