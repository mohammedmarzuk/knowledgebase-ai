#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/backups}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEST"
ARCHIVE="$DEST/knowledgebase-ai-backup-$STAMP.tar.gz"
cd "$ROOT"
# Back up uploaded documents, Chroma index, and runtime history/analytics.
# Ollama model weights can be backed up separately; they may be many GB.
tar -czf "$ARCHIVE" data chroma_db runtime analytics_events.jsonl 2>/dev/null || {
  status=$?
  if [[ $status -ne 1 ]]; then exit "$status"; fi
}
echo "Backup created: $ARCHIVE"
echo "Store this archive securely; it may contain confidential documents."
