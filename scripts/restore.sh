#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ! -f "$1" ]]; then
  echo "Usage: $0 path/to/knowledgebase-ai-backup.tar.gz" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "WARNING: restore overwrites matching local data/index/runtime paths."
read -r -p "Continue? [y/N] " answer
[[ "$answer" == "y" || "$answer" == "Y" ]] || exit 1
tar -xzf "$1" -C "$ROOT"
echo "Restore complete. Restart the app and verify the knowledge base."
