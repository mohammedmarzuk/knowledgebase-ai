"""Run local setup diagnostics for KnowledgeBase AI."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

REQUIRED = ("streamlit", "chromadb", "ollama", "pypdf", "docx", "dotenv")


def main() -> int:
    print("KnowledgeBase AI · setup diagnostics")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project: {ROOT}")
    failed = False
    for module in REQUIRED:
        installed = importlib.util.find_spec(module) is not None
        print(f"{'OK' if installed else 'MISSING'} dependency: {module}")
        failed |= not installed

    data_dir = ROOT / "data"
    documents = [p for p in data_dir.iterdir() if p.is_file()] if data_dir.exists() else []
    print(f"Documents in data/: {len(documents)}")
    print(f"Configured model: {os.getenv('OLLAMA_CHAT_MODEL', 'qwen3:4b')}")
    print(f"Ollama host: {os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')}")

    if importlib.util.find_spec("ollama") is not None:
        try:
            import ollama
            result = ollama.list()
            models = [getattr(model, "model", None) or getattr(model, "name", None) for model in result.models]
            models = sorted(name for name in models if name)
            print(f"Ollama connection: OK ({len(models)} model(s) installed)")
            if models:
                for name in models:
                    print(f"  - {name}")
            else:
                print("  No local models found. Install one with: ollama pull qwen3:4b")
        except Exception as exc:
            print(f"Ollama connection: FAILED ({exc})")
            print("Start Ollama, or set OLLAMA_HOST to your Ollama server address.")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
