# Contributing

Thanks for helping improve KnowledgeBase AI.

## Development setup

1. Install Python 3.11+ and Ollama.
2. Create and activate a virtual environment.
3. Install development dependencies with `python -m pip install -r requirements-dev.txt`.
4. Run unit tests with `python -m pytest -q`.
5. Start the application with `python -m streamlit run app.py`.

## Pull requests

- Keep changes focused and explain the motivation.
- Add or update tests for behavior changes.
- Update the README when commands, configuration, or user-facing behavior changes.
- Do not commit secrets, private documents, local vector indexes, or virtual environments.
- Check that `python -m pytest -q` passes before opening a pull request.

## Dataset note

The included company handbook is fictional sample content for demonstrating retrieval. Do not present it as a real organization's policy.
