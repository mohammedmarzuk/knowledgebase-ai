# Security Policy

## Scope

KnowledgeBase AI is an educational local-first prototype. It is not a production security boundary and does not provide authentication, access control, encryption at rest, or document-level authorization.

## Reporting a vulnerability

Please do not publish sensitive information, credentials, or private documents in an issue. If this repository is published publicly, use GitHub's private vulnerability reporting feature when enabled; otherwise contact the repository maintainer privately before sharing details publicly.

## Safe use

- Do not commit API keys, passwords, `.env` files, or confidential documents.
- Treat retrieved documents as untrusted input. Prompt-injection defenses are best-effort, not a guarantee.
- Keep Ollama and Python dependencies up to date.
- Review source passages before relying on generated answers.
