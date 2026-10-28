# KnowledgeBase AI v12 — Deployment-ready baseline

## Added
- Hardened Docker image that runs the Streamlit process as a non-root user.
- Production Docker Compose overlay with Caddy reverse proxy, HTTPS, HTTP Basic Auth,
  persistent Caddy state, resource limits, and container security options.
- Deployment guide for DNS, TLS, model setup, operational checks, and limitations.
- Backup and restore scripts for documents, vector index, and runtime state.
- CI checks for deployment files and shell script syntax.

## Configuration
- Upload ceiling remains configurable up to 500 MB per file. For public deployments,
  use a conservative limit and load-test before raising it.
- Do not expose ports 8501 (Streamlit) or 11434 (Ollama) directly to the internet.
- Public deployment requires a domain, an ACME email, an auth username, and a Caddy
  bcrypt password hash in `.env`.

## Important limitations
- This is a single-instance application, not a high-availability or multi-tenant service.
- HTTP Basic Auth is a basic shared-password gate; it is not per-user authorization.
- Review the deployment guide and perform a security review before processing sensitive data.
