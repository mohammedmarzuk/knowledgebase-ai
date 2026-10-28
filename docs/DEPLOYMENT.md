# Deployment guide

This project supports two modes:

- **Local development:** `docker compose up -d --build` (app bound to loopback at
  `http://127.0.0.1:8501`; Ollama bound to loopback at `127.0.0.1:11434`).
- **Public VPS deployment:** Docker Compose plus Caddy HTTPS and password protection.

## Before public deployment

1. Use a Linux VPS with Docker Engine and the Docker Compose plugin installed.
2. Point a domain's DNS A record (and AAAA record only if IPv6 is configured) to the VPS.
3. Allow inbound TCP 80/443 in the cloud firewall. Do not expose port 8501 or 11434 publicly.
4. Copy `.env.example` to `.env`; configure `DOMAIN`, `ACME_EMAIL`, and `APP_AUTH_USER`.
5. Generate a password hash:

   ```bash
   docker run --rm caddy:2-alpine caddy hash-password --plaintext 'use-a-long-unique-password'
   ```

   Put the resulting bcrypt hash in `APP_AUTH_HASH`. Do not put the plaintext password in `.env`.
6. Review `MAX_UPLOAD_MB`, `APP_MEMORY_LIMIT`, `APP_CPU_LIMIT`, and your VPS disk/RAM capacity.
7. Start the production stack:

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.production.yml up -d --build
   docker compose logs -f app caddy ollama
   ```

Caddy obtains and renews TLS certificates automatically after DNS and firewall setup.
The application is protected with HTTP Basic Authentication over HTTPS. Use a unique,
strong password and share it only with trusted users. Basic Auth is a simple gate, not
full identity management, per-user permissions, or audit-grade access control.

## Ollama model setup

After the stack is up, download the configured model into the persistent Ollama volume:

```bash
docker compose exec ollama ollama pull qwen3:4b
docker compose restart app
```

The model requires sufficient RAM. Larger models can be much slower or fail on small VPS
instances. For CPU-only hosts, start with a small model and test response latency. GPU
acceleration requires a compatible host and a separately configured Ollama container.

## Persistent data and backups

The deployment persists uploaded files under `./data`, the Chroma vector index in the
`chroma_data` Docker volume, runtime analytics/history in `analytics_data`, and model
weights in `ollama_data`. Keep uploaded documents and backups private.

Create a local backup before upgrades:

```bash
bash scripts/backup.sh
```

Restore only after stopping the app, and review the script before running it:

```bash
docker compose stop app
bash scripts/restore.sh backups/knowledgebase-ai-backup-YYYYMMDD_HHMMSS.tar.gz
docker compose up -d app
```

Backups may contain confidential documents. Encrypt them and store them away from the
application server. Ollama model weights are not included because they can occupy many GB.

## Resource and upload limits

`MAX_UPLOAD_MB` controls the app's per-file limit. Streamlit's request limit is set through
`STREAMLIT_SERVER_MAX_UPLOAD_SIZE`; `STREAMLIT_SERVER_MAX_MESSAGE_SIZE` controls its message
ceiling. Raising these limits increases memory and processing risk: Streamlit buffers uploads,
parsers and embeddings consume RAM, and Chroma consumes disk. For public use, begin with a
lower limit (for example 50–100 MB) and raise it only after load testing. A 500 MB limit is
a maximum setting, not a guarantee that a server can safely process such a file.

## Operational checks

```bash
docker compose ps
docker compose logs --tail=200 app
docker compose exec app python scripts/doctor.py
```

Test the full flow after deployment: upload a small known document, refresh the index, ask
a question answered by that document, verify the cited source, restart the stack, and confirm
the document and index still exist.

## Important production limitations

This project is still a single-instance Streamlit application with local file storage and a
local Chroma index. The deployment overlay adds HTTPS, a basic password gate, container
restrictions, persistence, and backup scripts, but it does not add multi-tenant isolation,
per-user document permissions, SSO, malware scanning, a managed task queue, or high availability.
Do not upload regulated or highly sensitive data without a security review. For multiple users,
use a proper identity provider, per-user authorization, isolated storage/indexes, quotas, and
centralized logging before onboarding users.
