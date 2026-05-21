# Maintainer's Copilot

An authenticated chatbot for open-source maintainers that classifies GitHub issues, performs RAG over project docs and resolved issues, and carries memory across conversations.

Architecture (ASCII):

API (FastAPI) <--> Modelserver (FastAPI)  
API <--> PostgreSQL (pgvector)  
API <--> Redis  
API <--> MinIO  
API <--> Vault  
Chat UI (Streamlit) --> API

Services:
- `api` — FastAPI backend for auth, conversations, RAG orchestration
- `modelserver` — separate FastAPI process for classifier, NER, summarizer
- `migrate` — runs Alembic migrations
- `db`, `redis`, `minio`, `vault` — infra services

Quickstart

1. Copy example env:

```bash
cp .env.example .env
# edit .env to fill real values or point values to Vault paths
```

Using uv (optional)

```bash
uv pip install -r requirements-test.txt
```

2. Build and start

```bash
docker compose up --build
```

Docs: see `DECISIONS.md`, `SECURITY.md`, `ARCH.md`, `RUNBOOK.md`, `EVALS.md` for details.
