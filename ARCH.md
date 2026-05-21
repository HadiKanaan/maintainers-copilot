# Architecture

ASCII diagram:

Chat UI (Streamlit)
  |
  v
API (FastAPI) <--> Modelserver (FastAPI)
  |
  +--> PostgreSQL (pgvector)
  +--> Redis
  +--> MinIO
  +--> Vault
  +--> LangSmith

## Service descriptions

API: Hosts auth, chat orchestration, RAG, and memory flows. It never calls SQL/Redis directly from routers.

Modelserver: Runs classifier, NER, and summarizer. It verifies classifier weights on startup and uses Azure OpenAI for summarization.

PostgreSQL + pgvector: Stores users, conversations, messages, long-term memory, and RAG chunks.

Redis: Holds short-term memory per conversation with TTL.

MinIO: Stores model artifacts and evaluation reports.

Vault: Central secret store; all secrets flow through get_secret().

LangSmith: Central tracing backend for spans and trace IDs.

## Data flow: issue fetch → train → modelserver

scripts/fetch_issues.py → data/issues_*.jsonl → scripts/train_classifier.py → models/distilbert-issue-classifier/ → MinIO (models bucket) → modelserver/classifier.py

## Data flow: user message → chatbot → api → tools → RAG → response

User message → Streamlit → /chat/message → chat_service.process_message() → tool calls (modelserver/RAG/memory) → LLM response → Redis + messages table

## Memory architecture

Short-term: Redis key conversation:{conversation_id}:messages, last 20 messages, TTL 1800 seconds.

Long-term: pgvector table long_term_memory, top-3 cosine search injected into system prompt.

## Secrets flow

.env → Vault → get_secret() → api/modelserver/services

## Tracing flow

Every span → LangSmith (via app/infra/tracing.py) → trace_id logged on each request
