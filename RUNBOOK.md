# Runbook

## Quickstart

```bash
cp .env.example .env
# fill values
# docker compose up --build
```

## How to fetch and prepare the dataset

1. Set GITHUB_REPO and GITHUB_TOKEN in your environment.
2. Run scripts/fetch_issues.py to create data/issues_*.jsonl.

## How to train the classifier

1. Ensure data/issues_train.jsonl and data/issues_val.jsonl exist.
2. Run scripts/train_classifier.py to produce models/distilbert-issue-classifier.
3. Upload artifacts to MinIO bucket "models".

## How to build the RAG corpus

1. Run scripts/build_rag_corpus.py after fetch_issues.py.
2. Verify rag_chunks table is populated.

## How to run evals locally

1. Run scripts/eval_classification.py.
2. Run scripts/eval_rag.py.

## How to add a new tool to the chatbot

1. Add a tool spec in app/services/chat_service.py _tool_specs().
2. Implement tool execution in process_message().
3. Add unit tests for tool behavior.

## How to rotate a secret in Vault

1. Update the secret value in Vault under the same key.
2. Restart dependent services to pick up the new secret.

## How to access LangSmith traces

1. Set LANGSMITH_API_KEY and LANGSMITH_PROJECT.
2. Open LangSmith UI and filter by trace_id from logs.
