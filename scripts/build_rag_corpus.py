# Purpose: Build RAG corpus from issues and docs and store in pgvector.
# Significance: Supplies retrieval corpus for the chatbot.
import os
import json
from datetime import datetime
from typing import List, Dict
import structlog
import asyncio
import httpx
from app.infra.rag import embed_and_store

logger = structlog.get_logger()


# Load JSONL into list of dicts.
def _load_jsonl(path: str) -> List[Dict]:
    """Load JSONL into list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


# Split text into paragraph-level chunks with size constraints.
def _chunk_text(text_value: str) -> List[str]:
    """Split text into paragraph-level chunks with size constraints."""
    chunks = []
    for para in text_value.split("\n\n"):
        para = para.strip()
        if len(para) < 50:
            continue
        if len(para) <= 800:
            chunks.append(para)
            continue
        # Split long paragraphs on sentence boundary (naive period split).
        sentences = [s.strip() for s in para.split(".") if s.strip()]
        buff = ""
        for s in sentences:
            if len(buff) + len(s) + 1 > 800:
                if len(buff) >= 50:
                    chunks.append(buff)
                buff = s
            else:
                buff = f"{buff} {s}".strip()
        if len(buff) >= 50:
            chunks.append(buff)
    return chunks


# Fetch README and docs/*.md files from GitHub repo.
def _fetch_repo_docs() -> List[Dict]:
    """Fetch README and docs/*.md files from GitHub repo."""
    repo = os.environ.get("GITHUB_REPO")
    token = os.environ.get("GITHUB_TOKEN")
    if not repo:
        raise RuntimeError("GITHUB_REPO not set")

    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    owner, name = repo.split("/")
    docs = []

    with httpx.Client(headers=headers, timeout=30.0) as client:
        # README
        readme_url = f"https://api.github.com/repos/{owner}/{name}/readme"
        r = client.get(readme_url)
        if r.status_code == 200:
            readme = r.json()
            content_url = readme.get("download_url")
            if content_url:
                text = client.get(content_url).text
                docs.append({"filename": "README.md", "text": text})

        # docs folder
        docs_url = f"https://api.github.com/repos/{owner}/{name}/contents/docs"
        r = client.get(docs_url)
        if r.status_code == 200:
            for item in r.json():
                if item.get("name", "").endswith(".md"):
                    content_url = item.get("download_url")
                    if content_url:
                        text = client.get(content_url).text
                        docs.append({"filename": item.get("name"), "text": text})

    return docs


# Build corpus from issues and docs and store it via RAG API.
def main() -> None:
    """Build corpus from issues and docs and store it via RAG API."""
    issues = _load_jsonl("data/issues_test.jsonl")

    issue_chunks = []
    for issue in issues:
        body = issue.get("body") or ""
        chunks = _chunk_text(body)
        for idx, chunk in enumerate(chunks):
            issue_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": "issue",
                    "issue_id": str(issue.get("id")),
                    "label": issue.get("mapped_label"),
                    "date": issue.get("created_at"),
                    "chunk_index": idx,
                },
            })

    docs = _fetch_repo_docs()
    doc_chunks = []
    for doc in docs:
        chunks = _chunk_text(doc["text"])
        for idx, chunk in enumerate(chunks):
            doc_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": "docs",
                    "issue_id": None,
                    "label": None,
                    "date": datetime.utcnow().isoformat(),
                    "chunk_index": idx,
                    "filename": doc.get("filename"),
                },
            })

    logger.info("rag.corpus_counts", issues=len(issue_chunks), docs=len(doc_chunks))

    all_chunks = issue_chunks + doc_chunks
    asyncio.run(embed_and_store(all_chunks, "default"))
    logger.info("rag.corpus_built", total=len(all_chunks))


if __name__ == "__main__":
    main()
