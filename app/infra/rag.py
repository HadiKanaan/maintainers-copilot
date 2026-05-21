# Purpose: RAG operations (embedding, storage, retrieval, query rewrite).
# Significance: Centralizes hybrid search and RAG persistence.
from typing import List, Dict, Any, Optional
import uuid
import structlog
from sqlalchemy import text
from app.db import engine
from app.infra.llm import get_embedding, chat_complete
from app.infra.tracing import create_span
from app.infra.redaction import redact
from rank_bm25 import BM25Okapi

logger = structlog.get_logger()

# In-memory BM25 index cache
_bm25_index: Optional[BM25Okapi] = None
_bm25_docs: List[Dict[str, Any]] = []
_bm25_collection: Optional[str] = None


# Tokenize text for BM25 (simple whitespace).
def _tokenize(text_value: str) -> List[str]:
    """Simple whitespace tokenizer for BM25 indexing."""
    return text_value.lower().split()


# Load chunks and build a BM25 index for a collection.
async def _load_bm25(collection: str) -> None:
    """Load all chunks for a collection and build a BM25 index in memory."""
    global _bm25_index, _bm25_docs, _bm25_collection
    async with engine.begin() as conn:
        result = await conn.execute(
            text("SELECT text, source, issue_id, label, date, chunk_index FROM rag_chunks WHERE collection = :c"),
            {"c": collection},
        )
        rows = result.fetchall()
    _bm25_docs = [
        {
            "text": row[0],
            "metadata": {
                "source": row[1],
                "issue_id": row[2],
                "label": row[3],
                "date": row[4],
                "chunk_index": row[5],
            },
        }
        for row in rows
    ]
    tokenized = [_tokenize(doc["text"]) for doc in _bm25_docs]
    _bm25_index = BM25Okapi(tokenized) if tokenized else None
    _bm25_collection = collection


# Ensure the BM25 cache is ready for the requested collection.
async def _ensure_bm25(collection: str) -> None:
    """Ensure the BM25 index is built for the requested collection."""
    if _bm25_index is None or _bm25_collection != collection:
        await _load_bm25(collection)


# Embed chunks and store them in the rag_chunks table.
async def embed_and_store(chunks: List[Dict[str, Any]], collection: str) -> None:
    """Embed each chunk and store it in rag_chunks with metadata."""
    with create_span("rag.embed_and_store", {"collection": collection, "count": len(chunks)}):
        async with engine.begin() as conn:
            for idx, chunk in enumerate(chunks):
                text_value = chunk["text"]
                metadata = chunk.get("metadata", {})
                embedding = get_embedding(text_value)
                # Non-obvious block: embed and insert each chunk with metadata.
                await conn.execute(
                    text(
                        """
                        INSERT INTO rag_chunks (id, collection, text, embedding, source, issue_id, label, date, chunk_index)
                        VALUES (:id, :collection, :text, :embedding, :source, :issue_id, :label, :date, :chunk_index)
                        """
                    ),
                    {
                        "id": str(uuid.uuid4()),
                        "collection": collection,
                        "text": text_value,
                        "embedding": embedding,
                        "source": metadata.get("source"),
                        "issue_id": metadata.get("issue_id"),
                        "label": metadata.get("label"),
                        "date": metadata.get("date"),
                        "chunk_index": metadata.get("chunk_index", idx),
                    },
                )
        await _ensure_bm25(collection)
        logger.info("rag.store_complete", collection=collection, count=len(chunks))


# Perform hybrid dense+sparse search with RRF fusion.
async def hybrid_search(query: str, collection: str, top_k: int, label_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Perform hybrid search using dense retrieval + BM25 with RRF fusion."""
    with create_span("rag.hybrid_search", {"collection": collection, "query": redact(query)}):
        await _ensure_bm25(collection)
        query_embedding = get_embedding(query)

        # Dense search (cosine similarity)
        async with engine.begin() as conn:
            result = await conn.execute(
                text(
                    """
                    SELECT text, source, issue_id, label, date, chunk_index,
                           1 - (embedding <=> :query_embedding) AS score
                    FROM rag_chunks
                    WHERE collection = :collection
                    ORDER BY embedding <=> :query_embedding
                    LIMIT :limit
                    """
                ),
                {"query_embedding": query_embedding, "collection": collection, "limit": top_k * 2},
            )
            dense_rows = result.fetchall()

        dense = [
            {
                "text": row[0],
                "metadata": {
                    "source": row[1],
                    "issue_id": row[2],
                    "label": row[3],
                    "date": row[4],
                    "chunk_index": row[5],
                },
                "score": float(row[6]),
            }
            for row in dense_rows
        ]

        # Sparse search (BM25)
        sparse = []
        if _bm25_index is not None:
            tokenized_query = _tokenize(query)
            scores = _bm25_index.get_scores(tokenized_query)
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[: top_k * 2]
            for idx, score in ranked:
                doc = _bm25_docs[idx]
                sparse.append({"text": doc["text"], "metadata": doc["metadata"], "score": float(score)})

        # Reciprocal Rank Fusion combines dense and sparse rankings.
        rrf_k = 60
        fused: Dict[str, Dict[str, Any]] = {}
        for rank, item in enumerate(dense, start=1):
            key = f"{item['metadata'].get('source')}:{item['metadata'].get('issue_id')}:{item['metadata'].get('chunk_index')}"
            fused.setdefault(key, {"text": item["text"], "metadata": item["metadata"], "score": 0.0})
            fused[key]["score"] += 1.0 / (rrf_k + rank)
        for rank, item in enumerate(sparse, start=1):
            key = f"{item['metadata'].get('source')}:{item['metadata'].get('issue_id')}:{item['metadata'].get('chunk_index')}"
            fused.setdefault(key, {"text": item["text"], "metadata": item["metadata"], "score": 0.0})
            fused[key]["score"] += 1.0 / (rrf_k + rank)

        results = list(fused.values())

        # Apply label filter only after fusion.
        if label_filter:
            results = [r for r in results if r["metadata"].get("label") == label_filter]

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# Rewrite a user query for better retrieval using Azure OpenAI.
def rewrite_query(query: str) -> str:
    """Rewrite a query to be more searchable using Azure OpenAI."""
    with create_span("rag.rewrite_query", {"query": redact(query)}):
        system_prompt = (
            "Rewrite the following question to be more specific and searchable "
            "for a GitHub issue database. Return only the rewritten query, nothing else."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        response = chat_complete(messages)
        try:
            content = response["choices"][0]["message"]["content"]
            return content.strip()
        except Exception:
            logger.warning("rag.rewrite_failed", query=redact(query))
            return query
