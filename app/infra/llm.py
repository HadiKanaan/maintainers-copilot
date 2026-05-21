# Purpose: Azure OpenAI client helpers and LLM call wrappers.
# Significance: Centralizes model configuration, logging, and token metrics.
from typing import List, Optional, Dict, Any
import time
import structlog
from app.infra.vault import get_secret
from app.infra.redaction import redact
from app.infra.tracing import create_span

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None  # type: ignore

_logger = structlog.get_logger()

_chat_client: Optional[AzureOpenAI] = None
_embed_client: Optional[AzureOpenAI] = None


# Initialize Azure OpenAI clients using Vault-sourced secrets.
def init_llm_clients() -> None:
    global _chat_client, _embed_client
    api_key = get_secret("AZURE_OPENAI_API_KEY")
    endpoint = get_secret("AZURE_OPENAI_ENDPOINT")
    api_version = get_secret("AZURE_OPENAI_API_VERSION")
    deployment = get_secret("AZURE_OPENAI_DEPLOYMENT_NAME")
    embed_deployment = get_secret("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    if AzureOpenAI is None:
        _logger.warning("openai.AzureOpenAI not available; LLM calls will fail")
        return

    _chat_client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
    _embed_client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)


# Return the cached chat client or initialize it.
def get_chat_client() -> AzureOpenAI:
    if _chat_client is None:
        init_llm_clients()
    if _chat_client is None:
        raise RuntimeError("Chat client not initialized")
    return _chat_client


# Return the cached embedding client or initialize it.
def get_embedding_client() -> AzureOpenAI:
    if _embed_client is None:
        init_llm_clients()
    if _embed_client is None:
        raise RuntimeError("Embedding client not initialized")
    return _embed_client


# Generate an embedding vector for a single input string.
def get_embedding(text: str) -> List[float]:
    client = get_embedding_client()
    deployment = get_secret("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    with create_span("embedding", {"model": deployment, "input": redact(text)}):
        start = time.time()
        resp = client.embeddings.create(model=deployment, input=[text])
        latency = time.time() - start
        try:
            vector = resp.data[0].embedding
        except Exception:
            _logger.error("embedding.parse_failed", resp=repr(resp))
            raise
        usage = getattr(resp, "usage", None)
        _logger.info("embedding.generated", tokens=usage, latency=latency)
        return vector


# Call chat completions with optional tool definitions.
def chat_complete(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    client = get_chat_client()
    deployment = get_secret("AZURE_OPENAI_DEPLOYMENT_NAME")
    start = time.time()
    # redact messages for logs
    safe_messages = [{"role": m.get("role"), "content": redact(m.get("content", ""))} for m in messages]
    safe_tools = None
    if tools:
        safe_tools = [redact(str(t)) for t in tools]
    with create_span("chat", {"model": deployment, "messages": safe_messages, "tools": safe_tools}):
        _logger.info("chat.request", messages=safe_messages, tools=safe_tools)
        resp = client.chat.completions.create(model=deployment, messages=messages, tools=tools)
        latency = time.time() - start
        # attempt to log tokens if provided
        token_info = getattr(resp, "usage", None)
        _logger.info("chat.response", latency=latency, tokens=token_info)
        try:
            return resp.model_dump()
        except Exception:
            return {"response": resp}
