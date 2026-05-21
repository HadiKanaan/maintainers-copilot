# Purpose: Core chatbot logic with tools, memory, and RAG.
# Significance: Orchestrates the entire chat experience.
import json
from typing import Dict, Any, List, Optional
import structlog
import httpx
import redis
from app.db import AsyncSessionLocal
from app.infra.tracing import create_span, create_tool_span, get_trace_id
from app.infra.redaction import redact
from app.infra.llm import chat_complete, get_embedding
from app.infra.rag import hybrid_search
from app.infra.vault import get_secret
from app.repositories.chat_repository import (
    create_conversation,
    get_conversations,
    save_message,
    save_long_term_memory,
    search_long_term_memory,
    write_audit_log,
)
from app.domain.models import Conversation, ChatResponse

logger = structlog.get_logger()


# Create a Redis client for short-term memory storage.
def _get_redis() -> redis.Redis:
    """Create a Redis client for short-term memory storage."""
    url = get_secret("REDIS_URL")
    return redis.Redis.from_url(url, decode_responses=True)


# Load the system prompt from disk to keep it editable.
def _load_system_prompt() -> str:
    """Load the system prompt from disk to keep it editable."""
    with open("prompts/system_prompt.txt", "r", encoding="utf-8") as f:
        return f.read().strip()


# Define tool specs for OpenAI function calling.
def _tool_specs() -> List[Dict[str, Any]]:
    """Define tool specs for OpenAI function calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "classify_issue",
                "description": "Classify a GitHub issue into bug/feature/docs/question",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_entities",
                "description": "Extract entities from issue text",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "summarize_issue",
                "description": "Summarize a GitHub issue in 2-3 sentences",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "Search documentation and resolved issues",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "label_filter": {"type": ["string", "null"]},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_memory",
                "description": "Write a long-term memory entry",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                },
            },
        },
    ]


# Append a message to short-term memory and enforce a 20-message cap.
def _append_short_term(redis_client: redis.Redis, conversation_id: str, message: Dict[str, str]) -> List[Dict[str, str]]:
    """Append a message to short-term memory and trim to last 20 messages."""
    key = f"conversation:{conversation_id}:messages"
    raw = redis_client.get(key)
    messages = json.loads(raw) if raw else []
    messages.append(message)
    messages = messages[-20:]
    redis_client.setex(key, 1800, json.dumps(messages))
    return messages


# Load short-term memory from Redis for a conversation.
def _load_short_term(redis_client: redis.Redis, conversation_id: str) -> List[Dict[str, str]]:
    """Load short-term memory from Redis or return empty list."""
    key = f"conversation:{conversation_id}:messages"
    raw = redis_client.get(key)
    return json.loads(raw) if raw else []


# Build the system prompt and inject relevant long-term memories.
async def _build_system_prompt(user_id: int, first_message: str) -> str:
    """Construct system prompt with relevant long-term memory injected."""
    with create_span("chat.memory_search", {"user_id": user_id}):
        base_prompt = _load_system_prompt()
        query_embedding = get_embedding(first_message)
        async with AsyncSessionLocal() as session:
            memories = await search_long_term_memory(session, user_id, query_embedding, top_k=3)

        if not memories:
            return base_prompt

        # Non-obvious block: format memory context for system prompt injection.
        memory_lines = [f"- {m.content}" for m in memories]
        context = "Relevant context from past sessions:\n" + "\n".join(memory_lines)
        return f"{base_prompt}\n\n{context}"


# Call modelserver endpoints with tracing and redacted logging.
def _call_modelserver(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call modelserver endpoints with redacted logging and spans."""
    with create_tool_span(f"modelserver.{path}", {"payload": redact(json.dumps(payload))}):
        base = get_secret("MODELSERVER_URL")
        with httpx.Client(timeout=30.0) as client:
            r = client.post(f"{base}{path}", json=payload)
            r.raise_for_status()
            return r.json()


# Main chat loop: memory, tools, LLM, and persistence.
async def process_message(user_id: int, conversation_id: str, message: str) -> ChatResponse:
    """Main chat loop: memory, tools, LLM, and persistence."""
    with create_span("chat.process_message", {"user_id": user_id, "conversation_id": conversation_id}):
        redis_client = _get_redis()

        # Short-term memory
        short_term = _load_short_term(redis_client, conversation_id)

        # System prompt with long-term memory
        system_prompt = await _build_system_prompt(user_id, message)

        # Build message list for LLM
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(short_term)
        messages.append({"role": "user", "content": message})

        # First LLM call with tools
        response = chat_complete(messages, tools=_tool_specs())
        tool_calls = response["choices"][0]["message"].get("tool_calls")

        # Execute tools if requested
        if tool_calls:
            for tool in tool_calls:
                name = tool["function"]["name"]
                args = json.loads(tool["function"]["arguments"])

                # Non-obvious block: dispatch tool calls to internal services.
                if name == "classify_issue":
                    result = _call_modelserver("/classify", {"text": args["text"]})
                elif name == "extract_entities":
                    result = _call_modelserver("/ner", {"text": args["text"]})
                elif name == "summarize_issue":
                    result = _call_modelserver("/summarize", {"text": args["text"], "max_length": 200})
                elif name == "search_docs":
                    with create_tool_span("search_docs", {"query": redact(args["query"])}):
                        result = await hybrid_search(args["query"], "default", top_k=5, label_filter=args.get("label_filter"))
                elif name == "write_memory":
                    content = args["content"]
                    with create_tool_span("memory.write", {"content": redact(content)}):
                        async with AsyncSessionLocal() as session:
                            embedding = get_embedding(content)
                            await save_long_term_memory(session, user_id, content, embedding)
                            await write_audit_log(session, user_id, "memory_write", content[:100])
                    result = {"status": "ok"}
                else:
                    result = {"error": "unknown tool"}

                # Non-obvious block: log tool outputs with redaction for traceability.
                logger.info("tool.result", name=name, output=redact(json.dumps(result)))

                messages.append({"role": "tool", "tool_call_id": tool["id"], "content": json.dumps(result)})

            # Final LLM call after tools
            response = chat_complete(messages)

        final_text = response["choices"][0]["message"]["content"]

        # Persist short-term memory and assistant message
        _append_short_term(redis_client, conversation_id, {"role": "user", "content": message})
        _append_short_term(redis_client, conversation_id, {"role": "assistant", "content": final_text})

        async with AsyncSessionLocal() as session:
            await save_message(session, int(conversation_id), "assistant", final_text)

        trace_id = get_trace_id()
        logger.info("chat.completed", conversation_id=conversation_id, trace_id=trace_id)
        return ChatResponse(response=final_text, conversation_id=conversation_id, trace_id=trace_id)


# Create a new conversation row for a user.
async def start_conversation(user_id: int) -> Conversation:
    """Create a new conversation row for a user."""
    async with AsyncSessionLocal() as session:
        return await create_conversation(session, user_id)


# List all conversations for a user.
async def list_conversations(user_id: int) -> List[Conversation]:
    """List all conversations for a user."""
    async with AsyncSessionLocal() as session:
        return await get_conversations(session, user_id)
