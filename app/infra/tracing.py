# Purpose: Tracing helpers and span wrapper with LangSmith.
# Significance: Standardizes spans for LLM, tools, and retrieval.
import time
import uuid
import os
from contextlib import contextmanager
from typing import Dict, Any, Optional
import structlog
from structlog.contextvars import bind_contextvars, get_contextvars
from app.infra.redaction import redact

try:
    from langsmith import Client
except Exception:
    Client = None  # type: ignore

_logger = structlog.get_logger()
_client: Optional[Client] = None
_project: Optional[str] = None


# Initialize LangSmith tracing client and project.
def init_tracing(api_key: Optional[str] = None, project: Optional[str] = None) -> None:
    """Initialize LangSmith client and project name."""
    global _client, _project
    _project = project or os.environ.get("LANGSMITH_PROJECT")
    if Client is None:
        _logger.warning("langsmith client not available")
        return
    if not api_key:
        _logger.warning("LANGSMITH_API_KEY missing; tracing disabled")
        return
    _client = Client(api_key=api_key)
    _logger.info("tracing.initialized", project=_project)


# Return or create a trace_id bound to the current context.
def get_trace_id() -> str:
    """Return or create a trace_id for the current request context."""
    current = get_contextvars().get("trace_id")
    if current:
        return str(current)
    trace_id = str(uuid.uuid4())
    bind_contextvars(trace_id=trace_id)
    return trace_id


# Redact string attributes before logging or tracing.
def _redact_attrs(attrs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Redact string attributes before logging or tracing."""
    out: Dict[str, Any] = {}
    for k, v in (attrs or {}).items():
        if isinstance(v, str):
            out[k] = redact(v)
        else:
            out[k] = v
    return out


# Create a tracing span with redacted attributes.
@contextmanager
def create_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Create a tracing span and ensure trace_id is bound to logs."""
    start = time.time()
    trace_id = get_trace_id()
    safe_attrs = _redact_attrs(attributes)
    run_id = None

    if _client is not None:
        try:
            run = _client.create_run(
                name=name,
                run_type="chain",
                inputs=safe_attrs,
                project_name=_project,
            )
            run_id = run.id
        except Exception:
            run_id = None

    _logger.info("span.start", span=name, trace_id=trace_id, attributes=safe_attrs)
    try:
        yield
    finally:
        duration = time.time() - start
        _logger.info("span.end", span=name, trace_id=trace_id, duration=duration, attributes=safe_attrs)
        if _client is not None and run_id is not None:
            try:
                _client.update_run(run_id, outputs={"duration": duration}, end_time=time.time())
            except Exception:
                pass


# Create a span wrapper for RAG retrieval steps.
@contextmanager
def create_rag_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Span helper for retrieval steps (RAG)."""
    with create_span(f"rag.{name}", attributes):
        yield


# Create a span wrapper for tool calls.
@contextmanager
def create_tool_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Span helper for tool calls."""
    with create_span(f"tool.{name}", attributes):
        yield
