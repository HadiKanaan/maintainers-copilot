# Purpose: Local tracing helpers for modelserver.
# Significance: Wraps LLM/model calls with timing spans and redaction.
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional
import structlog
from modelserver.redaction import redact

_logger = structlog.get_logger()


# Redact string attributes before logging/tracing.
def _redact_attrs(attrs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Redact string attributes before logging/tracing."""
    out: Dict[str, Any] = {}
    for k, v in (attrs or {}).items():
        if isinstance(v, str):
            out[k] = redact(v)
        else:
            out[k] = v
    return out


@contextmanager
# Create a simple local span with start/end logs.
def create_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Create a simple local span that logs start/end with timing."""
    start = time.time()
    safe_attrs = _redact_attrs(attributes)
    _logger.info("span.start", span=name, attributes=safe_attrs)
    try:
        yield
    finally:
        duration = time.time() - start
        _logger.info("span.end", span=name, duration=duration, attributes=safe_attrs)
