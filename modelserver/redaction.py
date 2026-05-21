# Purpose: Redact sensitive data before logging in modelserver.
# Significance: Prevents secrets/PII from leaking in logs.
import re
from typing import List

_REDACT_TOKEN = "[REDACTED]"

_API_KEY_PATTERNS: List[re.Pattern] = [
    re.compile(r"sk-[A-Za-z0-9-_]{16,}", re.IGNORECASE),
    re.compile(r"ghp_[A-Za-z0-9_]{10,}", re.IGNORECASE),
    re.compile(r"api_key=\s*[\w-]+", re.IGNORECASE),
    re.compile(r"x-?api-?key\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9-._~+/]+=*", re.IGNORECASE),
]

_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_JWT_PATTERN = re.compile(r"eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+")
_PASSWORD_LIKE = re.compile(r"(?i)(password|pwd)[\s:=]+\S+")


# Redact sensitive patterns from text before logging.
def redact(text: str) -> str:
    """Redact sensitive patterns from text before logging."""
    if not text:
        return text
    out = text
    for pat in _API_KEY_PATTERNS:
        out = pat.sub(_REDACT_TOKEN, out)
    out = _EMAIL_PATTERN.sub(_REDACT_TOKEN, out)
    out = _JWT_PATTERN.sub(_REDACT_TOKEN, out)
    out = _PASSWORD_LIKE.sub(_REDACT_TOKEN, out)
    return out
