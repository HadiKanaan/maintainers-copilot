# Purpose: Unit tests for redaction safeguards.
# Significance: Ensures secrets/PII are removed from logs/traces.
from app.infra.redaction import redact


def test_redaction_api_key():
    input_text = "my key is sk-1234567890abcdef"
    out = redact(input_text)
    assert "sk-1234567890abcdef" not in out


def test_redaction_github_token():
    input_text = "token ghp_abc123def456 used here"
    out = redact(input_text)
    assert "ghp_abc123def456" not in out


def test_redaction_email():
    input_text = "contact: alice@example.com"
    out = redact(input_text)
    assert "alice@example.com" not in out


def test_redaction_preserves_normal_text():
    input_text = "the issue is a bug in the parser"
    out = redact(input_text)
    assert out == "the issue is a bug in the parser"
