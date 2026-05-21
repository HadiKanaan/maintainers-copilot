# SECURITY

## Redaction Patterns
- API keys (Azure OpenAI keys, GitHub tokens): reason: prevent secret leakage
- Bearer tokens / Authorization headers: reason: prevent auth leaks
- Emails and personal data: reason: PII protection
- Passwords and password-like fields: reason: credentials

## What appears in real issue text that must not reach logs
- Secrets accidentally pasted in issue bodies
- Access tokens included by users
- Private email addresses

## Where sensitive data could leak
- Logs (structured logs and stdout)
- Traces (LangSmith spans)
- Memory writes (long_term_memory, RAG chunks)
- RAG chunk store (if raw text persisted without redaction)
 - Tool input/output (if not redacted before tracing)

## How the redaction test proves coverage
- Unit tests cover API keys, GitHub tokens, emails, and ensure normal text is preserved
