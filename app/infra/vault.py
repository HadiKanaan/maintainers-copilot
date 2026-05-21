# Purpose: Vault adapter to seed and retrieve secrets in dev mode.
# Significance: Single place for secret access so other layers stay clean.
import os
import httpx
from typing import Optional

VAULT_ADDR = os.environ.get("VAULT_ADDR", "http://localhost:8200")
VAULT_ROOT_TOKEN = os.environ.get("VAULT_ROOT_TOKEN")

_client: Optional[httpx.Client] = None


# Initialize Vault client and seed secrets for dev mode.
def init_vault() -> None:
    global _client
    if not VAULT_ROOT_TOKEN:
        raise RuntimeError("VAULT_ROOT_TOKEN is not set in environment")
    _client = httpx.Client(base_url=VAULT_ADDR, headers={"X-Vault-Token": VAULT_ROOT_TOKEN}, timeout=10.0)

    # quick health check
    r = _client.get("/v1/sys/health")
    if r.status_code >= 400:
        raise RuntimeError(f"Vault health check failed: {r.status_code} {r.text}")

    # Seed secrets from environment (dev convenience)
    # Only seed well-known secret env vars or those that don't reference Vault paths
    for k, v in list(os.environ.items()):
        if not v:
            continue
        # Skip vault control vars
        if k in ("VAULT_ADDR", "VAULT_ROOT_TOKEN"):
            continue
        # Skip values that intentionally point to Vault (e.g. VAULT:secret/...)
        if isinstance(v, str) and v.startswith("VAULT:"):
            continue
        # Seed a secret at path secret/<key>
        try:
            _client.put(f"/v1/secret/{k}", json={"value": v})
        except Exception:
            # best-effort seeding; fail only on connection issues (handled above)
            pass


# Fetch a secret value by key from Vault.
def get_secret(key: str) -> str:
    """Retrieve a secret from Vault at secret/<key>.

    Args:
        key: secret name
    Returns:
        secret value
    Raises:
        RuntimeError if secret not found or Vault unreachable
    """
    global _client
    if _client is None:
        raise RuntimeError("Vault client not initialized")
    # Try KV v1 at /v1/secret/<key>
    r = _client.get(f"/v1/secret/{key}")
    if r.status_code == 200:
        try:
            payload = r.json()
            # kv v1 returns {"data": {"value": ...}}
            if "data" in payload and "value" in payload["data"]:
                return payload["data"]["value"]
            # older dev mode may return value directly
            if "value" in payload:
                return payload["value"]
        except Exception:
            pass
    # Try KV v2 path /v1/secret/data/<key>
    r = _client.get(f"/v1/secret/data/{key}")
    if r.status_code == 200:
        payload = r.json()
        return payload.get("data", {}).get("data", {}).get("value")
    raise RuntimeError(f"Secret not found in Vault: {key}")
