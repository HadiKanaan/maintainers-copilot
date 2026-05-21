# Purpose: MinIO adapter for object storage access and checks.
# Significance: Centralizes model artifact verification on startup.
from minio import Minio
from minio.error import S3Error
from typing import Optional
from app.infra.vault import get_secret

_client: Optional[Minio] = None


# Initialize MinIO client using Vault-sourced credentials.
def init_minio_client() -> Minio:
    global _client
    if _client:
        return _client
    endpoint = get_secret("MINIO_ENDPOINT")
    access_key = get_secret("MINIO_ROOT_USER")
    secret_key = get_secret("MINIO_ROOT_PASSWORD")
    # strip protocol for Minio client
    url = endpoint.replace("http://", "").replace("https://", "")
    _client = Minio(url, access_key=access_key, secret_key=secret_key, secure=endpoint.startswith("https"))
    return _client


# Check whether an object exists in MinIO.
def object_exists(bucket: str, object_name: str) -> bool:
    client = init_minio_client()
    try:
        client.stat_object(bucket, object_name)
        return True
    except S3Error:
        return False
