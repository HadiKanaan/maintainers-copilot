# Purpose: FastAPI entry point and startup wiring for infra checks.
# Significance: Centralizes app lifecycle, middleware, and router registration.
from fastapi import FastAPI, Request
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars
from app.infra.vault import init_vault, get_secret
from app.infra.tracing import init_tracing, get_trace_id
from app.infra.llm import init_llm_clients
from app.infra.exceptions import exception_handler
from app.api import auth as auth_router
from app.api import chat as chat_router
from app.infra.minio_client import init_minio_client, object_exists
import httpx

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()

app = FastAPI()


# Add request_id/trace_id to logs for each request.
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id", "-")
    trace_id = get_trace_id()
    bind_contextvars(request_id=request_id, trace_id=trace_id)
    logger.info("request.start", path=request.url.path, method=request.method)
    response = await call_next(request)
    logger.info("request.end", path=request.url.path, status_code=response.status_code)
    clear_contextvars()
    return response


# Initialize Vault, tracing, LLM clients, and startup checks.
@app.on_event("startup")
def startup_event():
    # Initialize Vault
    init_vault()
    # Initialize tracing
    init_tracing(get_secret("LANGSMITH_API_KEY"), get_secret("LANGSMITH_PROJECT"))
    # Initialize LLM clients
    init_llm_clients()
    # Verify classifier weights exist in MinIO
    try:
        init_minio_client()
        classifier_bucket = get_secret("CLASSIFIER_BUCKET")
        classifier_key = get_secret("CLASSIFIER_WEIGHTS_KEY")
        if not object_exists(classifier_bucket, classifier_key):
            raise RuntimeError("classifier weights not found in MinIO")
    except Exception as e:
        logger.error("startup.check_failed_minio", reason=str(e))
        raise
    # Verify modelserver health
    modelserver = get_secret("MODELSERVER_URL")
    try:
        r = httpx.get(f"{modelserver}/health", timeout=5.0)
        if r.status_code != 200:
            raise RuntimeError("modelserver health check failed")
    except Exception as e:
        logger.error("startup.check_failed", reason=str(e))
        raise


app.add_exception_handler(Exception, exception_handler)
app.include_router(auth_router.router)
app.include_router(chat_router.router)
