# Purpose: Domain exception hierarchy and FastAPI handler.
# Significance: Ensures structured errors without stack traces to users.
from typing import Optional
from fastapi import Request
from fastapi.responses import JSONResponse


# Base domain exception for the API.
class AppError(Exception):
    def __init__(self, message: str, code: str = "app_error", http_status: int = 500):
        super().__init__(message)
        self.message = message
        self.code = code
        self.http_status = http_status


# Raised when a requested resource is missing.
class NotFoundError(AppError):
    def __init__(self, message: str = "not found"):
        super().__init__(message, code="not_found", http_status=404)


# Raised when access is forbidden.
class PermissionDenied(AppError):
    def __init__(self, message: str = "permission denied"):
        super().__init__(message, code="permission_denied", http_status=403)


# Raised when an external tool call fails.
class ToolFailure(AppError):
    def __init__(self, message: str = "tool failure"):
        super().__init__(message, code="tool_failure", http_status=502)


# Raised on authentication failures.
class AuthError(AppError):
    def __init__(self, message: str = "authentication failed"):
        super().__init__(message, code="auth_error", http_status=401)


# Raised when user input fails validation.
class ValidationError(AppError):
    def __init__(self, message: str = "validation error"):
        super().__init__(message, code="validation_error", http_status=400)


# Convert domain exceptions into structured JSON responses.
async def exception_handler(request: Request, exc: Exception):
    request_id = request.headers.get("X-Request-Id") or "-"
    if isinstance(exc, AppError):
        payload = {"error": {"code": exc.code, "message": exc.message, "request_id": request_id}}
        return JSONResponse(status_code=exc.http_status, content=payload)
    # Generic error
    payload = {"error": {"code": "internal_error", "message": "internal error", "request_id": request_id}}
    return JSONResponse(status_code=500, content=payload)
