# Purpose: Auth API router for register/login and auth dependency.
# Significance: Defines HTTP surface for authentication without business logic.
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr
from app.services.auth_service import register_user, authenticate_user, get_user_from_token
from app.domain.models import TokenResponse, User as DomainUser

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# Register a new user with email/password.
@router.post("/register", response_model=DomainUser)
async def register(req: RegisterRequest):
    try:
        u = await register_user(req.email, req.password)
        return u
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Authenticate a user and return a JWT.
@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    token = await authenticate_user(req.email, req.password)
    if token is None:
        raise HTTPException(status_code=401, detail="invalid credentials")
    return TokenResponse(access_token=token)


# Resolve the current user from a Bearer token.
async def get_current_user(authorization: str | None = Header(None)) -> DomainUser:
    """Dependency: extract user from Bearer token and return DomainUser"""
    if not authorization:
        raise HTTPException(status_code=401, detail="missing authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        token = authorization
    user = await get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="invalid token")
    return user
