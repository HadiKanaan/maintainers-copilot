# Purpose: Auth business logic (hashing, register, login).
# Significance: Enforces authentication rules and JWT creation via Vault.
from typing import Optional
from passlib.context import CryptContext
from jose import jwt, JWTError
from app.infra.vault import get_secret
from app.repositories.user_repository import get_user_by_email, create_user
from app.domain.models import User
from app.db import AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
JWT_ALGO = "HS256"


# Hash a plaintext password using bcrypt.
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


# Verify a plaintext password against a hash.
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# Register a user and return the domain model.
async def register_user(email: str, password: str, role: str = "user") -> User:
    async with AsyncSessionLocal() as session:  # type: ignore
        existing = await get_user_by_email(session, email)
        if existing:
            raise ValueError("user exists")
        hashed = hash_password(password)
        u = await create_user(session, email, hashed, role)
        return User(id=u.id, email=u.email, role=u.role, created_at=u.created_at)


# Authenticate credentials and return JWT if valid.
async def authenticate_user(email: str, password: str) -> Optional[str]:
    async with AsyncSessionLocal() as session:  # type: ignore
        u = await get_user_by_email(session, email)
        if not u:
            return None
        if not verify_password(password, u.hashed_password):
            return None
        payload = {"sub": u.email, "uid": u.id}
        jwt_secret = get_secret("JWT_SECRET_KEY")
        token = jwt.encode(payload, jwt_secret, algorithm=JWT_ALGO)
        return token


# Decode a JWT and return the user domain model.
async def get_user_from_token(token: str) -> Optional[User]:
    """Decode a JWT and return the associated user domain model."""
    try:
        secret = get_secret("JWT_SECRET_KEY")
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGO])
        email = payload.get("sub")
        if not email:
            return None
    except JWTError:
        return None

    async with AsyncSessionLocal() as session:  # type: ignore
        u = await get_user_by_email(session, email)
        if not u:
            return None
        return User(id=u.id, email=u.email, role=u.role, created_at=u.created_at)
