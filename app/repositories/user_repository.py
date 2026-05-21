# Purpose: SQL-only user data access.
# Significance: Keeps DB logic out of services and APIs.
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.repositories.models import UserORM


# Fetch a user by email address.
async def get_user_by_email(session: AsyncSession, email: str) -> Optional[UserORM]:
    q = select(UserORM).where(UserORM.email == email)
    res = await session.execute(q)
    return res.scalars().first()


# Create a user row with hashed password.
async def create_user(session: AsyncSession, email: str, hashed_password: str, role: str = "user") -> UserORM:
    u = UserORM(email=email, hashed_password=hashed_password, role=role)
    session.add(u)
    await session.commit()
    await session.refresh(u)
    return u
