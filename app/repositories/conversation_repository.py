# Purpose: SQL-only conversation data access.
# Significance: Provides CRUD helpers for conversation records.
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.models import ConversationORM
from sqlalchemy import select
from typing import Optional


# Create a conversation record for a user.
async def create_conversation(session: AsyncSession, user_id: int) -> ConversationORM:
    c = ConversationORM(user_id=user_id)
    session.add(c)
    await session.commit()
    await session.refresh(c)
    return c


# Fetch a conversation by id.
async def get_conversation(session: AsyncSession, conv_id: int) -> Optional[ConversationORM]:
    q = select(ConversationORM).where(ConversationORM.id == conv_id)
    res = await session.execute(q)
    return res.scalars().first()
