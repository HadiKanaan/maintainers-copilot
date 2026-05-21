# Purpose: SQL-only message data access.
# Significance: Stores and retrieves conversation messages.
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.models import MessageORM
from sqlalchemy import select
from typing import List


# Add a message to a conversation.
async def add_message(session: AsyncSession, conversation_id: int, role: str, content: str) -> MessageORM:
    m = MessageORM(conversation_id=conversation_id, role=role, content=content)
    session.add(m)
    await session.commit()
    await session.refresh(m)
    return m


# Fetch messages for a conversation.
async def get_messages(session: AsyncSession, conversation_id: int) -> List[MessageORM]:
    q = select(MessageORM).where(MessageORM.conversation_id == conversation_id).order_by(MessageORM.created_at)
    res = await session.execute(q)
    return res.scalars().all()
