# Purpose: SQL-only long-term memory data access.
# Significance: Persists and fetches user memory entries.
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.models import LongTermMemoryORM
from sqlalchemy import select
from typing import List


# Add a long-term memory row for a user.
async def add_memory(session: AsyncSession, user_id: int, memory_type: str, content: str, embedding: list[float]) -> LongTermMemoryORM:
    m = LongTermMemoryORM(user_id=user_id, memory_type=memory_type, content=content, embedding=embedding)
    session.add(m)
    await session.commit()
    await session.refresh(m)
    return m


# Fetch all memories for a user.
async def get_memories_for_user(session: AsyncSession, user_id: int) -> List[LongTermMemoryORM]:
    q = select(LongTermMemoryORM).where(LongTermMemoryORM.user_id == user_id).order_by(LongTermMemoryORM.created_at)
    res = await session.execute(q)
    return res.scalars().all()
