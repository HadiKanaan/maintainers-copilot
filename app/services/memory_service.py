# Purpose: Memory workflow orchestration.
# Significance: Central place for storing and retrieving memory.
from app.repositories.memory_repository import add_memory, get_memories_for_user
from app.db import AsyncSessionLocal
from app.infra.redaction import redact


# Store a long-term memory entry for a user.
async def store_memory(user_id: int, memory_type: str, content: str, embedding: list[float]):
    async with AsyncSessionLocal() as session:
        safe_content = redact(content)
        m = await add_memory(session, user_id, memory_type, safe_content, embedding)
        return m


# Fetch long-term memories for a user.
async def get_memories(user_id: int):
    async with AsyncSessionLocal() as session:
        mems = await get_memories_for_user(session, user_id)
        return mems
