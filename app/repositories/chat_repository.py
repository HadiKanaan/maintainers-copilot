# Purpose: SQL-only chat repository functions for conversations, messages, memory, audit.
# Significance: Keeps database access separate from API/service logic.
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.repositories.models import ConversationORM, MessageORM, LongTermMemoryORM, AuditLogORM
from app.domain.models import Conversation, Message, LongTermMemory


# Create a conversation row and return a domain model.
async def create_conversation(session: AsyncSession, user_id: int) -> Conversation:
    """Create a conversation row and return a domain model."""
    conv = ConversationORM(user_id=user_id)
    session.add(conv)
    await session.commit()
    await session.refresh(conv)
    return Conversation(id=conv.id, user_id=conv.user_id, created_at=conv.created_at, updated_at=conv.updated_at)


# Fetch all conversations for a user.
async def get_conversations(session: AsyncSession, user_id: int) -> List[Conversation]:
    """Fetch all conversations for a user."""
    q = select(ConversationORM).where(ConversationORM.user_id == user_id)
    res = await session.execute(q)
    return [Conversation(id=c.id, user_id=c.user_id, created_at=c.created_at, updated_at=c.updated_at) for c in res.scalars().all()]


# Persist a message and return the domain model.
async def save_message(session: AsyncSession, conversation_id: int, role: str, content: str) -> Message:
    """Persist a message and return the domain model."""
    msg = MessageORM(conversation_id=conversation_id, role=role, content=content)
    session.add(msg)
    await session.commit()
    await session.refresh(msg)
    return Message(id=msg.id, conversation_id=msg.conversation_id, role=msg.role, content=msg.content, created_at=msg.created_at)


# Fetch messages for a conversation.
async def get_messages(session: AsyncSession, conversation_id: int) -> List[Message]:
    """Fetch messages for a conversation."""
    q = select(MessageORM).where(MessageORM.conversation_id == conversation_id).order_by(MessageORM.created_at)
    res = await session.execute(q)
    return [Message(id=m.id, conversation_id=m.conversation_id, role=m.role, content=m.content, created_at=m.created_at) for m in res.scalars().all()]


# Persist a long-term memory row and return the domain model.
async def save_long_term_memory(session: AsyncSession, user_id: int, content: str, embedding: List[float]) -> LongTermMemory:
    """Persist a long-term memory row and return the domain model."""
    mem = LongTermMemoryORM(user_id=user_id, memory_type="semantic", content=content, embedding=embedding)
    session.add(mem)
    await session.commit()
    await session.refresh(mem)
    return LongTermMemory(id=mem.id, user_id=mem.user_id, memory_type=mem.memory_type, content=mem.content, created_at=mem.created_at)


# Cosine similarity search over long_term_memory for a user.
async def search_long_term_memory(session: AsyncSession, user_id: int, query_embedding: List[float], top_k: int) -> List[LongTermMemory]:
    """Cosine similarity search over long_term_memory for a user."""
    q = (
        select(LongTermMemoryORM)
        .where(LongTermMemoryORM.user_id == user_id)
        .order_by(LongTermMemoryORM.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )
    res = await session.execute(q)
    return [LongTermMemory(id=m.id, user_id=m.user_id, memory_type=m.memory_type, content=m.content, created_at=m.created_at) for m in res.scalars().all()]


# Write an audit_log row for sensitive actions.
async def write_audit_log(session: AsyncSession, actor_id: int, action: str, target: str) -> None:
    """Write an audit_log row for sensitive actions."""
    log_row = AuditLogORM(actor_id=actor_id, action=action, target=target)
    session.add(log_row)
    await session.commit()
