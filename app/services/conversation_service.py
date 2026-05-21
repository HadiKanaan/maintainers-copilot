# Purpose: Conversation workflow orchestration.
# Significance: Keeps business logic out of the API layer.
from typing import List
from app.repositories.conversation_repository import create_conversation, get_conversation
from app.repositories.message_repository import add_message, get_messages
from app.db import AsyncSessionLocal


# Start a new conversation for a user.
async def start_conversation(user_id: int):
    async with AsyncSessionLocal() as session:
        conv = await create_conversation(session, user_id)
        return conv


# Append a message to a conversation.
async def append_message(conversation_id: int, role: str, content: str):
    async with AsyncSessionLocal() as session:
        msg = await add_message(session, conversation_id, role, content)
        return msg


# Fetch messages for a conversation.
async def fetch_messages(conversation_id: int):
    async with AsyncSessionLocal() as session:
        msgs = await get_messages(session, conversation_id)
        return msgs
