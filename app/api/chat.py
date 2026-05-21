# Purpose: Chat API router for sending messages and managing conversations.
# Significance: Exposes chat endpoints without business logic.
from fastapi import APIRouter, Depends
from app.api.auth import get_current_user
from app.domain.models import ChatRequest, ChatResponse, Conversation
from app.services.chat_service import process_message, start_conversation, list_conversations

router = APIRouter(prefix="/chat", tags=["chat"])


# Post a chat message and return the assistant response.
@router.post("/message", response_model=ChatResponse)
async def post_message(req: ChatRequest, user=Depends(get_current_user)) -> ChatResponse:
    """Send a message to the chat service and return response."""
    return await process_message(user.id, req.conversation_id, req.message)


# Create a new conversation for the current user.
@router.post("/conversations", response_model=Conversation)
async def create_conversation(user=Depends(get_current_user)) -> Conversation:
    """Create a new conversation for the authenticated user."""
    return await start_conversation(user.id)


# List conversations for the current user.
@router.get("/conversations", response_model=list[Conversation])
async def get_conversations(user=Depends(get_current_user)) -> list[Conversation]:
    """List conversations for the authenticated user."""
    return await list_conversations(user.id)
