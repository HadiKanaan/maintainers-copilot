# Purpose: Pydantic domain models for API requests and responses.
# Significance: Keeps API data separate from ORM models.
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime


class User(BaseModel):
    """User domain model for API responses."""
    id: Optional[int]
    email: EmailStr
    role: str = "user"
    created_at: Optional[datetime]


class Conversation(BaseModel):
    """Conversation domain model."""
    id: Optional[int]
    user_id: int
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


class Message(BaseModel):
    """Message domain model."""
    id: Optional[int]
    conversation_id: int
    role: str
    content: str
    created_at: Optional[datetime]


class LongTermMemory(BaseModel):
    """Long-term memory domain model."""
    id: Optional[int]
    user_id: int
    memory_type: str
    content: str
    created_at: Optional[datetime]


class TokenResponse(BaseModel):
    """JWT response payload."""
    access_token: str
    token_type: str = "bearer"


class ChatRequest(BaseModel):
    """Chat request payload."""
    conversation_id: str
    message: str


class ChatResponse(BaseModel):
    """Chat response payload."""
    response: str
    conversation_id: str
    trace_id: str


class ClassifyRequest(BaseModel):
    """Modelserver classify request."""
    text: str


class ClassifyResponse(BaseModel):
    """Modelserver classify response."""
    label: str
    confidence: float
    model: str


class Entity(BaseModel):
    """Named entity representation."""
    text: str
    label: str
    start: int
    end: int


class NERRequest(BaseModel):
    """Modelserver NER request."""
    text: str


class NERResponse(BaseModel):
    """Modelserver NER response."""
    entities: List[Entity]


class SummarizeRequest(BaseModel):
    """Modelserver summarize request."""
    text: str
    max_length: int


class SummarizeResponse(BaseModel):
    """Modelserver summarize response."""
    summary: str
