# Purpose: SQLAlchemy ORM models for core tables.
# Significance: Defines DB schema for Alembic and repository queries.
from sqlalchemy import Column, Integer, String, DateTime, Enum, ForeignKey, Text
from pgvector.sqlalchemy import Vector
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base
import enum

Base = declarative_base()


class RoleEnum(str, enum.Enum):
    user = "user"
    admin = "admin"


class UserORM(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(320), unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(RoleEnum), default=RoleEnum.user, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ConversationORM(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class MessageORM(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class LongTermMemoryORM(Base):
    __tablename__ = "long_term_memory"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    memory_type = Column(String(64), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AuditLogORM(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True, index=True)
    actor_id = Column(Integer, nullable=True)
    action = Column(String(256), nullable=False)
    target = Column(String(256), nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
