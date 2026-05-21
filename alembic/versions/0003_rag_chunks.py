# Purpose: Add rag_chunks table for RAG storage.
# Significance: Enables hybrid retrieval over stored chunks.
"""add rag_chunks table

Revision ID: 0003_rag_chunks
Revises: 0002_pgvector
Create Date: 2026-05-20
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0003_rag_chunks"
down_revision = "0002_pgvector"
branch_labels = None
depends_on = None


def upgrade():
    """Create rag_chunks table to store embedded chunks."""
    op.create_table(
        "rag_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("collection", sa.String(length=128), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("issue_id", sa.String(length=64), nullable=True),
        sa.Column("label", sa.String(length=32), nullable=True),
        sa.Column("date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )


def downgrade():
    """Drop rag_chunks table."""
    op.drop_table("rag_chunks")
