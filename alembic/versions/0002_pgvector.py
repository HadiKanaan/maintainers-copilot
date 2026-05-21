# Purpose: Enable pgvector extension.
# Significance: Supports vector storage for embeddings.
"""add pgvector extension

Revision ID: 0002_pgvector
Revises: 0001_initial
Create Date: 2026-05-19
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = '0002_pgvector'
down_revision = '0001_initial'
branch_labels = None
depends_on = None


def upgrade():
    op.execute('CREATE EXTENSION IF NOT EXISTS vector;')
    op.alter_column('long_term_memory', 'embedding', type_=Vector(1536))


def downgrade():
    op.execute('DROP EXTENSION IF EXISTS vector;')
