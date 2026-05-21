# Purpose: Alembic env wiring for migration context.
# Significance: Connects migrations to the SQLAlchemy metadata.
import os
from app.infra.vault import init_vault, get_secret
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

def get_url():
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    # Fallback: resolve from Vault
    init_vault()
    return get_secret("DATABASE_URL")

# Interpret the config file for Python logging.
fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
from app.repositories.models import Base

target_metadata = Base.metadata


def run_migrations_offline():
    url = get_url()
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        url=get_url(),
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
