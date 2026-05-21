FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml /app/
RUN pip install --no-cache-dir fastapi uvicorn httpx sqlalchemy[asyncio] asyncpg alembic psycopg2-binary pydantic structlog passlib[bcrypt] python-jose openai langsmith
COPY . /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
