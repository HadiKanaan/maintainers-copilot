FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml /app/
RUN pip install --no-cache-dir fastapi uvicorn spacy openai structlog torch transformers && python -m spacy download en_core_web_sm
COPY . /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
