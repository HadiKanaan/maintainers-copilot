# Purpose: Modelserver API for classify/NER/summarize endpoints.
# Significance: Isolates ML workloads from the main API service.
import os
import structlog
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import spacy
from openai import AzureOpenAI
from modelserver.classifier import classify, init_classifier
from modelserver.tracing import create_span
from modelserver.redaction import redact

logger = structlog.get_logger()

app = FastAPI()

nlp = None
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    # Non-obvious block: spaCy model is optional until installed in dev.
    nlp = None


class TextIn(BaseModel):
    """Text input payload for classification and NER."""
    text: str


class ClassifyOut(BaseModel):
    """Classifier output payload."""
    label: str
    confidence: float
    model: str


class NEROut(BaseModel):
    """NER output payload."""
    entities: List[Dict]


class SummarizeIn(BaseModel):
    """Summarization input payload."""
    text: str
    max_length: int = 200


# Create Azure OpenAI client from environment variables.
def _get_azure_client() -> AzureOpenAI:
    """Create Azure OpenAI client from environment variables."""
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)


@app.on_event("startup")
# Initialize classifier on startup and verify weights.
def startup_event() -> None:
    """Verify model weights on startup and initialize classifier."""
    with create_span("modelserver.startup", {}):
        init_classifier()
        logger.info("modelserver.started")


@app.get("/health")
# Report service health for startup checks.
async def health():
    """Health endpoint used by API startup checks."""
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyOut)
# Classify an issue via the fine-tuned model.
async def classify_issue(payload: TextIn):
    """Classify an issue using the fine-tuned model."""
    with create_span("modelserver.classify", {"text": redact(payload.text)}):
        result = classify(payload.text)
        return ClassifyOut(**result)


@app.post("/ner", response_model=NEROut)
# Extract entities from text using spaCy.
async def ner(payload: TextIn):
    """Extract entities from text using spaCy."""
    with create_span("modelserver.ner", {"text": redact(payload.text)}):
        if nlp is None:
            return {"entities": []}
        doc = nlp(payload.text)
        entities = []
        for ent in doc.ents:
            entities.append({"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char})
        return {"entities": entities}


@app.post("/summarize")
# Summarize issue text via Azure OpenAI.
async def summarize(payload: SummarizeIn):
    """Summarize a GitHub issue using Azure OpenAI."""
    with create_span("modelserver.summarize", {"text": redact(payload.text)}):
        client = _get_azure_client()
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        system_prompt = "Summarize this GitHub issue in 2-3 sentences."
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload.text},
            ],
        )
        summary = resp.choices[0].message.content
        return {"summary": summary}
