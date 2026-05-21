# Purpose: Load and run the fine-tuned DistilBERT classifier.
# Significance: Ensures model integrity and consistent inference for /classify.
import os
import json
import hashlib
from typing import Optional, Dict
import structlog
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from modelserver.redaction import redact

logger = structlog.get_logger()

_MODEL_DIR = "./models/distilbert-issue-classifier"
_MODEL_CARD = os.path.join(_MODEL_DIR, "MODEL_CARD.md")

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForSequenceClassification] = None

_LABELS = {0: "bug", 1: "feature", 2: "docs", 3: "question"}


# Compute SHA-256 hash for a file to verify integrity.
def _hash_file(path: str) -> str:
    """Compute SHA-256 hash of a file to verify integrity."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# Read expected model artifact hash from the model card.
def _read_model_card_hash() -> Optional[str]:
    """Extract the model artifact hash from MODEL_CARD.md."""
    if not os.path.exists(_MODEL_CARD):
        return None
    with open(_MODEL_CARD, "r", encoding="utf-8") as f:
        for line in f:
            if line.lower().startswith("model artifact hash:"):
                return line.split(":", 1)[1].strip()
    return None


# Locate the model weights file in the model directory.
def _weights_path() -> Optional[str]:
    """Locate the model weights file in the model directory."""
    bin_path = os.path.join(_MODEL_DIR, "pytorch_model.bin")
    safe_path = os.path.join(_MODEL_DIR, "model.safetensors")
    if os.path.exists(bin_path):
        return bin_path
    if os.path.exists(safe_path):
        return safe_path
    return None


# Load the classifier and verify the weights hash.
def init_classifier() -> None:
    """Load tokenizer and model with hash verification (hard fail on mismatch)."""
    global _tokenizer, _model
    if not os.path.exists(_MODEL_DIR):
        logger.warning("classifier.missing", model_dir=redact(_MODEL_DIR))
        return

    weights = _weights_path()
    if not weights:
        raise RuntimeError("Model weights not found in model directory")

    expected_hash = _read_model_card_hash()
    actual_hash = _hash_file(weights)
    if expected_hash and expected_hash != actual_hash:
        raise RuntimeError("Model artifact hash mismatch; refusing to start")

    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_DIR)
    _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_DIR)
    _model.eval()
    logger.info("classifier.loaded", weights_hash=actual_hash)


# Run inference and return label/confidence.
def classify(text: str) -> Dict[str, object]:
    """Run inference on input text and return label/confidence."""
    if _model is None or _tokenizer is None:
        # Placeholder response for pre-training development
        return {"label": "bug", "confidence": 1.0, "model": "placeholder"}

    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        conf, idx = torch.max(probs, dim=-1)
        label = _LABELS.get(int(idx.item()), "bug")
        return {"label": label, "confidence": float(conf.item()), "model": "distilbert-fine-tuned"}
