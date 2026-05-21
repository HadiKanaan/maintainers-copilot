# Purpose: Fine-tune DistilBERT classifier and log with MLflow.
# Significance: Produces the production-ready issue classifier.
import os
import json
import hashlib
from typing import Dict, List
import structlog
import mlflow
from datasets import Dataset
from transformers import DistilBertTokenizerFast, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
from minio import Minio

logger = structlog.get_logger()

LABEL_TO_ID = {"bug": 0, "feature": 1, "docs": 2, "question": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


# Compute SHA-256 hash of a file for integrity tracking.
def _sha256_file(path: str) -> str:
    """Compute SHA-256 hash of a file for integrity tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# Load a JSONL file into a list of dicts.
def _load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


# Tokenize a batch of texts for DistilBERT.
def _tokenize(tokenizer: DistilBertTokenizerFast, batch: Dict) -> Dict:
    """Tokenize a batch of texts for DistilBERT."""
    return tokenizer(
        batch["text"],
        truncation=True,
        padding=True,
        max_length=128,
    )


# Compute accuracy and F1 metrics for evaluation.
def _compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute accuracy and F1 metrics for evaluation."""
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    results = {}
    results.update(acc.compute(predictions=preds, references=labels))
    results["f1_macro"] = f1.compute(predictions=preds, references=labels, average="macro", labels=[0, 1, 2, 3])["f1"]
    per_class = f1.compute(predictions=preds, references=labels, average=None, labels=[0, 1, 2, 3])["f1"]
    for label, idx in LABEL_TO_ID.items():
        results[f"f1_{label}"] = per_class[idx]
    return results


# Write the model card with training metadata and hashes.
def _save_model_card(model_dir: str, train_hash: str, metrics: Dict[str, float], weights_hash: str) -> None:
    """Write the model card with training metadata and hashes."""
    card_path = os.path.join(model_dir, "MODEL_CARD.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write("Architecture: distilbert-base-uncased\n")
        f.write("num_labels: 4\n")
        f.write("Label mapping: bug=0, feature=1, docs=2, question=3\n")
        f.write("Hyperparameters:\n")
        f.write("- num_train_epochs: 3\n")
        f.write("- per_device_train_batch_size: 16\n")
        f.write("- per_device_eval_batch_size: 32\n")
        f.write("- learning_rate: 2e-5\n")
        f.write("- weight_decay: 0.01\n")
        f.write("- evaluation_strategy: epoch\n")
        f.write("- save_strategy: epoch\n")
        f.write("- load_best_model_at_end: True\n")
        f.write("- metric_for_best_model: f1_macro\n")
        f.write(f"Training data hash: {train_hash}\n")
        f.write(f"Final accuracy: {metrics.get('accuracy')}\n")
        f.write(f"Final macro-F1: {metrics.get('f1_macro')}\n")
        f.write(f"Final per-class F1: bug={metrics.get('f1_bug')}, feature={metrics.get('f1_feature')}, docs={metrics.get('f1_docs')}, question={metrics.get('f1_question')}\n")
        f.write(f"Model artifact hash: {weights_hash}\n")


# Upload the model directory to MinIO bucket 'models'.
def _upload_to_minio(model_dir: str) -> None:
    """Upload the model directory to MinIO bucket 'models'."""
    endpoint = os.environ.get("MINIO_ENDPOINT")
    access_key = os.environ.get("MINIO_ROOT_USER")
    secret_key = os.environ.get("MINIO_ROOT_PASSWORD")
    if not endpoint or not access_key or not secret_key:
        raise RuntimeError("Missing MinIO credentials in environment")

    client = Minio(
        endpoint.replace("http://", "").replace("https://", ""),
        access_key=access_key,
        secret_key=secret_key,
        secure=endpoint.startswith("https"),
    )

    if not client.bucket_exists("models"):
        client.make_bucket("models")

    for root, _, files in os.walk(model_dir):
        for fname in files:
            file_path = os.path.join(root, fname)
            obj_name = os.path.relpath(file_path, model_dir)
            client.fput_object("models", f"distilbert-issue-classifier/{obj_name}", file_path)


# Run the full training pipeline and upload the model.
def main() -> None:
    """Run the full training pipeline and upload the model."""
    train_path = "data/issues_train.jsonl"
    val_path = "data/issues_val.jsonl"

    train_items = _load_jsonl(train_path)
    val_items = _load_jsonl(val_path)

    for item in train_items:
        item["label"] = LABEL_TO_ID[item["mapped_label"]]
        item["text"] = f"{item.get('title', '')}\n\n{item.get('body', '')}"
    for item in val_items:
        item["label"] = LABEL_TO_ID[item["mapped_label"]]
        item["text"] = f"{item.get('title', '')}\n\n{item.get('body', '')}"

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_ds = Dataset.from_list(train_items).map(lambda b: _tokenize(tokenizer, b), batched=True)
    val_ds = Dataset.from_list(val_items).map(lambda b: _tokenize(tokenizer, b), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)

    args = TrainingArguments(
        output_dir="./models/distilbert-issue-classifier",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )

    # Non-obvious block: configure Trainer for eval/checkpointing per spec.
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    mlflow.set_tracking_uri("./mlruns")
    with mlflow.start_run():
        mlflow.log_params({
            "num_train_epochs": 3,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 32,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1_macro",
        })

        trainer.train()
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)

    model_dir = "./models/distilbert-issue-classifier"
    trainer.save_model(model_dir)

    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_dir, "model.safetensors")

    weights_hash = _sha256_file(weights_path)
    train_hash = _sha256_file(train_path)

    _save_model_card(model_dir, train_hash, {
        "accuracy": metrics.get("eval_accuracy"),
        "f1_macro": metrics.get("eval_f1_macro"),
        "f1_bug": metrics.get("eval_f1_bug"),
        "f1_feature": metrics.get("eval_f1_feature"),
        "f1_docs": metrics.get("eval_f1_docs"),
        "f1_question": metrics.get("eval_f1_question"),
    }, weights_hash)

    _upload_to_minio(model_dir)
    logger.info("training.complete", model_dir=model_dir, weights_hash=weights_hash)


if __name__ == "__main__":
    main()
