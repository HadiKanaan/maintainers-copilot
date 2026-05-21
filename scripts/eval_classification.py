# Purpose: Evaluate classifier against golden set and enforce thresholds.
# Significance: CI gate for classification quality.
import os
import json
import sys
from typing import List, Dict
import structlog
import httpx
import yaml
from sklearn.metrics import f1_score, confusion_matrix
from minio import Minio

logger = structlog.get_logger()


# Load JSONL file into a list of dicts.
def _load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file into a list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            items.append(json.loads(line))
    return items


# Load evaluation thresholds from YAML.
def _load_thresholds(path: str) -> Dict:
    """Load evaluation thresholds from YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Upload evaluation report to MinIO bucket 'evals'.
def _upload_report(report_path: str) -> None:
    """Upload evaluation report to MinIO bucket 'evals'."""
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

    if not client.bucket_exists("evals"):
        client.make_bucket("evals")

    client.fput_object("evals", "eval_report_classification.json", report_path)


# Run evaluation and compare results to thresholds.
def main() -> None:
    """Run evaluation and compare results to thresholds."""
    modelserver_url = os.environ.get("MODELSERVER_URL", "http://localhost:8001")
    examples = _load_jsonl("data/golden_classification.jsonl")

    y_true = []
    y_pred = []

    with httpx.Client(timeout=30.0) as client:
        for ex in examples:
            r = client.post(f"{modelserver_url}/classify", json={"text": ex["text"]})
            r.raise_for_status()
            label = r.json().get("label")
            y_true.append(ex["expected_label"])
            y_pred.append(label)

    labels = ["bug", "feature", "docs", "question"]
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro")
    per_class = f1_score(y_true, y_pred, labels=labels, average=None)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    thresholds = _load_thresholds("eval_thresholds.yaml")
    pass_fail = True

    if macro_f1 < thresholds["classification"]["macro_f1"]:
        pass_fail = False

    for lbl, f1_val in zip(labels, per_class):
        if f1_val < thresholds["classification"]["per_class_f1"][lbl]:
            pass_fail = False

    report = {
        "macro_f1": macro_f1,
        "per_class_f1": dict(zip(labels, per_class)),
        "confusion_matrix": cm,
        "passed": pass_fail,
    }

    report_path = "eval_report_classification.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _upload_report(report_path)

    if not pass_fail:
        logger.error("eval.failed", report=report)
        sys.exit(1)

    logger.info("eval.passed", report=report)


if __name__ == "__main__":
    main()
