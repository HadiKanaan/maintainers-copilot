# Purpose: Evaluate LLM zero-shot baseline on test set.
# Significance: Provides baseline metrics for comparison.
import os
import json
import time
from typing import Dict, List
import structlog
from openai import AzureOpenAI
from app.infra.tracing import create_span
from sklearn.metrics import f1_score, accuracy_score

logger = structlog.get_logger()

LABELS = ["bug", "feature", "docs", "question"]


# Load JSONL file into list of dicts.
def _load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file into list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


# Extract metrics from the model card for comparison table.
def _load_model_card_metrics(path: str) -> Dict[str, float]:
    """Extract metrics from the model card for comparison table."""
    metrics = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.lower().startswith("final accuracy:"):
                metrics["accuracy"] = float(line.split(":", 1)[1].strip() or 0)
            if line.lower().startswith("final macro-f1:"):
                metrics["macro_f1"] = float(line.split(":", 1)[1].strip() or 0)
    return metrics


# Estimate cost using the provided token pricing formula.
def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost using the provided token pricing formula."""
    return (prompt_tokens * 0.000002) + (completion_tokens * 0.000002)


# Run zero-shot classification and write results JSON.
def main() -> None:
    """Run zero-shot classification and write results JSON."""
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

    client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)

    items = _load_jsonl("data/issues_test.jsonl")
    y_true = []
    y_pred = []
    latencies = []
    total_cost = 0.0

    system_prompt = (
        "You are an issue classifier. Classify the following GitHub issue into exactly "
        "one of these categories: bug, feature, docs, question. "
        "Respond with only the category name, nothing else."
    )

    for it in items:
        text = f"{it.get('title', '')}\n\n{it.get('body', '')}"
        start = time.time()
        with create_span("llm_baseline.request", {"text": text[:200]}):
            resp = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            )
        latency = time.time() - start
        latencies.append(latency)

        content = resp.choices[0].message.content.strip().lower()
        if content not in LABELS:
            content = "unknown"
        y_pred.append(content)
        y_true.append(it.get("mapped_label"))

        usage = resp.usage
        if usage:
            total_cost += _estimate_cost(usage.prompt_tokens, usage.completion_tokens)

    # Compute metrics (skip unknown in F1 by mapping to valid labels only).
    filtered_true = [t for t, p in zip(y_true, y_pred) if p in LABELS]
    filtered_pred = [p for p in y_pred if p in LABELS]

    acc = accuracy_score(filtered_true, filtered_pred) if filtered_true else 0.0
    macro_f1 = f1_score(filtered_true, filtered_pred, average="macro") if filtered_true else 0.0
    per_class_f1 = f1_score(filtered_true, filtered_pred, labels=LABELS, average=None) if filtered_true else [0, 0, 0, 0]

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    model_card_metrics = _load_model_card_metrics("./models/distilbert-issue-classifier/MODEL_CARD.md")

    table = {
        "fine_tuned": {
            "accuracy": model_card_metrics.get("accuracy"),
            "macro_f1": model_card_metrics.get("macro_f1"),
        },
        "llm_baseline": {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "latency_avg": avg_latency,
            "cost_est": total_cost,
            "per_class_f1": dict(zip(LABELS, per_class_f1)),
        },
    }

    # Non-obvious block: human-readable comparison summary for quick inspection.
    summary_table = (
        "Model | Accuracy | Macro-F1 | Latency (avg) | Cost (est.)\n"
        f"Fine-tuned DBERT | {table['fine_tuned']['accuracy']} | {table['fine_tuned']['macro_f1']} | N/A | N/A\n"
        f"LLM baseline | {acc} | {macro_f1} | {avg_latency} | {total_cost}"
    )

    with open("scripts/llm_baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)

    logger.info("llm_baseline.complete", results=table, comparison=summary_table)


if __name__ == "__main__":
    main()
