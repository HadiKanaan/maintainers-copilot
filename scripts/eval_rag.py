# Purpose: Evaluate RAG retrieval and generation quality.
# Significance: CI gate for RAG performance.
import os
import json
import sys
from typing import List, Dict
import structlog
import yaml
import asyncio
from minio import Minio
from openai import AzureOpenAI
from app.infra.rag import hybrid_search
from app.infra.tracing import create_span

logger = structlog.get_logger()


# Load JSONL file into list of dicts.
def _load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file into list of dicts."""
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

    client.fput_object("evals", "eval_report_rag.json", report_path)


# Call Azure OpenAI chat completion using environment credentials.
def _chat_complete_env(messages: List[Dict[str, str]]) -> str:
    """Call Azure OpenAI chat completion using environment credentials."""
    with create_span("eval.llm_call", {"messages": str(messages)[:200]}):
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
        client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
        resp = client.chat.completions.create(model=deployment, messages=messages)
        return resp.choices[0].message.content


# Ask LLM judge if answer is fully grounded in chunks.
def _judge_faithfulness(answer: str, chunks: List[str]) -> bool:
    """Ask LLM judge if answer is fully grounded in chunks."""
    prompt = "Does this answer contain only information from these chunks? Answer yes or no."
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Chunks:\n{chunks}\n\nAnswer:\n{answer}"},
    ]
    content = _chat_complete_env(messages).strip().lower()
    return content.startswith("yes")


# Ask LLM judge for answer relevancy score between 0 and 1.
def _judge_relevancy(question: str, answer: str) -> float:
    """Ask LLM judge for answer relevancy score between 0 and 1."""
    prompt = "Is this answer relevant to the question? Score from 0 to 1."
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Question:\n{question}\n\nAnswer:\n{answer}"},
    ]
    content = _chat_complete_env(messages).strip()
    try:
        return float(content)
    except Exception:
        return 0.0


# Run async hybrid_search in a sync context for eval scripts.
def _hybrid_search_sync(question: str) -> List[Dict]:
    """Run async hybrid_search in a sync context for eval scripts."""
    return asyncio.run(hybrid_search(question, "default", top_k=10, label_filter=None))


# Run RAG evaluation and compare to thresholds.
def main() -> None:
    """Run RAG evaluation and compare to thresholds."""
    examples = _load_jsonl("data/golden_rag.jsonl")
    thresholds = _load_thresholds("eval_thresholds.yaml")

    hit_at_5_total = 0
    mrr_total = 0.0
    faith_yes = 0
    relevancy_scores = []

    for ex in examples:
        question = ex["question"]
        ground_truth = ex["ground_truth_chunks"]

        with create_span("eval.rag", {"question": question}):
            results = _hybrid_search_sync(question)

            top_chunks = [r["text"] for r in results]
            hit_at_5 = any(gt in top_chunks[:5] for gt in ground_truth)
            hit_at_5_total += 1 if hit_at_5 else 0

            # MRR@10
            rank = 0
            for idx, chunk in enumerate(top_chunks[:10], start=1):
                if chunk in ground_truth:
                    rank = idx
                    break
            if rank:
                mrr_total += 1.0 / rank

            # Generate answer using retrieved chunks.
            prompt = f"Chunks:\n{top_chunks}\n\nQuestion:\n{question}"
            answer = _chat_complete_env([
                {"role": "system", "content": "Answer the question using only the provided chunks."},
                {"role": "user", "content": prompt},
            ])

            if _judge_faithfulness(answer, top_chunks):
                faith_yes += 1

            relevancy_scores.append(_judge_relevancy(question, answer))

    total = len(examples) if examples else 1
    hit_at_5 = hit_at_5_total / total
    mrr_at_10 = mrr_total / total
    faithfulness = faith_yes / total
    answer_relevancy = sum(relevancy_scores) / total

    report = {
        "hit_at_5": hit_at_5,
        "mrr_at_10": mrr_at_10,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "passed": True,
    }

            # Threshold checks.
    if hit_at_5 < thresholds["rag"]["hit_at_5"]:
        report["passed"] = False
    if mrr_at_10 < thresholds["rag"]["mrr_at_10"]:
        report["passed"] = False
    if faithfulness < thresholds["rag"]["faithfulness"]:
        report["passed"] = False
    if answer_relevancy < thresholds["rag"]["answer_relevancy"]:
        report["passed"] = False

    report_path = "eval_report_rag.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _upload_report(report_path)

    if not report["passed"]:
        logger.error("eval.failed", report=report)
        sys.exit(1)

    logger.info("eval.passed", report=report)


if __name__ == "__main__":
    main()
