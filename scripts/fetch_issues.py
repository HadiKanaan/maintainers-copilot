# Purpose: Fetch closed GitHub issues and build train/val/test datasets.
# Significance: Generates labeled data for classifier training/evaluation.
import os
import json
from datetime import datetime
from typing import List, Dict
from collections import Counter, defaultdict
import httpx

LABEL_MAPPING = {
    # Map GitHub labels to canonical classes (edit as needed)
    "bug": "bug",
    "bugfix": "bug",
    "enhancement": "feature",
    "feature": "feature",
    "documentation": "docs",
    "docs": "docs",
    "question": "question",
}

GITHUB_REPO = os.environ.get("GITHUB_REPO")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

HEADERS = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"


def map_labels(raw_labels: List[Dict]) -> List[str]:
    mapped = []
    for l in raw_labels:
        name = (l.get("name") or "").lower()
        if name in LABEL_MAPPING:
            mapped.append(LABEL_MAPPING[name])
    return mapped


def fetch_closed_issues() -> List[Dict]:
    if not GITHUB_REPO:
        raise RuntimeError("GITHUB_REPO not set")
    owner, repo = GITHUB_REPO.split("/")
    per_page = 100
    page = 1
    issues = []
    with httpx.Client(headers=HEADERS, timeout=30.0) as client:
        while True:
            url = f"https://api.github.com/repos/{owner}/{repo}/issues"
            params = {"state": "closed", "per_page": per_page, "page": page}
            r = client.get(url, params=params)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            for it in batch:
                if "pull_request" in it:
                    continue
                issues.append(it)
            page += 1
    return issues


def normalize_issue(it: Dict) -> Dict:
    mapped = map_labels(it.get("labels", []))
    return {
        "id": it.get("id"),
        "title": it.get("title"),
        "body": it.get("body") or "",
        "labels": [l.get("name") for l in it.get("labels", [])],
        "mapped_label": mapped[0] if mapped else None,
        "created_at": it.get("created_at"),
        "closed_at": it.get("closed_at"),
        "comments_count": it.get("comments", 0),
    }


def stratified_split(items: List[Dict]):
    # newest 15% as test (strictly more recent)
    items_sorted = sorted(items, key=lambda i: datetime.fromisoformat(i["created_at"].replace("Z", "+00:00")))
    n = len(items_sorted)
    if n == 0:
        return [], [], []
    test_count = max(1, int(n * 0.15))
    test = items_sorted[-test_count:]
    remaining = items_sorted[: n - test_count]

    # stratify remaining into train/val with 70/15 of total
    train_ratio = 0.70 / 0.85
    by_label = defaultdict(list)
    for it in remaining:
        by_label[it["mapped_label"]].append(it)

    train = []
    val = []
    for lbl, group in by_label.items():
        m = len(group)
        train_n = int(m * train_ratio)
        train.extend(group[:train_n])
        val.extend(group[train_n:])

    return train, val, test


def save_jsonl(path: str, items: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    issues = fetch_closed_issues()
    total = len(issues)
    normalized = []
    skipped = 0
    for i in issues:
        n = normalize_issue(i)
        if not n.get("mapped_label"):
            skipped += 1
            print(f"skipped issue {n.get('id')}: no mappable label")
            continue
        normalized.append(n)
    mapped = normalized

    train, val, test = stratified_split(mapped)

    os.makedirs("data", exist_ok=True)
    save_jsonl("data/issues_train.jsonl", train)
    save_jsonl("data/issues_val.jsonl", val)
    save_jsonl("data/issues_test.jsonl", test)

    def counts(lst: List[Dict]):
        return Counter([it.get("mapped_label") for it in lst])

    print(f"total fetched: {total}")
    print(f"skipped (no mappable label): {skipped}")
    print("train counts:", counts(train))
    print("val counts:", counts(val))
    print("test counts:", counts(test))


if __name__ == "__main__":
    main()
