#!/usr/bin/env python3
"""Evaluate Bedrock acceptance prediction on scraped ICLR markdown papers.

Expected input layout (from scrape_iclr_hf.py):
- output/iclr_2025_hf/papers/<paper_id>.json   with key: markdown
- output/iclr_2025_hf/reviews/<paper_id>.json  with key: accepted (+ optional reviews)

Example:
    python iclr_acceptance_eval.py
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import boto3


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "output" / "iclr_2018_hf"
PAPERS_DIR = DATASET_DIR / "papers"
REVIEWS_DIR = DATASET_DIR / "reviews"

NUM_SAMPLES = 100
RANDOM_SEED = 10718
BALANCED = True
HAS_REVIEWS = False
CONTEXT_LEN = 100000

# Prompt mode options: "neutral", "conservative", "severe_conservative".
PROMPT_MODE = "conservative"
CONFERENCE = "ICLR 2018"

DEFAULT_MODEL_ID = "us.meta.llama3-3-70b-instruct-v1:0"
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def format_reviews(reviews: List[Any]) -> str:
    parts: List[str] = []
    for i, r in enumerate(reviews, 1):
        if isinstance(r, dict):
            title = str(r.get("TITLE") or r.get("title") or f"Review {i}")
            comments = str(r.get("comments") or r.get("text") or "")
            meta = "(meta-review)" if r.get("IS_META_REVIEW") else ""
            parts.append(f"--- Review {i}: {title} {meta} ---")
            parts.append(comments)
            parts.append("")
        else:
            text = str(r)
            if text.strip():
                parts.append(f"--- Review {i} ---")
                parts.append(text)
                parts.append("")
    return "\n".join(parts).strip()


def build_prompt(markdown: str, reviews_text: Optional[str] = None) -> str:
    task = (
        f"You are an expert {CONFERENCE} area chair.\n\n"
        "Task:\n"
        f"- Read the following anonymized {CONFERENCE} paper in Markdown form.\n"
    )
    if reviews_text:
        task += (
            "- You are also given the official reviews for this submission.\n"
            "- Based on the paper content and the reviews, PREDICT whether it "
            f"was accepted to {CONFERENCE}.\n"
        )
    else:
        task += (
            "- Based ONLY on the content and quality of the paper, PREDICT whether it "
            f"was accepted to {CONFERENCE}.\n"
        )

    mode = PROMPT_MODE if PROMPT_MODE in {
        "neutral",
        "conservative",
        "severe_conservative",
    } else "severe_conservative"

    if mode == "neutral":
        policy_text = (
            "Decision policy (important):\n"
            "- Make your best single-shot guess from the available evidence.\n"
            "- If strengths and weaknesses are close, choose the more likely outcome.\n\n"
        )
    elif mode == "conservative":
        policy_text = (
            "Decision policy (important):\n"
            "- Be conservative: default to REJECT unless there is clear evidence for ACCEPT.\n"
            "- Borderline papers should be REJECT.\n"
            "- If uncertain, output REJECT.\n"
            "- If major concerns exist (novelty, technical correctness, empirical rigor, clarity, or significance), output REJECT.\n\n"
        )
    else:
        policy_text = (
            "Decision policy (important):\n"
            "- Use a strict desk-reject style prior: most submissions are reject unless exceptional evidence is present.\n"
            "- Default decision is REJECT.\n"
            "- Any meaningful uncertainty => REJECT.\n"
            "- Any major weakness in novelty, correctness, empirical rigor, clarity, significance, or reproducibility => REJECT.\n"
            "- Borderline, mixed, partially convincing, or under-justified papers => REJECT.\n"
            "- Output ACCEPT only if ALL criteria are clearly strong: novelty, technical soundness, strong empirical evidence, clear writing, and meaningful impact.\n"
            "- If one criterion is not clearly strong, output REJECT.\n\n"
        )

    prompt = (
        "- You do not know the true decision; you must guess.\n\n"
        f"{policy_text}"
        "Output format (important):\n"
        "- Respond with a single JSON object and NOTHING else.\n"
        "- The JSON must be exactly one of:\n"
        '  {"prediction": "ACCEPT"}\n'
        '  {"prediction": "REJECT"}\n\n'
        "Paper Markdown:\n"
        "---------------- BEGIN PAPER ----------------\n"
        f"{markdown[:CONTEXT_LEN]}\n"
        "----------------- END PAPER -----------------\n"
    )
    if reviews_text:
        prompt += (
            "\n\nOfficial Reviews:\n"
            "---------------- BEGIN REVIEWS ----------------\n"
            f"{reviews_text}\n"
            "----------------- END REVIEWS -----------------\n"
        )
    return task + prompt


def parse_prediction(full_text: str) -> Optional[Literal["ACCEPT", "REJECT"]]:
    try:
        parsed = json.loads(full_text)
        if isinstance(parsed, dict):
            pred = parsed.get("prediction")
            if isinstance(pred, str) and pred.strip().upper() in {"ACCEPT", "REJECT"}:
                return pred.strip().upper()  # type: ignore[return-value]
    except json.JSONDecodeError:
        pass

    matches = re.findall(
        r'\{\s*"prediction"\s*:\s*"(ACCEPT|REJECT)"\s*\}',
        full_text,
        re.IGNORECASE,
    )
    if matches:
        return matches[-1].upper()  # type: ignore[return-value]

    upper = full_text.upper()
    accept_idx = upper.rfind("ACCEPT")
    reject_idx = upper.rfind("REJECT")
    if accept_idx != -1 and reject_idx == -1:
        return "ACCEPT"
    if reject_idx != -1 and accept_idx == -1:
        return "REJECT"
    if accept_idx != -1 and reject_idx != -1:
        return "ACCEPT" if accept_idx > reject_idx else "REJECT"

    return None


def call_bedrock_model(
    client,
    markdown: str,
    reviews_text: Optional[str] = None,
) -> Optional[Literal["ACCEPT", "REJECT"]]:
    prompt = build_prompt(markdown, reviews_text=reviews_text)
    response = client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={
            "maxTokens": 4096,
            "temperature": 0,
            "stopSequences": [],
        },
    )

    response_content = response.get("output", {}).get("message", {}).get("content", [])
    text_parts = [seg["text"] for seg in response_content if "text" in seg]
    full_text = "".join(text_parts).strip()
    return parse_prediction(full_text)


def get_ground_truth_label(review_json: Dict[str, Any]) -> Optional[Literal["ACCEPT", "REJECT"]]:
    accepted_value = review_json.get("accepted")

    if isinstance(accepted_value, bool):
        return "ACCEPT" if accepted_value else "REJECT"
    if isinstance(accepted_value, (int, float)):
        return "ACCEPT" if bool(accepted_value) else "REJECT"
    if isinstance(accepted_value, str):
        val = accepted_value.strip().lower()
        if val in {"true", "yes", "y", "accept", "accepted", "1"}:
            return "ACCEPT"
        if val in {"false", "no", "n", "reject", "rejected", "0"}:
            return "REJECT"
    return None


def main() -> None:
    if not PAPERS_DIR.is_dir():
        raise SystemExit(f"Papers dir not found: {PAPERS_DIR}")
    if not REVIEWS_DIR.is_dir():
        raise SystemExit(f"Reviews dir not found: {REVIEWS_DIR}")

    random.seed(RANDOM_SEED)

    paper_files = sorted(PAPERS_DIR.glob("*.json"))
    if not paper_files:
        raise SystemExit(f"No paper .json files found in {PAPERS_DIR}")

    labeled = []
    for paper_path in paper_files:
        paper_id = paper_path.stem
        review_path = REVIEWS_DIR / f"{paper_id}.json"
        if not review_path.is_file():
            continue

        with paper_path.open("r", encoding="utf-8") as f:
            paper_json = json.load(f)
        with review_path.open("r", encoding="utf-8") as f:
            review_json = json.load(f)

        markdown = str(paper_json.get("markdown") or "").strip()
        if not markdown:
            continue

        label = get_ground_truth_label(review_json)
        if label is None:
            continue

        labeled.append(
            {
                "paper_id": paper_id,
                "paper_json": paper_json,
                "review_json": review_json,
                "label": label,
            }
        )

    if not labeled:
        print("No labeled markdown papers found.")
        return

    accepted = [s for s in labeled if s["label"] == "ACCEPT"]
    rejected = [s for s in labeled if s["label"] == "REJECT"]
    sampled: List[Dict[str, Any]] = []

    balanced = BALANCED
    if balanced:
        if not accepted or not rejected:
            print("Cannot perform balanced sampling; falling back to random sampling.")
            balanced = False
        else:
            per_class = NUM_SAMPLES // 2
            n_accept = min(per_class, len(accepted))
            n_reject = min(per_class, len(rejected))
            sampled = random.sample(accepted, n_accept) + random.sample(rejected, n_reject)

            remaining = NUM_SAMPLES - len(sampled)
            if remaining > 0:
                leftovers = [s for s in labeled if s not in sampled]
                if leftovers:
                    sampled.extend(random.sample(leftovers, min(remaining, len(leftovers))))
            random.shuffle(sampled)

    if not balanced:
        if len(labeled) <= NUM_SAMPLES:
            sampled = labeled
        else:
            sampled = random.sample(labeled, NUM_SAMPLES)

    client = get_bedrock_client()

    results = []
    correct = 0
    tp = fp = tn = fn = 0
    unparsed_predictions = 0

    for sample in sampled:
        paper_id = sample["paper_id"]
        ground_truth = sample["label"]
        markdown = str(sample["paper_json"].get("markdown") or "")

        reviews_text = None
        reviews = sample["review_json"].get("reviews")
        if HAS_REVIEWS and isinstance(reviews, list) and reviews:
            reviews_text = format_reviews(reviews)

        prediction = call_bedrock_model(client, markdown, reviews_text=reviews_text)

        if prediction is None:
            unparsed_predictions += 1
        elif ground_truth == "ACCEPT":
            if prediction == "ACCEPT":
                tp += 1
            else:
                fn += 1
        else:
            if prediction == "ACCEPT":
                fp += 1
            else:
                tn += 1

        is_correct = prediction == ground_truth
        if is_correct:
            correct += 1

        results.append(
            {
                "paper_id": paper_id,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "correct": is_correct,
            }
        )
        print(
            f"Paper {paper_id}: prediction={prediction}, "
            f"ground_truth={ground_truth}, correct={is_correct}"
        )

    total_evaluated = len(results)
    if total_evaluated == 0:
        print("No samples were evaluated.")
        return

    accuracy = correct / total_evaluated
    tpr = tp / (tp + fn) if (tp + fn) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None

    print("\n=== Summary ===")
    print(f"Evaluated samples: {total_evaluated}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Unparsed predictions: {unparsed_predictions}")
    print(f"Confusion matrix (positive=ACCEPT): TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(
        f"TPR (Recall for ACCEPT): {tpr:.2%}"
        if tpr is not None
        else "TPR (Recall for ACCEPT): N/A"
    )
    print(
        f"FPR (REJECT->ACCEPT rate): {fpr:.2%}"
        if fpr is not None
        else "FPR (REJECT->ACCEPT rate): N/A"
    )

    output_data = {
        "model": MODEL_ID,
        "conference": CONFERENCE,
        "dataset_dir": str(DATASET_DIR),
        "has_reviews": HAS_REVIEWS,
        "prompt_mode": PROMPT_MODE,
        "results": [
            {
                "paper_id": r["paper_id"],
                "prediction": r["prediction"],
                "ground_truth": r["ground_truth"],
            }
            for r in results
        ],
        "accuracy": accuracy,
        "tpr": tpr,
        "fpr": fpr,
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "positive_class": "ACCEPT",
        },
        "unparsed_predictions": unparsed_predictions,
        "total_evaluated": total_evaluated,
        "correct": correct,
    }
    print(output_data)


if __name__ == "__main__":
    main()
