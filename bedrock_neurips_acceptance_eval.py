#!/usr/bin/env python3
"""
Run a small Amazon Bedrock experiment:

- Sample 100 anonymized NeurIPS 2025 papers from
  `output/neurips_2025_full/anonymized_pdfs`.
- Convert each JSON to markdown using `json_to_markdown`.
- Ask a Bedrock model (default: Llama 3.3 70B Instruct) to predict whether the paper was accepted.
- Look up the ground-truth `accepted` label from the corresponding file in
  `output/neurips_2025_full/reviews`.
- Report per-sample predictions and overall accuracy.

Authentication:
- This script uses the standard AWS Bedrock Runtime client via `boto3`.
- Configure your AWS credentials and region in the usual AWS ways
  (environment variables, shared config/credentials files, or IAM role).
- Optionally override the model ID and region via environment variables:
    BEDROCK_MODEL_ID
    AWS_REGION
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Any, List, Literal, Optional

import boto3

from convert_json_to_markdown import json_to_markdown


# BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent.parent
# ANON_PDFS_DIR = BASE_DIR / "output" / "neurips_2025_full" / "anonymized_pdfs"
# REVIEWS_DIR = BASE_DIR / "output" / "neurips_2025_full" / "reviews"
ANON_PDFS_DIR = BASE_DIR / "PeerRead" / "data" / "iclr_2017" / "test" / "parsed_pdfs"
REVIEWS_DIR = BASE_DIR / "PeerRead" / "data" / "iclr_2017" / "test" / "reviews"
# RESULTS_JSON = BASE_DIR / "results" / "llama3.3_70b_instruct_balanced_reviews_iclr_2017.json"

NUM_SAMPLES = 100
RANDOM_SEED = 10718
CONTEXT_LEN = 100000
BALANCED = True
# When True, append the "reviews" from each paper's review JSON to the model prompt.
HAS_REVIEWS = False
# Prompt mode options: "neutral", "conservative", "severe_conservative".
# You can also override via environment variable PROMPT_MODE.
PROMPT_MODE = "neutral"
CONFERENCE = "ICLR 2017"

# Use the foundation model ID (no ARN) so Bedrock runs in  YOUR account.
# Request access to this model in AWS Console > Bedrock > Model access if needed.
# Override with BEDROCK_MODEL_ID in your environment if desired.
DEFAULT_MODEL_ID = "us.meta.llama3-3-70b-instruct-v1:0"  # Llama 3.3 70B Instruct
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def format_reviews(reviews: List[Any]) -> str:
    """Format the 'reviews' array from a review JSON into a single string."""
    parts = []
    for i, r in enumerate(reviews, 1):
        title = r.get("TITLE", "Review")
        comments = r.get("comments", "")
        meta = "(meta-review)" if r.get("IS_META_REVIEW") else ""
        parts.append(f"--- Review {i}: {title} {meta} ---")
        parts.append(comments)
        parts.append("")
    return "\n".join(parts).strip()


def build_prompt(markdown: str, reviews_text: Optional[str] = None) -> str:
    """Build the prompt sent to the model."""
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
    policy_mode = PROMPT_MODE if PROMPT_MODE in {
        "neutral",
        "conservative",
        "severe_conservative",
    } else "neutral"

    if policy_mode == "neutral":
        policy_text = (
            "Decision policy (important):\n"
            "- Make your best single-shot guess from the available evidence.\n"
            "- If strengths and weaknesses are close, choose the more likely outcome.\n\n"
        )
    elif policy_mode == "conservative":
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

    task += (
        "- You do not know the true decision; you must guess.\n\n"
        f"{policy_text}"
        "Output format (important):\n"
        '- Respond with a single JSON object and NOTHING else.\n'
        '- The JSON must be exactly one of:\n'
        '  {\"prediction\": \"ACCEPT\"}\n'
        '  {\"prediction\": \"REJECT\"}\n\n'
        "Paper Markdown:\n"
        "---------------- BEGIN PAPER ----------------\n"
        f"{markdown}\n"
        "----------------- END PAPER -----------------\n"
    )
    if reviews_text:
        task += (
            "\n\nOfficial Reviews:\n"
            "---------------- BEGIN REVIEWS ----------------\n"
            f"{reviews_text}\n"
            "----------------- END REVIEWS -----------------\n"
        )
    return task


def call_bedrock_model(
    client,
    markdown: str,
    reviews_text: Optional[str] = None,
) -> Optional[Literal["ACCEPT", "REJECT"]]:
    """Call Bedrock Converse API and parse a binary ACCEPT/REJECT prediction."""
    prompt = build_prompt(markdown, reviews_text=reviews_text)
    kwargs = {
        "modelId": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ],
        "inferenceConfig": {
            "maxTokens": 4096,
            "temperature": 0,
            "stopSequences": [],
        },
    }
    response = client.converse(**kwargs)

    # Converse API returns the output message content here.
    response_content = response.get("output", {}).get("message", {}).get("content", [])

    text_parts = []
    for segment in response_content:
        if "text" in segment:
            text_parts.append(segment["text"])
    full_text = "".join(text_parts).strip()
    print(full_text)
    # Parse as raw JSON first (best case: model obeyed output contract exactly).
    try:
        parsed = json.loads(full_text)
        if isinstance(parsed, dict):
            pred = parsed.get("prediction")
            if isinstance(pred, str) and pred.strip().upper() in {"ACCEPT", "REJECT"}:
                return pred.strip().upper()  # type: ignore[return-value]
    except json.JSONDecodeError:
        pass

    # Robust extraction: if the model echoed instructions/examples, there can be
    # multiple JSON snippets in the output. Use the last prediction mention.
    matches = re.findall(
        r'\{\s*"prediction"\s*:\s*"(ACCEPT|REJECT)"\s*\}',
        full_text,
        re.IGNORECASE,
    )
    if matches:
        return matches[-1].upper()  # type: ignore[return-value]

    # Last-resort fallback: choose whichever label appears last in the text.
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


def get_ground_truth_label(review_json: dict) -> Optional[Literal["ACCEPT", "REJECT"]]:
    """
    Map the `accepted` attribute in a review JSON to ACCEPT / REJECT.

    The file at `output/neurips_2025_full/reviews/<id>.json` includes
    an `accepted` attribute.
    """
    accepted_value = review_json.get("accepted", None)

    if isinstance(accepted_value, bool):
        return "ACCEPT" if accepted_value else "REJECT"

    if isinstance(accepted_value, (int, float)):
        return "ACCEPT" if bool(accepted_value) else "REJECT"

    if isinstance(accepted_value, str):
        val = accepted_value.strip().lower()
        if val in {"true", "yes", "y", "accept", "accepted"}:
            return "ACCEPT"
        if val in {"false", "no", "n", "reject", "rejected"}:
            return "REJECT"

    return None


def main():
    if not ANON_PDFS_DIR.is_dir():
        raise SystemExit(f"Anonymized PDFs dir not found: {ANON_PDFS_DIR}")
    if not REVIEWS_DIR.is_dir():
        raise SystemExit(f"Reviews dir not found: {REVIEWS_DIR}")

    num_samples = NUM_SAMPLES
    balanced = BALANCED

    random.seed(RANDOM_SEED)

    json_files = sorted(ANON_PDFS_DIR.glob("*.pdf.json"))
    if len(json_files) == 0:
        raise SystemExit(f"No .pdf.json files found in {ANON_PDFS_DIR}")

    # Build a labeled pool of papers with valid reviews.
    labeled = []
    for path in json_files:
        paper_id = path.name.replace(".pdf.json", "")
        review_path = REVIEWS_DIR / f"{paper_id}.json"
        if not review_path.is_file():
            continue

        with review_path.open("r", encoding="utf-8") as f:
            review_json = json.load(f)

        label = get_ground_truth_label(review_json)
        if label is None:
            continue

        labeled.append({
            "paper_id": paper_id,
            "path": path,
            "label": label,
            "review_json": review_json,
        })

    if not labeled:
        print("No labeled papers with valid reviews were found.")
        return

    accepted = [s for s in labeled if s["label"] == "ACCEPT"]
    rejected = [s for s in labeled if s["label"] == "REJECT"]
    sampled = []

    if balanced:
        if not accepted or not rejected:
            print(
                "Cannot perform balanced sampling (need both accepted and rejected papers). "
                "Falling back to unbalanced random sampling."
            )
            balanced = False
        else:
            per_class = num_samples // 2
            n_accept = min(per_class, len(accepted))
            n_reject = min(per_class, len(rejected))

            if n_accept == 0 or n_reject == 0:
                print(
                    "Cannot perform balanced sampling with requested num-samples; "
                    "falling back to unbalanced random sampling."
                )
                balanced = False
            else:
                sampled = random.sample(accepted, n_accept) + random.sample(
                    rejected, n_reject
                )

                # If num_samples is odd or we had to clip one side, optionally top up
                # with remaining papers (still deterministic given the seed).
                remaining = num_samples - len(sampled)
                if remaining > 0:
                    leftover_pool = [s for s in labeled if s not in sampled]
                    if leftover_pool:
                        sampled.extend(
                            random.sample(
                                leftover_pool, min(remaining, len(leftover_pool))
                            )
                        )

                random.shuffle(sampled)

    if not balanced:
        pool = labeled
        if len(pool) <= num_samples:
            if len(pool) < num_samples:
                print(
                    f"Warning: only found {len(pool)} labeled papers, "
                    f"but requested num_samples={num_samples}. Using all available."
                )
            sampled = pool
        else:
            sampled = random.sample(pool, num_samples)

    bedrock_client = get_bedrock_client()

    results = []
    correct = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    unparsed_predictions = 0

    for sample in sampled:
        paper_id = sample["paper_id"]
        path = sample["path"]
        ground_truth = sample["label"]

        with path.open("r", encoding="utf-8") as f:
            doc = json.load(f)

        markdown = json_to_markdown(doc)

        reviews_text = None
        if HAS_REVIEWS and sample.get("review_json") and sample["review_json"].get("reviews"):
            reviews_text = format_reviews(sample["review_json"]["reviews"])

        prediction = call_bedrock_model(
            bedrock_client, markdown, reviews_text=reviews_text
        )

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
        print("No samples were evaluated (all were skipped).")
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
    print(f"TPR (Recall for ACCEPT): {tpr:.2%}" if tpr is not None else "TPR (Recall for ACCEPT): N/A")
    print(f"FPR (REJECT->ACCEPT rate): {fpr:.2%}" if fpr is not None else "FPR (REJECT->ACCEPT rate): N/A")

    output_data = {
        "model": MODEL_ID,
        "has_reviews": HAS_REVIEWS,
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
    # RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    # with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    #     json.dump(output_data, f, indent=2)
    # print(f"\nResults saved to: {RESULTS_JSON}")


if __name__ == "__main__":
    main()

