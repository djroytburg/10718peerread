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


BASE_DIR = Path(__file__).resolve().parent
ANON_PDFS_DIR = BASE_DIR / "output" / "neurips_2025_full" / "anonymized_pdfs"
REVIEWS_DIR = BASE_DIR / "output" / "neurips_2025_full" / "reviews"
RESULTS_JSON = BASE_DIR / "results" / "llama3.3_70b_instruct_balanced_reviews_neurips_2025_full.json"

NUM_SAMPLES = 100
RANDOM_SEED = 10718
BALANCED = True
# When True, append the "reviews" from each paper's review JSON to the model prompt.
HAS_REVIEWS = True

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
        "You are an expert NeurIPS 2025 area chair.\n\n"
        "Task:\n"
        "- Read the following anonymized NeurIPS 2025 paper in Markdown form.\n"
    )
    if reviews_text:
        task += (
            "- You are also given the official reviews for this submission.\n"
            "- Based on the paper content and the reviews, PREDICT whether it "
            "was accepted to NeurIPS 2025.\n"
        )
    else:
        task += (
            "- Based ONLY on the content and quality of the paper, PREDICT whether it "
            "was accepted to NeurIPS 2025.\n"
        )
    task += (
        "- You do not know the true decision; you must guess.\n\n"
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

    # Try to extract {"prediction": "ACCEPT"} or {"prediction": "REJECT"} via regex.
    match = re.search(
        r'\{\s*"prediction"\s*:\s*"(ACCEPT|REJECT)"\s*\}',
        full_text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()  # type: ignore[return-value]

    # Fallback: look for the words ACCEPT / REJECT if the JSON contract was not obeyed.
    upper = full_text.upper()
    if "ACCEPT" in upper and "REJECT" not in upper:
        return "ACCEPT"
    if "REJECT" in upper and "ACCEPT" not in upper:
        return "REJECT"

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
    print("\n=== Summary ===")
    print(f"Evaluated samples: {total_evaluated}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

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
        "total_evaluated": total_evaluated,
        "correct": correct,
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {RESULTS_JSON}")


if __name__ == "__main__":
    main()

