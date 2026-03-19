#!/usr/bin/env python3
"""
Pairwise comparison eval: give the model one accepted and one rejected paper;
it must choose which is better. Each pair is run twice with order flipped to
control for positional bias. If the two runs disagree, the outcome is marked
inconsistent and counted as wrong for accuracy.

- Pools: accepted and rejected papers from output/neurips_2025_full/reviews.
- Sample NUM_PAIRS pairs (one accepted, one rejected) with fixed seed.
- For each pair: call model with (Paper1=accepted, Paper2=rejected), then
  (Paper1=rejected, Paper2=accepted). Parse choice 1 or 2 each time.
- Consistent iff both runs pick the same underlying paper (choice_1=1 and
  choice_2=2, or choice_1=2 and choice_2=1). Correct iff consistent and
  model chose the accepted paper (choice_1=1, choice_2=2).
- Accuracy = correct / NUM_PAIRS (inconsistent or wrong choice counts as wrong).

Context length: Pairs whose combined prompt exceeds MAX_TOKENS (estimated) are skipped
and replaced by resampling so that exactly NUM_PAIRS valid pairs are evaluated.
"""

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Literal, Optional

from botocore.exceptions import ClientError

from convert_json_to_markdown import json_to_markdown

from bedrock_neurips_acceptance_eval import (
    ANON_PDFS_DIR,
    DEFAULT_MODEL_ID,
    REVIEWS_DIR,
    format_reviews,
    get_bedrock_client,
    get_ground_truth_label,
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_JSON = BASE_DIR / "results" / "llama3.3_70b_instruct_pairwise_choice.json"

NUM_PAIRS = 100
RANDOM_SEED = 10718
# Maximum input context length (tokens). Pairs exceeding this are skipped and resampled.
# Rough estimate: ~4 chars per token. Tune for your model (e.g. 128000 for Llama 3).
MAX_TOKENS = 128000
# When True, append the reviews from each paper's review JSON to help the model decide.
HAS_REVIEWS = False

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars per token for English)."""
    return len(text) // 4


def build_pairwise_prompt(
    paper1_md: str,
    paper2_md: str,
    reviews1_text: Optional[str] = None,
    reviews2_text: Optional[str] = None,
) -> str:
    """Build prompt: Paper 1 and Paper 2; model must output choice 1 or 2."""
    task = (
        "You are an expert NeurIPS 2025 area chair.\n\n"
        "Task: You are given two anonymized NeurIPS 2025 submission papers. "
    )
    if reviews1_text or reviews2_text:
        task += (
            "You are also given the official reviews for each submission. "
            "Based on the paper content and the reviews, decide which paper is better "
            "in terms of scientific quality, clarity, and contribution.\n\n"
        )
    else:
        task += (
            "Based only on the paper content, decide which paper is better "
            "in terms of scientific quality, clarity, and contribution.\n\n"
        )

    task += (
        "Output format (important):\n"
        '- Respond with a single JSON object and NOTHING else.\n'
        '- The JSON must be exactly one of:\n'
        '  {\"choice\": 1}\n'
        '  {\"choice\": 2}\n'
        "meaning you choose Paper 1 or Paper 2 respectively.\n\n"
        "Paper 1:\n"
        "---------------- BEGIN PAPER 1 ----------------\n"
        f"{paper1_md}\n"
        "----------------- END PAPER 1 -----------------\n"
    )

    if reviews1_text:
        task += (
            "\nOfficial reviews for Paper 1:\n"
            "---------------- BEGIN REVIEWS 1 ----------------\n"
            f"{reviews1_text}\n"
            "----------------- END REVIEWS 1 -----------------\n"
        )

    task += (
        "\n\nPaper 2:\n"
        "---------------- BEGIN PAPER 2 ----------------\n"
        f"{paper2_md}\n"
        "----------------- END PAPER 2 -----------------\n"
    )

    if reviews2_text:
        task += (
            "\nOfficial reviews for Paper 2:\n"
            "---------------- BEGIN REVIEWS 2 ----------------\n"
            f"{reviews2_text}\n"
            "----------------- END REVIEWS 2 -----------------\n"
        )

    return task


def call_pairwise_choice(
    client,
    paper1_md: str,
    paper2_md: str,
    reviews1_text: Optional[str] = None,
    reviews2_text: Optional[str] = None,
) -> Optional[Literal[1, 2]]:
    """Call Bedrock; return 1 or 2 for which paper was chosen, or None if unparseable."""
    prompt = build_pairwise_prompt(paper1_md, paper2_md, reviews1_text, reviews2_text)
    kwargs = {
        "modelId": MODEL_ID,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {
            "maxTokens": 256,
            "temperature": 0,
            "stopSequences": [],
        },
    }
    throttle_codes = ("ThrottlingException", "TooManyRequestsException", "ServiceQuotaExceededException")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.converse(**kwargs)
            break
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in throttle_codes and attempt < max_retries - 1:
                print(f"Rate limit / throttling ({code}); waiting 30s then retrying (attempt {attempt + 1}/{max_retries})...")
                time.sleep(30)
                continue
            print(f"Error calling Bedrock: {e}")
            return None
        except Exception as e:
            print(f"Error calling Bedrock: {e}")
            return None
    content = response.get("output", {}).get("message", {}).get("content", [])
    full_text = "".join(s.get("text", "") for s in content).strip()

    # Parse {"choice": 1} or {"choice": 2}
    match = re.search(r'\{\s*"choice"\s*:\s*([12])\s*\}', full_text)
    if match:
        return int(match.group(1))  # type: ignore[return-value]
    # Fallback: first occurrence of "choice": 1 or 2
    if '"choice": 1' in full_text or '"choice":1' in full_text:
        return 1
    if '"choice": 2' in full_text or '"choice":2' in full_text:
        return 2
    return None


def main():
    if not ANON_PDFS_DIR.is_dir():
        raise SystemExit(f"Anonymized PDFs dir not found: {ANON_PDFS_DIR}")
    if not REVIEWS_DIR.is_dir():
        raise SystemExit(f"Reviews dir not found: {REVIEWS_DIR}")

    random.seed(RANDOM_SEED)

    json_files = sorted(ANON_PDFS_DIR.glob("*.pdf.json"))
    if not json_files:
        raise SystemExit(f"No .pdf.json files found in {ANON_PDFS_DIR}")

    # Build accepted and rejected pools (paper_id, path).
    accepted_pool = []
    rejected_pool = []
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
        entry = {"paper_id": paper_id, "path": path, "review_json": review_json}
        if label == "ACCEPT":
            accepted_pool.append(entry)
        else:
            rejected_pool.append(entry)

    if not accepted_pool or not rejected_pool:
        raise SystemExit(
            "Need both accepted and rejected papers. "
            f"Accepted: {len(accepted_pool)}, Rejected: {len(rejected_pool)}"
        )

    # Sample NUM_PAIRS pairs that fit within MAX_TOKENS. Resample if a pair's prompt is too long.
    pairs = []
    max_attempts = NUM_PAIRS * 50  # avoid infinite loop
    attempts = 0
    while len(pairs) < NUM_PAIRS and attempts < max_attempts:
        attempts += 1
        acc = random.choice(accepted_pool)
        rej = random.choice(rejected_pool)
        with acc["path"].open("r", encoding="utf-8") as f:
            acc_md = json_to_markdown(json.load(f))
        with rej["path"].open("r", encoding="utf-8") as f:
            rej_md = json_to_markdown(json.load(f))

        acc_reviews_text = None
        rej_reviews_text = None
        if HAS_REVIEWS:
            if acc.get("review_json") and acc["review_json"].get("reviews"):
                acc_reviews_text = format_reviews(acc["review_json"]["reviews"])
            if rej.get("review_json") and rej["review_json"].get("reviews"):
                rej_reviews_text = format_reviews(rej["review_json"]["reviews"])

        prompt = build_pairwise_prompt(acc_md, rej_md, acc_reviews_text, rej_reviews_text)
        if estimate_tokens(prompt) <= MAX_TOKENS:
            # keep markdown and reviews to avoid recomputing
            pairs.append((acc, rej, acc_md, rej_md, acc_reviews_text, rej_reviews_text))
        else:
            continue
    if len(pairs) < NUM_PAIRS:
        raise SystemExit(
            f"Only {len(pairs)} pairs fit within MAX_TOKENS={MAX_TOKENS} after {max_attempts} attempts. "
            "Increase MAX_TOKENS or add more papers."
        )

    client = get_bedrock_client()
    results = []
    correct = 0

    for i, (acc, rej, acc_md, rej_md, acc_reviews_text, rej_reviews_text) in enumerate(pairs):
        # Run 1: Paper1 = accepted, Paper2 = rejected
        choice_1 = call_pairwise_choice(
            client, acc_md, rej_md, acc_reviews_text, rej_reviews_text
        )
        # Run 2: Paper1 = rejected, Paper2 = accepted (flipped)
        choice_2 = call_pairwise_choice(
            client, rej_md, acc_md, rej_reviews_text, acc_reviews_text
        )

        # Consistent: both runs pick same underlying paper.
        # Run 1: 1=accepted, 2=rejected. Run 2: 1=rejected, 2=accepted.
        # So consistent <=> (choice_1, choice_2) in {(1,2), (2,1)}.
        # Correct <=> consistent and chosen paper is accepted <=> (choice_1, choice_2) == (1, 2).
        consistent = (
            choice_1 is not None
            and choice_2 is not None
            and ((choice_1 == 1 and choice_2 == 2) or (choice_1 == 2 and choice_2 == 1))
        )
        is_correct = consistent and choice_1 == 1 and choice_2 == 2
        if is_correct:
            correct += 1

        results.append({
            "accepted_id": acc["paper_id"],
            "rejected_id": rej["paper_id"],
            "choice_1": choice_1,
            "choice_2": choice_2,
            "consistent": consistent,
            "correct": is_correct,
        })
        print(
            f"Pair {i+1}/{NUM_PAIRS}: accepted={acc['paper_id']}, rejected={rej['paper_id']} | "
            f"choices=({choice_1}, {choice_2}) consistent={consistent} correct={is_correct}"
        )

    accuracy = correct / NUM_PAIRS
    num_consistent = sum(1 for r in results if r["consistent"])

    print("\n=== Summary ===")
    print(f"Pairs evaluated: {NUM_PAIRS}")
    print(f"Consistent (same choice when order flipped): {num_consistent}")
    print(f"Correct (consistent and chose accepted paper): {correct}")
    print(f"Accuracy: {accuracy:.2%}")

    output_data = {
        "model": MODEL_ID,
        "seed": RANDOM_SEED,
        "max_tokens": MAX_TOKENS,
        "has_reviews": HAS_REVIEWS,
        "num_pairs": NUM_PAIRS,
        "results": results,
        "accuracy": accuracy,
        "num_consistent": num_consistent,
        "correct": correct,
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
