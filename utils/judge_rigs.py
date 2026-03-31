"""
Judge rigs: prompt/format scaffolds for paper-acceptance evaluation.

Each rig exposes:
  - build_prompt(...)         -> str
  - parse(text)               -> prediction
  - to_config_dict()          -> dict  (used for cache keying)
  - __call__(client, ...,
             estimate_only=False,
             cache_config=None) -> (prediction, ..., Usage)

When estimate_only=True no API call is made; returned Usage contains estimated
input tokens and projected cost with zero output tokens.

When cache_config={"dataset": str, "paper_id": str} is provided, results are
looked up from and saved to results/cache/ automatically.
"""

import dataclasses
import itertools
import json
import random
import re
from pathlib import Path
from typing import Any, Literal, Optional

from utils.bedrock import (
    MODEL_ID,
    Usage,
    converse_text,
    estimate_tokens,
)
from utils.cache import (
    build_config_hash,
    load_cached,
    save_result,
    update_manifest,
)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AcceptReject = Literal["ACCEPT", "REJECT"]
PairwiseChoice = Literal[1, 2]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def format_reviews(reviews: list[Any]) -> str:
    """Format a reviews array from a review JSON into a single string."""
    parts = []
    for i, r in enumerate(reviews, 1):
        title = r.get("TITLE", "Review")
        comments = r.get("comments", "")
        meta = "(meta-review)" if r.get("IS_META_REVIEW") else ""
        parts.append(f"--- Review {i}: {title} {meta} ---")
        parts.append(comments)
        parts.append("")
    return "\n".join(parts).strip()


def _dry_run_usage(prompt: str) -> Usage:
    """Return a Usage with estimated input tokens and zero output tokens."""
    return Usage(input_tokens=estimate_tokens(prompt), output_tokens=0, latency_ms=0, calls=0)


def make_few_shot_examples(
    dataset_dir: str | Path,
    n: int = 4,
    balanced: bool = True,
    seed: int = 42,
) -> list[dict]:
    """
    Load up to n labeled examples from dataset_dir for use as few-shot demonstrations.

    dataset_dir must contain:
      - reviews/           (*.json files with 'accepted' and 'reviews' fields)
      - anonymized_pdfs/   (*.pdf.json files for json_to_markdown)

    Returns list of dicts:
      {"paper_md": str, "reviews_text": str, "label": "ACCEPT"|"REJECT"}

    If balanced=True, samples n//2 from each class; remainder filled from
    the larger class. Pass balanced=False to sample randomly regardless of label.
    """
    try:
        from convert_json_to_markdown import json_to_markdown
    except ImportError as e:
        raise ImportError(
            "convert_json_to_markdown not found. Run from the project root directory."
        ) from e

    dataset_dir = Path(dataset_dir)
    reviews_dir = dataset_dir / "reviews"
    pdfs_dir = dataset_dir / "anonymized_pdfs"

    rng = random.Random(seed)

    # Load all labeled papers
    accepted, rejected = [], []
    for review_path in sorted(reviews_dir.glob("*.json")):
        paper_id = review_path.stem
        pdf_path = pdfs_dir / f"{paper_id}.pdf.json"
        if not pdf_path.is_file():
            continue
        review_json = json.loads(review_path.read_text(encoding="utf-8"))
        accepted_val = review_json.get("accepted")
        if accepted_val is None:
            continue
        label: AcceptReject = "ACCEPT" if accepted_val else "REJECT"
        entry = {"paper_id": paper_id, "pdf_path": pdf_path,
                 "review_json": review_json, "label": label}
        (accepted if label == "ACCEPT" else rejected).append(entry)

    # Sample
    if balanced and accepted and rejected:
        per_class = n // 2
        n_acc = min(per_class, len(accepted))
        n_rej = min(per_class, len(rejected))
        sampled = rng.sample(accepted, n_acc) + rng.sample(rejected, n_rej)
        remaining = n - len(sampled)
        if remaining > 0:
            leftover = [e for e in (accepted + rejected) if e not in sampled]
            sampled += rng.sample(leftover, min(remaining, len(leftover)))
    else:
        pool = accepted + rejected
        sampled = rng.sample(pool, min(n, len(pool)))

    rng.shuffle(sampled)

    examples = []
    for entry in sampled:
        doc = json.loads(entry["pdf_path"].read_text(encoding="utf-8"))
        paper_md = json_to_markdown(doc)
        reviews = entry["review_json"].get("reviews", [])
        reviews_text = format_reviews(reviews) if reviews else None
        examples.append({
            "paper_md": paper_md,
            "reviews_text": reviews_text,
            "label": entry["label"],
        })
    return examples


# ---------------------------------------------------------------------------
# Rig 1: Binary acceptance prediction (single paper)
# ---------------------------------------------------------------------------

class AcceptanceRig:
    """
    Prompt the model to predict ACCEPT or REJECT for a single paper.

    Parameters
    ----------
    with_reviews : bool
        If True, reviews text is appended to the prompt when provided.
    conference : str
        Conference name used in the system prompt.
    max_output_tokens : int
        Token budget for the model response.
    few_shot_examples : list[dict], optional
        Each dict: {"paper_md": str, "reviews_text": Optional[str], "label": "ACCEPT"|"REJECT"}.
        Injected before the actual paper as demonstrations.
    """

    def __init__(
        self,
        with_reviews: bool = True,
        conference: str = "NeurIPS 2025",
        max_output_tokens: int = 64,
        few_shot_examples: Optional[list[dict]] = None,
    ):
        self.with_reviews = with_reviews
        self.conference = conference
        self.max_output_tokens = max_output_tokens
        self.few_shot_examples = few_shot_examples or []

    def to_config_dict(self) -> dict:
        return {
            "rig_type": "AcceptanceRig",
            "with_reviews": self.with_reviews,
            "conference": self.conference,
            "max_output_tokens": self.max_output_tokens,
            "n_few_shot": len(self.few_shot_examples),
        }

    def _paper_block(self, paper_md: str, reviews_text: Optional[str] = None) -> str:
        block = (
            "---------------- BEGIN PAPER ----------------\n"
            f"{paper_md}\n"
            "----------------- END PAPER -----------------\n"
        )
        if self.with_reviews and reviews_text:
            block += (
                "\nOfficial Reviews:\n"
                "---------------- BEGIN REVIEWS ----------------\n"
                f"{reviews_text}\n"
                "----------------- END REVIEWS -----------------\n"
            )
        return block

    def build_prompt(
        self,
        paper_md: str,
        reviews_text: Optional[str] = None,
    ) -> str:
        conf = self.conference
        task = (
            f"You are an expert {conf} area chair.\n\n"
            "Task:\n"
            f"- Read the following anonymized {conf} paper in Markdown form.\n"
        )
        if self.with_reviews and reviews_text:
            task += (
                "- You are also given the official reviews for this submission.\n"
                f"- Based on the paper content and the reviews, PREDICT whether it "
                f"was accepted to {conf}.\n"
            )
        else:
            task += (
                f"- Based ONLY on the content and quality of the paper, PREDICT whether it "
                f"was accepted to {conf}.\n"
            )
        task += (
            "- You do not know the true decision; you must guess.\n\n"
            "Output format (important):\n"
            "- Respond with a single JSON object and NOTHING else.\n"
            '- The JSON must be exactly one of:\n'
            '  {"prediction": "ACCEPT"}\n'
            '  {"prediction": "REJECT"}\n\n'
        )

        if self.few_shot_examples:
            task += "Here are some example evaluations to guide your response:\n\n"
            for i, ex in enumerate(self.few_shot_examples, 1):
                task += f"=== EXAMPLE {i} ===\n"
                task += "Paper Markdown:\n"
                task += self._paper_block(ex["paper_md"], ex.get("reviews_text"))
                task += f'\nCorrect prediction: {{"prediction": "{ex["label"]}"}}\n\n'
            task += "=== ACTUAL PAPER TO EVALUATE ===\n"

        task += "Paper Markdown:\n"
        task += self._paper_block(paper_md, reviews_text)
        return task

    def parse(self, text: str) -> Optional[AcceptReject]:
        match = re.search(
            r'\{\s*"prediction"\s*:\s*"(ACCEPT|REJECT)"\s*\}',
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()  # type: ignore[return-value]
        upper = text.upper()
        if "ACCEPT" in upper and "REJECT" not in upper:
            return "ACCEPT"
        if "REJECT" in upper and "ACCEPT" not in upper:
            return "REJECT"
        return None

    def __call__(
        self,
        client,
        paper_md: str,
        reviews_text: Optional[str] = None,
        model_id: str = MODEL_ID,
        estimate_only: bool = False,
        cache_config: Optional[dict] = None,
    ) -> tuple[Optional[AcceptReject], Usage]:
        prompt = self.build_prompt(paper_md, reviews_text)

        if estimate_only:
            return None, _dry_run_usage(prompt)

        if cache_config:
            config_hash = build_config_hash(self.to_config_dict())
            cached = load_cached(cache_config["paper_id"], config_hash, cache_config["dataset"])
            if cached:
                return cached["prediction"], Usage(**cached["usage"])

        text, usage = converse_text(client, prompt, model_id=model_id, max_tokens=self.max_output_tokens)
        prediction = self.parse(text) if text is not None else None

        if cache_config:
            save_result(cache_config["paper_id"], config_hash, cache_config["dataset"], {
                "prediction": prediction,
                "usage": dataclasses.asdict(usage),
            })
            update_manifest(
                config_hash, "AcceptanceRig",
                f"with_reviews={self.with_reviews} model={model_id}",
                cache_config["dataset"],
            )

        return prediction, usage


# ---------------------------------------------------------------------------
# Rig 2: Pairwise choice (two papers, pick the better one)
# ---------------------------------------------------------------------------

class PairwiseRig:
    """
    Show the model two papers and ask which is better.

    Each pair is evaluated twice with order flipped to control for positional
    bias. A result is consistent iff both orderings pick the same underlying
    paper; correct iff consistent and the accepted paper was chosen.

    Parameters
    ----------
    with_reviews : bool
        If True, per-paper reviews are appended when provided.
    conference : str
        Conference name used in the system prompt.
    max_output_tokens : int
        Token budget for each model call (two calls per pair).
    few_shot_examples : list[dict], optional
        Each dict: {"paper1_md", "paper2_md", "reviews1_text"?, "reviews2_text"?, "label": 1|2}.
    """

    def __init__(
        self,
        with_reviews: bool = False,
        conference: str = "NeurIPS 2025",
        max_output_tokens: int = 64,
        few_shot_examples: Optional[list[dict]] = None,
    ):
        self.with_reviews = with_reviews
        self.conference = conference
        self.max_output_tokens = max_output_tokens
        self.few_shot_examples = few_shot_examples or []

    def to_config_dict(self) -> dict:
        return {
            "rig_type": "PairwiseRig",
            "with_reviews": self.with_reviews,
            "conference": self.conference,
            "max_output_tokens": self.max_output_tokens,
            "n_few_shot": len(self.few_shot_examples),
        }

    def _papers_block(
        self,
        paper1_md: str,
        paper2_md: str,
        reviews1_text: Optional[str] = None,
        reviews2_text: Optional[str] = None,
    ) -> str:
        block = (
            "Paper 1:\n"
            "---------------- BEGIN PAPER 1 ----------------\n"
            f"{paper1_md}\n"
            "----------------- END PAPER 1 -----------------\n"
        )
        if self.with_reviews and reviews1_text:
            block += (
                "\nOfficial reviews for Paper 1:\n"
                "---------------- BEGIN REVIEWS 1 ----------------\n"
                f"{reviews1_text}\n"
                "----------------- END REVIEWS 1 -----------------\n"
            )
        block += (
            "\n\nPaper 2:\n"
            "---------------- BEGIN PAPER 2 ----------------\n"
            f"{paper2_md}\n"
            "----------------- END PAPER 2 -----------------\n"
        )
        if self.with_reviews and reviews2_text:
            block += (
                "\nOfficial reviews for Paper 2:\n"
                "---------------- BEGIN REVIEWS 2 ----------------\n"
                f"{reviews2_text}\n"
                "----------------- END REVIEWS 2 -----------------\n"
            )
        return block

    def build_prompt(
        self,
        paper1_md: str,
        paper2_md: str,
        reviews1_text: Optional[str] = None,
        reviews2_text: Optional[str] = None,
    ) -> str:
        conf = self.conference
        task = f"You are an expert {conf} area chair.\n\n"
        task += f"Task: You are given two anonymized {conf} submission papers. "
        if self.with_reviews and (reviews1_text or reviews2_text):
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
            "- Respond with a single JSON object and NOTHING else.\n"
            "- The JSON must be exactly one of:\n"
            '  {"choice": 1}\n'
            '  {"choice": 2}\n'
            "meaning you choose Paper 1 or Paper 2 respectively.\n\n"
        )

        if self.few_shot_examples:
            task += "Here are some example evaluations to guide your response:\n\n"
            for i, ex in enumerate(self.few_shot_examples, 1):
                task += f"=== EXAMPLE {i} ===\n"
                task += self._papers_block(
                    ex["paper1_md"], ex["paper2_md"],
                    ex.get("reviews1_text"), ex.get("reviews2_text"),
                )
                task += f'\nCorrect choice: {{"choice": {ex["label"]}}}\n\n'
            task += "=== ACTUAL PAPERS TO EVALUATE ===\n"

        task += self._papers_block(paper1_md, paper2_md, reviews1_text, reviews2_text)
        return task

    def parse(self, text: str) -> Optional[PairwiseChoice]:
        match = re.search(r'\{\s*"choice"\s*:\s*([12])\s*\}', text)
        if match:
            return int(match.group(1))  # type: ignore[return-value]
        if '"choice": 1' in text or '"choice":1' in text:
            return 1
        if '"choice": 2' in text or '"choice":2' in text:
            return 2
        return None

    def __call__(
        self,
        client,
        paper1_md: str,
        paper2_md: str,
        reviews1_text: Optional[str] = None,
        reviews2_text: Optional[str] = None,
        model_id: str = MODEL_ID,
        estimate_only: bool = False,
        cache_config: Optional[dict] = None,
    ) -> tuple[Optional[PairwiseChoice], Optional[PairwiseChoice], Usage]:
        """
        Run both orderings. Returns (choice_order1, choice_order2, combined Usage).
        In estimate_only mode only the forward prompt is costed (×2 for both passes).
        """
        prompt_fwd = self.build_prompt(paper1_md, paper2_md, reviews1_text, reviews2_text)

        if estimate_only:
            single = _dry_run_usage(prompt_fwd)
            return None, None, Usage(input_tokens=single.input_tokens * 2, output_tokens=0, latency_ms=0, calls=0)

        if cache_config:
            config_hash = build_config_hash(self.to_config_dict())
            cached = load_cached(cache_config["paper_id"], config_hash, cache_config["dataset"])
            if cached:
                return cached["choice1"], cached["choice2"], Usage(**cached["usage"])

        prompt_rev = self.build_prompt(paper2_md, paper1_md, reviews2_text, reviews1_text)
        text1, usage1 = converse_text(client, prompt_fwd, model_id=model_id, max_tokens=self.max_output_tokens)
        text2, usage2 = converse_text(client, prompt_rev, model_id=model_id, max_tokens=self.max_output_tokens)
        choice1 = self.parse(text1) if text1 is not None else None
        choice2 = self.parse(text2) if text2 is not None else None
        combined = usage1 + usage2

        if cache_config:
            save_result(cache_config["paper_id"], config_hash, cache_config["dataset"], {
                "choice1": choice1,
                "choice2": choice2,
                "usage": dataclasses.asdict(combined),
            })
            update_manifest(
                config_hash, "PairwiseRig",
                f"with_reviews={self.with_reviews} model={model_id}",
                cache_config["dataset"],
            )

        return choice1, choice2, combined


# ---------------------------------------------------------------------------
# Rig 3: Debate-as-reviewers
# ---------------------------------------------------------------------------

class DebateRig:
    """
    Simulate a multi-round peer review debate, then have an Area Chair decide.

    Default mode (no real reviews):
      - n_reviewers reviewer agents debate the paper for n_rounds rounds.
      - Each reviewer is assigned a persona ("neutral" or "critical") sampled
        uniformly at __init__ time using persona_seed.
      - Reviewer turn order is re-shuffled each round.

    Real-review mode (reviews provided):
      - Each reviewer is assigned reviews[i % len(reviews)] as their seeding
        review, which is injected into their round-1 prompt only.
      - Subsequent rounds use the full transcript as context.

    The Area Chair (AC) sees paper and/or transcript per ac_sees, then outputs:
      {"prediction": "ACCEPT"|"REJECT", "confidence": 0.0–1.0}

    Parameters
    ----------
    n_reviewers : int
        Number of reviewer agents (ignored if reviews is provided).
    reviewer_models : list[str], optional
        Model ID per reviewer slot; cycled if shorter than n_reviewers.
        Defaults to [MODEL_ID] * n_reviewers.
    ac_model : str
        Model ID for the Area Chair.
    n_rounds : int
        Number of debate rounds before the AC decides.
    ac_sees : "paper" | "transcript" | "both"
        What context the AC receives.
    conference : str
        Conference name used in all prompts.
    max_output_tokens : int
        Token budget per reviewer turn.
    ac_max_output_tokens : int
        Token budget for the AC response.
    reviews : list[dict], optional
        Real review objects (from the reviews JSON). Seeds reviewer round-1 prompts.
    persona_seed : int
        Seed for deterministic persona sampling. Included in config dict for
        stable cache keys.
    cache_config : dict, optional
        {"dataset": str, "paper_id": str} — enables auto cache lookup/save.
    """

    PERSONAS = {
        "neutral": (
            "You are a balanced, thorough NeurIPS reviewer. You follow the NeurIPS reviewer "
            "guidelines carefully, weighing both strengths and weaknesses with equal attention. "
            "You are fair and constructive."
        ),
        "critical": (
            "You are a rigorous, skeptical NeurIPS reviewer. You hold papers to a high standard "
            "and err toward rejection when claims are not clearly supported, experiments are "
            "insufficient, or novelty is marginal. You are direct and sharp in your assessments."
        ),
    }

    def __init__(
        self,
        n_reviewers: int = 2,
        reviewer_models: Optional[list[str]] = None,
        ac_model: str = MODEL_ID,
        n_rounds: int = 2,
        ac_sees: Literal["paper", "transcript", "both"] = "both",
        conference: str = "NeurIPS 2025",
        max_output_tokens: int = 512,
        ac_max_output_tokens: int = 128,
        reviews: Optional[list[dict]] = None,
        persona_seed: int = 0,
        cache_config: Optional[dict] = None,
    ):
        self.n_reviewers = n_reviewers
        self.ac_model = ac_model
        self.n_rounds = n_rounds
        self.ac_sees = ac_sees
        self.conference = conference
        self.max_output_tokens = max_output_tokens
        self.ac_max_output_tokens = ac_max_output_tokens
        self.reviews = reviews
        self.persona_seed = persona_seed
        self.cache_config = cache_config

        # Resolve reviewer model list (cycle if too short)
        base_models = reviewer_models or [MODEL_ID]
        self._reviewer_models: list[str] = list(
            itertools.islice(itertools.cycle(base_models), n_reviewers)
        )

        # Sample personas deterministically
        rng = random.Random(persona_seed)
        persona_names = list(self.PERSONAS.keys())
        self._personas: list[str] = [rng.choice(persona_names) for _ in range(n_reviewers)]

    def to_config_dict(self) -> dict:
        return {
            "rig_type": "DebateRig",
            "n_reviewers": self.n_reviewers,
            "reviewer_models": self._reviewer_models,
            "ac_model": self.ac_model,
            "n_rounds": self.n_rounds,
            "ac_sees": self.ac_sees,
            "conference": self.conference,
            "personas": self._personas,
            "has_reviews": self.reviews is not None,
        }

    def _format_transcript(self, transcript: list[dict]) -> str:
        lines = []
        for entry in transcript:
            if entry.get("role") == "AC":
                lines.append(f"[Area Chair]: {entry['text']}")
            else:
                r = entry["reviewer_index"] + 1
                persona = entry["persona"].capitalize()
                lines.append(f"[Reviewer {r} ({persona})]: {entry['text']}")
        return "\n\n".join(lines)

    def _build_reviewer_prompt(
        self,
        reviewer_index: int,
        round_num: int,
        paper_md: str,
        transcript: list[dict],
    ) -> str:
        n = self.n_reviewers
        conf = self.conference
        persona_name = self._personas[reviewer_index]
        persona_desc = self.PERSONAS[persona_name]

        prompt = (
            f"You are Reviewer {reviewer_index + 1} of {n} at {conf}.\n"
            f"Your reviewing style: {persona_desc}\n\n"
        )

        # Inject assigned real review in round 1 only
        if round_num == 1 and self.reviews:
            assigned_review = self.reviews[reviewer_index % len(self.reviews)]
            review_text = assigned_review.get("comments", "")
            prompt += (
                "Your assigned review for this paper (use this to seed your opening argument):\n"
                "--- BEGIN ASSIGNED REVIEW ---\n"
                f"{review_text}\n"
                "--- END ASSIGNED REVIEW ---\n\n"
            )

        prompt += (
            f"Paper:\n"
            "--- BEGIN PAPER ---\n"
            f"{paper_md}\n"
            "--- END PAPER ---\n\n"
        )

        if transcript:
            prompt += (
                "Debate transcript so far:\n"
                "--- BEGIN TRANSCRIPT ---\n"
                f"{self._format_transcript(transcript)}\n"
                "--- END TRANSCRIPT ---\n\n"
            )

        if round_num > 1:
            prompt += f"This is round {round_num} of {self.n_rounds}. "
        prompt += (
            "Write your review argument. Be concise (2–4 paragraphs). "
            "Do not state a final accept/reject verdict — the area chair will decide. "
            "Focus on the scientific merit, novelty, clarity, and experimental rigor."
        )
        return prompt

    def _build_ac_prompt(self, paper_md: str, transcript: list[dict]) -> str:
        conf = self.conference
        prompt = f"You are the Area Chair at {conf} responsible for making the final acceptance decision.\n\n"

        if self.ac_sees in ("paper", "both"):
            prompt += (
                "Paper:\n"
                "--- BEGIN PAPER ---\n"
                f"{paper_md}\n"
                "--- END PAPER ---\n\n"
            )

        if self.ac_sees in ("transcript", "both") and transcript:
            prompt += (
                "Reviewer debate transcript:\n"
                "--- BEGIN TRANSCRIPT ---\n"
                f"{self._format_transcript(transcript)}\n"
                "--- END TRANSCRIPT ---\n\n"
            )

        prompt += (
            "Based on the above, make your final acceptance decision.\n"
            "Respond with ONLY a JSON object and nothing else:\n"
            '{"prediction": "ACCEPT", "confidence": 0.85}\n'
            "or\n"
            '{"prediction": "REJECT", "confidence": 0.72}\n\n'
            "confidence is your certainty in the decision (0.0 = completely uncertain, 1.0 = certain)."
        )
        return prompt

    def _parse_ac_output(self, text: str) -> tuple[Optional[AcceptReject], float]:
        """Extract (prediction, confidence) from AC JSON output."""
        match = re.search(
            r'\{\s*"prediction"\s*:\s*"(ACCEPT|REJECT)"\s*,\s*"confidence"\s*:\s*([0-9]*\.?[0-9]+)\s*\}',
            text, re.IGNORECASE,
        )
        if match:
            pred = match.group(1).upper()
            conf = min(1.0, max(0.0, float(match.group(2))))
            return pred, conf  # type: ignore[return-value]
        # Try flipped key order
        match2 = re.search(
            r'\{\s*"confidence"\s*:\s*([0-9]*\.?[0-9]+)\s*,\s*"prediction"\s*:\s*"(ACCEPT|REJECT)"\s*\}',
            text, re.IGNORECASE,
        )
        if match2:
            conf = min(1.0, max(0.0, float(match2.group(1))))
            pred = match2.group(2).upper()
            return pred, conf  # type: ignore[return-value]
        # Fallback: extract prediction only, default confidence
        upper = text.upper()
        if "ACCEPT" in upper and "REJECT" not in upper:
            return "ACCEPT", 0.5
        if "REJECT" in upper and "ACCEPT" not in upper:
            return "REJECT", 0.5
        return None, 0.5

    def __call__(
        self,
        client,
        paper_md: str,
        model_id_override: Optional[str] = None,
        estimate_only: bool = False,
        cache_config: Optional[dict] = None,
    ) -> tuple[Optional[AcceptReject], float, list[dict], Usage]:
        """
        Run the debate and return (prediction, confidence, transcript, usage).

        Transcript entries:
          Reviewer turns: {round, reviewer_index, persona, model_id, text}
          AC final entry: {role:"AC", model_id, prediction, confidence, text}

        In estimate_only mode: returns (None, 0.0, [], lower_bound_Usage).
        Note: estimate is a lower bound because transcript grows each round.
        """
        cache_cfg = cache_config or self.cache_config

        # estimate_only: build worst-case round-1 prompt and scale
        if estimate_only:
            r1_prompt = self._build_reviewer_prompt(0, 1, paper_md, [])
            ac_prompt = self._build_ac_prompt(paper_md, [])
            reviewer_tokens = estimate_tokens(r1_prompt) * self.n_reviewers * self.n_rounds
            ac_tokens = estimate_tokens(ac_prompt)
            est = Usage(input_tokens=reviewer_tokens + ac_tokens, output_tokens=0, latency_ms=0, calls=0)
            return None, 0.0, [], est

        if cache_cfg:
            config_hash = build_config_hash(self.to_config_dict())
            cached = load_cached(cache_cfg["paper_id"], config_hash, cache_cfg["dataset"])
            if cached:
                return (
                    cached["prediction"],
                    cached["confidence"],
                    cached["transcript"],
                    Usage(**cached["usage"]),
                )

        # Resolve model IDs (override applies to all)
        reviewer_models = (
            [model_id_override] * self.n_reviewers
            if model_id_override
            else self._reviewer_models
        )
        ac_model = model_id_override or self.ac_model

        transcript: list[dict] = []
        total_usage = Usage()

        for round_num in range(1, self.n_rounds + 1):
            # Re-shuffle reviewer order each round
            rng = random.Random(round_num + self.persona_seed)
            order = list(range(self.n_reviewers))
            rng.shuffle(order)

            for idx in order:
                prompt = self._build_reviewer_prompt(idx, round_num, paper_md, transcript)
                text, usage = converse_text(
                    client, prompt,
                    model_id=reviewer_models[idx],
                    max_tokens=self.max_output_tokens,
                )
                total_usage += usage
                transcript.append({
                    "round": round_num,
                    "reviewer_index": idx,
                    "persona": self._personas[idx],
                    "model_id": reviewer_models[idx],
                    "text": text or "",
                })

        # AC final decision
        ac_prompt = self._build_ac_prompt(paper_md, transcript)
        ac_text, ac_usage = converse_text(
            client, ac_prompt, model_id=ac_model, max_tokens=self.ac_max_output_tokens
        )
        total_usage += ac_usage
        prediction, confidence = self._parse_ac_output(ac_text or "")
        transcript.append({
            "role": "AC",
            "model_id": ac_model,
            "prediction": prediction,
            "confidence": confidence,
            "text": ac_text or "",
        })

        if cache_cfg:
            save_result(cache_cfg["paper_id"], config_hash, cache_cfg["dataset"], {
                "prediction": prediction,
                "confidence": confidence,
                "transcript": transcript,
                "usage": dataclasses.asdict(total_usage),
            })
            update_manifest(
                config_hash, "DebateRig",
                f"n_reviewers={self.n_reviewers} n_rounds={self.n_rounds} ac_sees={self.ac_sees}",
                cache_cfg["dataset"],
            )

        return prediction, confidence, transcript, total_usage


# ---------------------------------------------------------------------------
# Rig 4: LLM Jury (confidence-weighted majority vote over mixed rigs)
# ---------------------------------------------------------------------------

class LLMJuryRig:
    """
    Run multiple judge rigs/models and aggregate via confidence-weighted majority vote.

    Members can be a mix of AcceptanceRig and DebateRig instances paired with
    model IDs. PairwiseRig cannot be used here (requires two papers per call).

    Tie-breaking: ACCEPT wins on equal weight (consistent with the high
    acceptance-rate prior in the datasets).

    Parameters
    ----------
    members : list of (rig_instance, model_id) tuples
        Each member is run independently; DebateRig members contribute their
        AC confidence score; AcceptanceRig members use confidence=1.0.
    conference : str
        Passed through for display purposes only.
    cache_config : dict, optional
        {"dataset": str, "paper_id": str} — enables auto cache lookup/save.
    """

    def __init__(
        self,
        members: list[tuple],
        conference: str = "NeurIPS 2025",
        cache_config: Optional[dict] = None,
    ):
        for rig, _ in members:
            if isinstance(rig, PairwiseRig):
                raise ValueError(
                    "PairwiseRig cannot be a LLMJuryRig member: it requires two papers per call, "
                    "but the jury interface evaluates a single paper. Use AcceptanceRig or DebateRig."
                )
        self.members = members
        self.conference = conference
        self.cache_config = cache_config

    def to_config_dict(self) -> dict:
        return {
            "rig_type": "LLMJuryRig",
            "conference": self.conference,
            "members": [
                {
                    "rig_type": type(rig).__name__,
                    "model_id": model_id,
                    "rig_params": rig.to_config_dict(),
                }
                for rig, model_id in self.members
            ],
        }

    def __call__(
        self,
        client,
        paper_md: str,
        reviews_text: Optional[str] = None,
        estimate_only: bool = False,
        cache_config: Optional[dict] = None,
    ) -> tuple[Optional[AcceptReject], list[dict], Usage]:
        """
        Run all members and return (final_prediction, member_verdicts, combined_usage).

        member_verdicts: list of dicts, one per member:
          {rig_type, model_id, prediction, confidence, usage, [transcript]}
        """
        cache_cfg = cache_config or self.cache_config

        if cache_cfg and not estimate_only:
            config_hash = build_config_hash(self.to_config_dict())
            cached = load_cached(cache_cfg["paper_id"], config_hash, cache_cfg["dataset"])
            if cached:
                return cached["prediction"], cached["member_verdicts"], Usage(**cached["usage"])

        verdicts: list[dict] = []
        total_usage = Usage()

        for rig, model_id in self.members:
            verdict: dict[str, Any] = {
                "rig_type": type(rig).__name__,
                "model_id": model_id,
            }

            if isinstance(rig, AcceptanceRig):
                pred, usage = rig(client, paper_md, reviews_text, model_id=model_id, estimate_only=estimate_only)
                verdict.update({"prediction": pred, "confidence": 1.0, "usage": dataclasses.asdict(usage)})

            elif isinstance(rig, DebateRig):
                pred, conf, transcript, usage = rig(
                    client, paper_md, model_id_override=model_id, estimate_only=estimate_only
                )
                verdict.update({
                    "prediction": pred,
                    "confidence": conf,
                    "usage": dataclasses.asdict(usage),
                    "transcript": transcript,
                })

            else:
                raise TypeError(f"Unsupported rig type in LLMJuryRig: {type(rig).__name__}")

            total_usage += Usage(**verdict["usage"])
            verdicts.append(verdict)

        # Confidence-weighted majority vote
        accept_weight = sum(v["confidence"] for v in verdicts if v["prediction"] == "ACCEPT")
        reject_weight = sum(v["confidence"] for v in verdicts if v["prediction"] == "REJECT")

        if accept_weight == 0 and reject_weight == 0:
            final_prediction = None
        elif accept_weight >= reject_weight:
            final_prediction = "ACCEPT"  # tie → ACCEPT
        else:
            final_prediction = "REJECT"

        if cache_cfg and not estimate_only:
            save_result(cache_cfg["paper_id"], config_hash, cache_cfg["dataset"], {
                "prediction": final_prediction,
                "member_verdicts": verdicts,
                "usage": dataclasses.asdict(total_usage),
            })
            update_manifest(
                config_hash, "LLMJuryRig",
                f"n_members={len(self.members)}",
                cache_cfg["dataset"],
            )

        return final_prediction, verdicts, total_usage
