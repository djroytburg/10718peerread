"""
Shared utilities for dispatching calls to Amazon Bedrock and estimating prompt cost.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError


# ---------------------------------------------------------------------------
# Model / region defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "us.meta.llama3-3-70b-instruct-v1:0"
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Rough chars-per-token estimate for English text.
CHARS_PER_TOKEN = 4

# Botocore error codes that indicate rate limiting.
THROTTLE_CODES = (
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceQuotaExceededException",
)

# Pricing per 1K tokens (USD) for supported models.
# Source: https://aws.amazon.com/bedrock/pricing/
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Meta Llama
    "us.meta.llama3-3-70b-instruct-v1:0":  {"input": 0.00072, "output": 0.00072},
    "us.meta.llama3-1-70b-instruct-v1:0":  {"input": 0.00072, "output": 0.00072},
    "us.meta.llama3-2-90b-instruct-v1:0":  {"input": 0.00072, "output": 0.00072},
    "us.meta.llama3-1-8b-instruct-v1:0":   {"input": 0.00022, "output": 0.00022},
    "us.meta.llama3-2-11b-instruct-v1:0":  {"input": 0.00016, "output": 0.00016},
    "us.meta.llama3-2-3b-instruct-v1:0":   {"input": 0.00015, "output": 0.00015},
    "us.meta.llama3-2-1b-instruct-v1:0":   {"input": 0.00010, "output": 0.00010},
    # Amazon Nova
    "us.amazon.nova-pro-v1:0":             {"input": 0.00080, "output": 0.00320},
    "us.amazon.nova-lite-v1:0":            {"input": 0.00006, "output": 0.00024},
    "us.amazon.nova-micro-v1:0":           {"input": 0.000035,"output": 0.000140},
    # Anthropic Claude
    "us.anthropic.claude-3-5-haiku-20241022-v1:0":  {"input": 0.00080, "output": 0.00400},
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {"input": 0.00300, "output": 0.01500},
    "us.anthropic.claude-3-haiku-20240307-v1:0":    {"input": 0.00025, "output": 0.00125},
    # Mistral
    "mistral.mixtral-8x7b-instruct-v0:1":  {"input": 0.00045, "output": 0.00070},
    "mistral.mistral-7b-instruct-v0:2":    {"input": 0.00015, "output": 0.00020},
    # DeepSeek
    "us.deepseek.r1-v1:0":                 {"input": 0.00135, "output": 0.00535},
    "deepseek.v3.2":                        {"input": 0.00062, "output": 0.00185},
    # Magistral Small: pricing TBD — token counts reported, cost set to 0 until confirmed
    "mistral.magistral-small-2509":         {"input": 0.0, "output": 0.0},
    "google.gemma-3-12b-it":               {"input": 0.000090, "output": 0.000290},
}


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost(self, model_id: str = MODEL_ID) -> float:
        """Estimated cost in USD based on token counts and model pricing."""
        pricing = MODEL_PRICING.get(model_id)
        if pricing is None:
            return float("nan")
        return (
            self.input_tokens  / 1000 * pricing["input"] +
            self.output_tokens / 1000 * pricing["output"]
        )

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            latency_ms=self.latency_ms + other.latency_ms,
            calls=self.calls + other.calls,
        )

    def __iadd__(self, other: "Usage") -> "Usage":
        self.input_tokens  += other.input_tokens
        self.output_tokens += other.output_tokens
        self.latency_ms    += other.latency_ms
        self.calls         += other.calls
        return self

    def summary(self, model_id: str = MODEL_ID) -> str:
        return (
            f"calls={self.calls}  "
            f"input={self.input_tokens:,}tok  "
            f"output={self.output_tokens:,}tok  "
            f"cost=${self.cost(model_id):.4f}"
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars per token for English)."""
    return len(text) // CHARS_PER_TOKEN


def estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens across all message content blocks."""
    total = 0
    for msg in messages:
        for block in msg.get("content", []):
            if "text" in block:
                total += estimate_tokens(block["text"])
    return total


def estimate_cost(input_tokens: int, output_tokens: int, model_id: str = MODEL_ID) -> float:
    """Estimate cost in USD for given token counts."""
    pricing = MODEL_PRICING.get(model_id)
    if pricing is None:
        return float("nan")
    return input_tokens / 1000 * pricing["input"] + output_tokens / 1000 * pricing["output"]


# ---------------------------------------------------------------------------
# Core dispatch
# ---------------------------------------------------------------------------

def converse(
    client,
    messages: list[dict[str, Any]],
    model_id: str = MODEL_ID,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    stop_sequences: Optional[list[str]] = None,
    max_retries: int = 5,
    retry_wait: float = 30.0,
) -> tuple[Optional[str], Usage]:
    """
    Call Bedrock Converse API and return (response_text, Usage).

    response_text is None on failure. Automatically retries on throttling.
    """
    inference_config: dict[str, Any] = {
        "maxTokens": max_tokens,
        "temperature": temperature,
    }
    if stop_sequences:
        inference_config["stopSequences"] = stop_sequences

    kwargs: dict[str, Any] = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": inference_config,
    }

    for attempt in range(max_retries):
        try:
            response = client.converse(**kwargs)
            break
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in THROTTLE_CODES and attempt < max_retries - 1:
                print(
                    f"Rate limit ({code}); waiting {retry_wait:.0f}s then retrying "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(retry_wait)
                continue
            print(f"Bedrock ClientError: {e}")
            return None, Usage()
        except Exception as e:
            print(f"Bedrock error: {e}")
            return None, Usage()

    raw_usage = response.get("usage", {})
    usage = Usage(
        input_tokens=raw_usage.get("inputTokens", 0),
        output_tokens=raw_usage.get("outputTokens", 0),
        latency_ms=response.get("metrics", {}).get("latencyMs", 0),
        calls=1,
    )

    content = response.get("output", {}).get("message", {}).get("content", [])
    text = "".join(block.get("text", "") for block in content).strip()
    return text, usage


def converse_text(
    client,
    prompt: str,
    **kwargs,
) -> tuple[Optional[str], Usage]:
    """Convenience wrapper: send a single user text prompt, return (text, Usage)."""
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    return converse(client, messages, **kwargs)
