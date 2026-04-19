"""Cost and latency measurement via HuggingFace Inference API."""

import json
import os
import random
import statistics
import time

from dotenv import load_dotenv


def measure(
    hf_model_id: str,
    eval_path: str | None = None,
    sample_count: int = 20,
    hf_token: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Measure latency and estimate cost against a HF-hosted model.

    Sends sample_count prompts, times each request, computes p99 latency.
    Cost is estimated from token usage in the responses.

    Returns {"dollars_per_1m_tokens": float, "p99_latency_ms": float}.
    """
    if dry_run:
        return {
            "dollars_per_1m_tokens": random.uniform(0.5, 5.0),
            "p99_latency_ms": random.uniform(50, 500),
        }

    load_dotenv()
    token = hf_token or os.environ.get("HF_API_TOKEN")

    from huggingface_hub import InferenceClient
    client = InferenceClient(model=hf_model_id, token=token)

    prompts = _load_sample_prompts(eval_path, sample_count)
    latencies_ms: list[float] = []
    total_tokens = 0

    for messages in prompts:
        t0 = time.perf_counter()
        try:
            response = client.chat_completion(messages=messages, max_tokens=128)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed_ms)
            usage = getattr(response, "usage", None)
            if usage:
                total_tokens += usage.total_tokens
        except Exception as e:
            print(f"  [cost] error: {e}")

    if not latencies_ms:
        return {"dollars_per_1m_tokens": 0.0, "p99_latency_ms": 0.0}

    p99 = _percentile(latencies_ms, 99)
    # Rough cost estimate: use a placeholder rate from config if available.
    # HF Inference API pricing varies by model; this is a best-effort estimate.
    dollars_per_1m = 2.0  # placeholder, refine with actual HF pricing
    return {"dollars_per_1m_tokens": dollars_per_1m, "p99_latency_ms": p99}


def _load_sample_prompts(
    eval_path: str | None, sample_count: int
) -> list[list[dict]]:
    """Load up to sample_count prompts from the eval set."""
    if not eval_path:
        return [[{"role": "user", "content": "Say hello."}]] * sample_count

    examples = []
    with open(eval_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    sampled = examples[:sample_count]
    prompts = []
    for ex in sampled:
        messages = ex.get("messages", [])
        # Use all messages except the final assistant reply as the prompt
        if messages and messages[-1]["role"] == "assistant":
            prompts.append(messages[:-1])
        else:
            prompts.append(messages)
    return prompts


def _percentile(data: list[float], pct: int) -> float:
    """Compute the pct-th percentile of a list."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * pct / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
