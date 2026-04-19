"""Eval runner. Scores a model via HuggingFace Inference API."""

import json
import os
import random

from dotenv import load_dotenv


def evaluate(
    hf_model_id: str,
    eval_path: str,
    hf_token: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Run the eval set against a HuggingFace-hosted model.

    Loads eval JSONL, sends each prompt via HF Inference API,
    compares response to expected output (exact match).
    Returns {"quality_score": float, "total": int, "correct": int}.
    """
    if dry_run:
        return {"quality_score": random.uniform(0.3, 0.95), "total": 10, "correct": 5}

    load_dotenv()
    token = hf_token or os.environ.get("HF_API_TOKEN")

    from huggingface_hub import InferenceClient
    client = InferenceClient(model=hf_model_id, token=token)

    examples = []
    with open(eval_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    correct = 0
    total = len(examples)

    for ex in examples:
        messages = ex["messages"]
        expected = ex.get("expected", messages[-1]["content"] if messages else "")
        prompt_messages = messages[:-1] if expected == messages[-1]["content"] else messages

        try:
            response = client.chat_completion(messages=prompt_messages)
            answer = response.choices[0].message.content.strip()
            if answer == expected.strip():
                correct += 1
        except Exception as e:
            print(f"  [eval] error: {e}")

    quality = correct / total if total > 0 else 0.0
    return {"quality_score": quality, "total": total, "correct": correct}
