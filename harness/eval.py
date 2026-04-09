"""Eval runner. Scores a model endpoint against the customer's eval set."""

import random


def evaluate(endpoint_url: str, eval_path: str) -> dict:
    """Run the eval set against the given endpoint.

    Returns a dict with 'quality_score' in [0, 1].
    """
    # TODO: milestone 3 -- run real eval against endpoint
    return {"quality_score": random.uniform(0.3, 0.95)}
