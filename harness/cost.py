"""Cost and latency measurement against a dedicated endpoint."""

import random


def measure(endpoint_url: str) -> dict:
    """Hit the endpoint with a synthetic traffic trace.

    Returns dollars per 1M tokens and p99 latency in ms.
    """
    # TODO: milestone 3 -- measure real endpoint cost and latency
    return {
        "dollars_per_1m_tokens": random.uniform(0.5, 5.0),
        "p99_latency_ms": random.uniform(50, 500),
    }
