"""Combines quality, cost, and latency into a single ratchet metric."""


def compute(
    quality_score: float,
    dollars_per_1m_tokens: float,
    p99_latency_ms: float,
    weights: dict,
) -> float:
    """Compute the weighted ratchet score.

    score = w_quality * quality
          - w_cost * normalized_cost
          - w_latency * normalized_latency

    Cost is normalized to [0,1] by capping at $10/1M tokens.
    Latency is normalized to [0,1] by capping at 1000ms.
    """
    w_q = weights.get("w_quality", 1.0)
    w_c = weights.get("w_cost", 0.3)
    w_l = weights.get("w_latency", 0.2)

    norm_cost = min(dollars_per_1m_tokens / 10.0, 1.0)
    norm_latency = min(p99_latency_ms / 1000.0, 1.0)

    return w_q * quality_score - w_c * norm_cost - w_l * norm_latency
