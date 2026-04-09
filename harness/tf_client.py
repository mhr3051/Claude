"""Token Factory API client. Wraps post-training, batch inference, and endpoints."""

from typing import Any


def batch_inference(config: dict, prompts: list[str]) -> dict:
    """Submit a batch inference job to generate synthetic data from the teacher.

    Returns a dict with 'dataset_path' pointing to the generated dataset.
    """
    # TODO: milestone 2 -- call Token Factory batch inference API
    return {"dataset_path": "/tmp/fake_dataset.jsonl", "num_samples": 100}


def post_training(config: dict, dataset_path: str) -> str:
    """Submit a post-training (fine-tuning) job on Token Factory.

    Returns the checkpoint id of the resulting model.
    """
    # TODO: milestone 2 -- call Token Factory post-training API
    return "fake-checkpoint-0000"


def create_endpoint(checkpoint_id: str) -> str:
    """Spin up a dedicated inference endpoint for the given checkpoint.

    Returns the endpoint URL.
    """
    # TODO: milestone 2 -- call Token Factory inference API
    return f"https://fake-endpoint.example.com/{checkpoint_id}"


def delete_endpoint(endpoint_url: str) -> None:
    """Tear down a dedicated endpoint to stop billing."""
    # TODO: milestone 2 -- call Token Factory API to delete endpoint
    pass
