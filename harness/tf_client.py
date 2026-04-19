"""Token Factory API client. Wraps fine-tuning, batch inference, and file ops."""

import json
import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


class TokenFactoryClient:
    BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

    def __init__(self, api_key: str | None = None):
        load_dotenv()
        self._key = api_key or os.environ["NEBIUS_API_KEY"]
        self.client = OpenAI(base_url=self.BASE_URL, api_key=self._key)

    def upload_dataset(self, path: str, purpose: str = "fine-tune") -> str:
        """Upload a JSONL file. Returns the file ID."""
        with open(path, "rb") as f:
            resp = self.client.files.create(file=f, purpose=purpose)
        return resp.id

    def create_fine_tuning_job(
        self,
        model: str,
        training_file_id: str,
        hyperparams: dict | None = None,
        suffix: str | None = None,
        hf_repo: str | None = None,
        hf_token: str | None = None,
    ) -> str:
        """Create a fine-tuning job. Returns the job ID.

        If hf_repo and hf_token are provided, the trained weights are
        automatically pushed to HuggingFace when the job succeeds.
        """
        kwargs: dict = {"model": model, "training_file": training_file_id}
        if hyperparams:
            kwargs["hyperparameters"] = hyperparams
        if suffix:
            kwargs["suffix"] = suffix

        integrations = []
        if hf_repo and hf_token:
            integrations.append({
                "type": "hf",
                "hf": {"output_repo_name": hf_repo, "api_token": hf_token},
            })
        if integrations:
            kwargs["integrations"] = integrations

        job = self.client.fine_tuning.jobs.create(**kwargs)
        return job.id

    def wait_for_job(
        self, job_id: str, poll_interval: int = 15, timeout: int | None = None
    ) -> object:
        """Poll a fine-tuning job until it reaches a terminal status.

        Returns the final job object. Raises RuntimeError on failure.
        """
        deadline = time.monotonic() + timeout if timeout else None
        terminal = {"succeeded", "failed", "cancelled"}
        while True:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            if job.status in terminal:
                if job.status != "succeeded":
                    err = getattr(job, "error", None)
                    raise RuntimeError(
                        f"Fine-tuning job {job_id} {job.status}: {err}"
                    )
                return job
            if deadline and time.monotonic() > deadline:
                raise TimeoutError(
                    f"Fine-tuning job {job_id} timed out (status={job.status})"
                )
            print(f"  [tf] job {job_id}: {job.status} "
                  f"({getattr(job, 'trained_steps', '?')}/{getattr(job, 'total_steps', '?')} steps)")
            time.sleep(poll_interval)

    def get_checkpoints(self, job_id: str) -> list:
        """List checkpoints for a completed fine-tuning job."""
        resp = self.client.fine_tuning.jobs.checkpoints.list(job_id)
        return resp.data

    def create_batch(self, model: str, requests_jsonl_path: str) -> str:
        """Upload a JSONL of chat completion requests and start a batch job.

        Each line in the JSONL should be:
        {"custom_id": "...", "method": "POST", "url": "/v1/chat/completions",
         "body": {"model": "...", "messages": [...]}}

        Returns the batch ID.
        """
        file_id = self.upload_dataset(requests_jsonl_path, purpose="batch")
        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return batch.id

    def wait_for_batch(
        self, batch_id: str, poll_interval: int = 30, timeout: int | None = None
    ) -> object:
        """Poll a batch job until completed. Returns the batch object."""
        deadline = time.monotonic() + timeout if timeout else None
        while True:
            batch = self.client.batches.retrieve(batch_id)
            if batch.status in {"completed", "failed", "cancelled", "expired"}:
                if batch.status != "completed":
                    raise RuntimeError(
                        f"Batch {batch_id} {batch.status}: "
                        f"{getattr(batch, 'errors', None)}"
                    )
                return batch
            if deadline and time.monotonic() > deadline:
                raise TimeoutError(
                    f"Batch {batch_id} timed out (status={batch.status})"
                )
            counts = getattr(batch, "request_counts", None)
            print(f"  [tf] batch {batch_id}: {batch.status} "
                  f"(completed={getattr(counts, 'completed', '?')}/"
                  f"total={getattr(counts, 'total', '?')})")
            time.sleep(poll_interval)

    def download_batch_results(self, batch, output_path: str) -> str:
        """Download batch output file. Returns the output path."""
        content = self.client.files.content(batch.output_file_id)
        content.write_to_file(output_path)
        return output_path

    def build_batch_requests_file(
        self, model: str, prompts: list[dict], output_path: str,
        sampling_params: dict | None = None,
    ) -> str:
        """Build a JSONL file of batch requests from a list of message lists.

        Each entry in prompts should be a list of message dicts.
        Returns the path to the written file.
        """
        params = sampling_params or {}
        with open(output_path, "w") as f:
            for i, messages in enumerate(prompts):
                req = {
                    "custom_id": f"req-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {"model": model, "messages": messages, **params},
                }
                f.write(json.dumps(req) + "\n")
        return output_path
