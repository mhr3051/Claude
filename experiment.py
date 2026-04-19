# iteration: 0, ts: 2026-04-19T00:00:00+00:00
"""Current experiment. The agent edits this file each iteration."""

import os
import uuid


def run(config: dict, tf_client=None) -> dict:
    """Run the current distillation experiment.

    In dry-run mode (tf_client=None), returns a fake checkpoint.
    In live mode, generates synthetic data from the teacher via batch
    inference, then fine-tunes the student with HF push.
    """
    if tf_client is None:
        return {"checkpoint_id": f"fake-ckpt-{uuid.uuid4().hex[:8]}"}

    student = config["students"][0]["model"]
    teacher = config["teacher"]["model"]
    hf_org = config.get("hf", {}).get("org", "")
    hf_token = os.environ.get("HF_API_TOKEN")
    ft_params = config.get("fine_tuning", {})

    # 1. Generate synthetic data from teacher via batch inference
    # TODO: build real prompt set from eval examples or self-instruct
    prompts = [
        [{"role": "user", "content": "Explain what distillation means in ML."}],
        [{"role": "user", "content": "What is LoRA fine-tuning?"}],
    ]
    batch_file = "/tmp/batch_requests.jsonl"
    tf_client.build_batch_requests_file(teacher, prompts, batch_file)
    batch_id = tf_client.create_batch(teacher, batch_file)
    batch = tf_client.wait_for_batch(batch_id, timeout=config["budget"]["time_per_experiment_minutes"] * 60)

    # Download synthetic data and convert to training format
    raw_output = "/tmp/batch_output.jsonl"
    tf_client.download_batch_results(batch, raw_output)
    training_file = _convert_to_training_format(raw_output, "/tmp/training.jsonl")

    # 2. Upload dataset and fine-tune
    file_id = tf_client.upload_dataset(training_file)
    suffix = f"distill-{uuid.uuid4().hex[:6]}"
    hf_repo = f"{hf_org}/{student.split('/')[-1]}-{suffix}" if hf_org else None

    job_id = tf_client.create_fine_tuning_job(
        model=student,
        training_file_id=file_id,
        hyperparams={
            "lora": ft_params.get("lora", True),
            "lora_r": ft_params.get("lora_r", 16),
            "lora_alpha": ft_params.get("lora_alpha", 16),
            "lora_dropout": ft_params.get("lora_dropout", 0.05),
            "n_epochs": ft_params.get("n_epochs", 3),
            "batch_size": ft_params.get("batch_size", 8),
            "learning_rate": ft_params.get("learning_rate", 1e-5),
            "context_length": ft_params.get("context_length", 8192),
            "packing": ft_params.get("packing", True),
            "max_grad_norm": ft_params.get("max_grad_norm", 1.0),
        },
        suffix=suffix,
        hf_repo=hf_repo,
        hf_token=hf_token,
    )
    job = tf_client.wait_for_job(
        job_id, timeout=config["budget"]["time_per_experiment_minutes"] * 60
    )

    checkpoints = tf_client.get_checkpoints(job_id)
    ckpt_id = checkpoints[-1].id if checkpoints else job_id

    return {
        "checkpoint_id": ckpt_id,
        "job_id": job_id,
        "model": student,
        "hf_model_id": hf_repo or ckpt_id,
    }


def _convert_to_training_format(raw_path: str, output_path: str) -> str:
    """Convert batch inference output into conversational training JSONL."""
    import json
    with open(raw_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            body = record.get("response", {}).get("body", {})
            choices = body.get("choices", [])
            if not choices:
                continue
            request_body = record.get("request", {}).get("body", {})
            messages = request_body.get("messages", [])
            assistant_msg = choices[0].get("message", {})
            messages.append({"role": "assistant", "content": assistant_msg.get("content", "")})
            f_out.write(json.dumps({"messages": messages}) + "\n")
    return output_path
