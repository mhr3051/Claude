# Distillation Autoresearch Harness

Automated distillation of large teacher models into smaller students using
Nebius Token Factory APIs for fine-tuning and HuggingFace for model hosting
and evaluation. Inspired by Karpathy's autoresearch design: one editable file,
one scoreable metric, keep-or-revert ratchet.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file with your API keys (this file is gitignored):

```bash
NEBIUS_API_KEY=your_token_factory_key
HF_API_TOKEN=your_huggingface_token
```

Edit `config.yaml` to set your HuggingFace org and model preferences.

## Running the loop

### Dry run (no API calls, uses stubs)

```bash
uv run python -m harness.runner --dry-run --iterations 5
```

### Live run (calls Token Factory and HuggingFace APIs)

```bash
uv run python -m harness.runner --iterations 5
```

Or use the default iteration count from `config.yaml`:

```bash
uv run python -m harness.runner
```

## Project structure

- `experiment.py` -- the single file the agent edits each round
- `program.md` -- agent instructions, experiment plan, and log
- `config.yaml` -- customer inputs (teacher, students, budget, weights)
- `harness/` -- locked-down infrastructure the agent does not touch
  - `tf_client.py` -- Token Factory API client (fine-tuning, batch inference)
  - `eval.py` -- evaluation via HuggingFace Inference API (exact match)
  - `cost.py` -- latency and cost measurement via HuggingFace Inference API
  - `score.py` -- weighted scoring formula
  - `runner.py` -- the outer ratchet loop
- `eval/` -- evaluation dataset (JSONL)
- `experiments/` -- auto-generated logs and best result tracking

## How it works

Each iteration the runner:

1. Injects a marker into `experiment.py` (creating a diff)
2. Runs the experiment (batch inference for data, fine-tuning on Token Factory)
3. Pushes trained weights to HuggingFace
4. Evaluates quality via HF Inference API (exact match)
5. Measures latency and cost via HF Inference API
6. Computes a weighted score combining quality, cost, and latency
7. Keeps the change if the score improved, reverts otherwise
8. Logs the result to `program.md`

See `distillation-autoresearch-plan.md` for the full design.
