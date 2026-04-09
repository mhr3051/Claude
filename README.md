# Distillation Autoresearch Harness

Automated distillation of large teacher models into smaller students using
Token Factory APIs. Inspired by Karpathy's autoresearch design: one editable
file, one scoreable metric, keep-or-revert ratchet.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv
uv pip install pyyaml
```

## Running the loop

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
- `harness/` -- locked-down infrastructure (runner, eval, cost, score, API client)
- `experiments/` -- auto-generated logs and best result tracking

## How it works

Each iteration the runner:

1. Injects a marker into `experiment.py` (creating a diff)
2. Runs the experiment to produce a checkpoint
3. Evaluates quality, cost, and latency (stubbed in milestone 1)
4. Computes a weighted score
5. Keeps the change if the score improved, reverts otherwise
6. Logs the result to `program.md`

See `distillation-autoresearch-plan.md` for the full design.
