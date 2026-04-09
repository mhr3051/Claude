# Distillation Autoresearch

## Experiment Plan

TODO: The agent will populate this section at the start of each research session.
It should contain a ranked list of hypotheses to test, each with a one-line
rationale, estimated cost, and priority.

## Experiment Log

| Iteration | Timestamp | Score | Status | Checkpoint |
| --- | --- | --- | --- | --- |

## Rules

1. The agent may only edit `experiment.py`. All harness code is off-limits.
2. The agent may append to this file (`program.md`) but not delete or rewrite existing content.
3. Each experiment must stay within the per-experiment time and dollar budget in `config.yaml`.
4. The agent should update the Experiment Plan section when its strategy changes.
5. The agent must not call external APIs directly. All API access goes through the harness modules.
