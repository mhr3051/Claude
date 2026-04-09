# Distillation Autoresearch

## Experiment Plan

TODO: The agent will populate this section at the start of each research session.
It should contain a ranked list of hypotheses to test, each with a one-line
rationale, estimated cost, and priority.

## Experiment Log

| Iteration | Timestamp | Score | Status | Checkpoint |
| --- | --- | --- | --- | --- |
| 4 | 2026-04-09 19:45:47 | 0.2948 | REVERTED | fake-ckpt-8553201c |
| 3 | 2026-04-09 19:45:46 | 0.2880 | REVERTED | fake-ckpt-146e624c |
| 2 | 2026-04-09 19:45:46 | 0.1738 | REVERTED | fake-ckpt-f83a3a50 |
| 1 | 2026-04-09 19:45:46 | 0.3515 | KEPT | fake-ckpt-17045d48 |

## Rules

1. The agent may only edit `experiment.py`. All harness code is off-limits.
2. The agent may append to this file (`program.md`) but not delete or rewrite existing content.
3. Each experiment must stay within the per-experiment time and dollar budget in `config.yaml`.
4. The agent should update the Experiment Plan section when its strategy changes.
5. The agent must not call external APIs directly. All API access goes through the harness modules.
