"""Outer loop: run experiment, score, commit or revert, repeat."""

import argparse
import importlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_PY = ROOT / "experiment.py"
PROGRAM_MD = ROOT / "program.md"
BEST_JSON = ROOT / "experiments" / "best.json"


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_best() -> dict:
    if BEST_JSON.exists():
        with open(BEST_JSON) as f:
            return json.load(f)
    return {"score": float("-inf"), "iteration": -1, "checkpoint_id": None}


def save_best(record: dict) -> None:
    BEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_JSON, "w") as f:
        json.dump(record, f, indent=2)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def inject_iteration_marker(iteration: int) -> None:
    """Write a marker comment into experiment.py so there is a real diff."""
    lines = EXPERIMENT_PY.read_text().splitlines(keepends=True)
    marker = f"# iteration: {iteration}, ts: {datetime.now(timezone.utc).isoformat()}\n"
    # Replace existing marker or prepend
    if lines and lines[0].startswith("# iteration:"):
        lines[0] = marker
    else:
        lines.insert(0, marker)
    EXPERIMENT_PY.write_text("".join(lines))


def append_log(iteration: int, score: float, kept: bool, checkpoint_id: str) -> None:
    """Append a one-line result to program.md under ## Experiment Log."""
    status = "KEPT" if kept else "REVERTED"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"| {iteration} | {ts} | {score:.4f} | {status} | {checkpoint_id} |\n"
    text = PROGRAM_MD.read_text()
    # Insert after the header row of the experiment log table
    anchor = "| --- | --- | --- | --- | --- |\n"
    if anchor in text:
        idx = text.index(anchor) + len(anchor)
        text = text[:idx] + line + text[idx:]
    else:
        text += line
    PROGRAM_MD.write_text(text)


def run_one_iteration(iteration: int, config: dict) -> None:
    print(f"\n{'='*50}")
    print(f"Iteration {iteration}")
    print(f"{'='*50}")

    # 1. Inject marker so experiment.py has a diff
    inject_iteration_marker(iteration)

    # 2. Run experiment
    if "experiment" in sys.modules:
        importlib.reload(sys.modules["experiment"])
    else:
        importlib.import_module("experiment")
    result = sys.modules["experiment"].run(config)
    checkpoint_id = result["checkpoint_id"]
    print(f"  checkpoint: {checkpoint_id}")

    # 3. Eval (stubbed)
    from harness.eval import evaluate
    eval_result = evaluate(
        endpoint_url=f"https://fake/{checkpoint_id}",
        eval_path=config["eval"]["path"],
    )
    print(f"  quality:    {eval_result['quality_score']:.4f}")

    # 4. Cost (stubbed)
    from harness.cost import measure
    cost_result = measure(endpoint_url=f"https://fake/{checkpoint_id}")
    print(f"  $/1M tok:   {cost_result['dollars_per_1m_tokens']:.2f}")
    print(f"  p99 lat:    {cost_result['p99_latency_ms']:.0f}ms")

    # 5. Score
    from harness.score import compute
    new_score = compute(
        quality_score=eval_result["quality_score"],
        dollars_per_1m_tokens=cost_result["dollars_per_1m_tokens"],
        p99_latency_ms=cost_result["p99_latency_ms"],
        weights=config["score_weights"],
    )
    print(f"  score:      {new_score:.4f}")

    # 6. Compare against best
    best = load_best()
    print(f"  best so far: {best['score']:.4f}" if best["score"] != float("-inf") else "  best so far: none")

    if new_score > best["score"]:
        print(f"  >> NEW BEST (improved by {new_score - best['score']:.4f})")
        save_best({
            "score": new_score,
            "iteration": iteration,
            "checkpoint_id": checkpoint_id,
        })
        git("add", "experiment.py", "experiments/best.json")
        git("commit", "-m", f"experiment {iteration}: score {new_score:.4f} (keep)")
        append_log(iteration, new_score, kept=True, checkpoint_id=checkpoint_id)
        git("add", "program.md")
        git("commit", "-m", f"log: iteration {iteration} kept")
    else:
        print(f"  >> REVERTED (no improvement)")
        git("checkout", "--", "experiment.py")
        append_log(iteration, new_score, kept=False, checkpoint_id=checkpoint_id)
        git("add", "program.md")
        git("commit", "-m", f"log: iteration {iteration} reverted")


def main() -> None:
    parser = argparse.ArgumentParser(description="Distillation autoresearch runner")
    parser.add_argument("--iterations", type=int, default=None, help="Number of iterations")
    args = parser.parse_args()

    # Ensure experiment.py is importable
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    config = load_config()
    iterations = args.iterations or config.get("runner", {}).get("iterations", 5)

    print(f"Starting distillation autoresearch loop ({iterations} iterations)")
    print(f"Teacher: {config['teacher']['model']}")
    print(f"Students: {[s['model'] for s in config['students']]}")

    for i in range(1, iterations + 1):
        run_one_iteration(i, config)

    best = load_best()
    print(f"\n{'='*50}")
    print(f"Done. Best score: {best['score']:.4f} (iteration {best['iteration']})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
