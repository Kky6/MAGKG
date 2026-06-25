from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_step(command: list[str]) -> None:
    print("\n$ " + " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    run_step([PYTHON, "scripts/validate_schema.py", "--schema", "configs/magkg_schema.json"])
    run_step([
        PYTHON,
        "scripts/curriculum_boundary_demo.py",
        "--base",
        "data/synthetic/synthetic_boundary_sample.jsonl",
        "--pseudo",
        "data/sample_pseudo_pool.jsonl",
        "--output",
        "outputs/train_manifest_round1.jsonl",
    ])
    run_step([
        PYTHON,
        "scripts/run_demo.py",
        "--input",
        "data/kg/kg_trace_chunks.jsonl",
        "--schema",
        "configs/magkg_schema.json",
        "--output",
        "outputs/demo_graph.json",
    ])
    run_step([
        PYTHON,
        "scripts/export_evidence_table.py",
        "--graph",
        "outputs/demo_graph.json",
        "--output",
        "outputs/evidence_trace.jsonl",
    ])
    print("\nMAGKG local workflow completed.")


if __name__ == "__main__":
    main()
