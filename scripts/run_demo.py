from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from magkg.workflow import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local MAGKG graph-construction demo workflow.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "kg" / "kg_trace_chunks.jsonl")
    parser.add_argument("--schema", type=Path, default=PROJECT_ROOT / "configs" / "magkg_schema.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "demo_graph.json")
    args = parser.parse_args()

    graph = run(args.input, args.schema, args.output)
    print(f"Wrote {args.output}")
    print(f"Nodes: {graph['stats']['nodes']}; edges: {graph['stats']['edges']}")


if __name__ == "__main__":
    main()
