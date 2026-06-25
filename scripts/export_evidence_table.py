from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export edge-level evidence from a MAGKG canonical graph.")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-evidence-per-edge", type=int, default=3)
    args = parser.parse_args()

    graph = json.loads(args.graph.read_text(encoding="utf-8-sig"))
    rows: list[dict[str, Any]] = []
    for edge_index, edge in enumerate(graph.get("edges", []), start=1):
        for evidence_index, evidence in enumerate(edge.get("evidence", [])[: args.max_evidence_per_edge], start=1):
            rows.append(
                {
                    "edge_id": f"E{edge_index:04d}_{evidence_index}",
                    "head": edge.get("source", ""),
                    "relation": edge.get("relation", ""),
                    "tail": edge.get("target", ""),
                    "document_id": evidence.get("document_id", ""),
                    "paragraph_id": evidence.get("paragraph_id", ""),
                    "chunk_id": evidence.get("chunk_id", ""),
                    "sentence_ids": evidence.get("sentence_ids", []),
                    "section": evidence.get("section", ""),
                    "evidence_text": evidence.get("text", ""),
                    "paragraph_text": evidence.get("paragraph_text", ""),
                }
            )

    write_jsonl(args.output, rows)
    print(json.dumps({"graph": str(args.graph), "rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
