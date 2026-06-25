from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny curriculum-style boundary training manifest.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--pseudo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--easy-threshold", type=float, default=0.90)
    parser.add_argument("--hard-threshold", type=float, default=0.65)
    args = parser.parse_args()

    base_rows = read_jsonl(args.base)
    pseudo_rows = read_jsonl(args.pseudo)

    manifest: list[dict[str, Any]] = []
    for row in base_rows:
        item = dict(row)
        item["sample_weight"] = 1.0
        item["source_group"] = "gold"
        manifest.append(item)

    for row in pseudo_rows:
        confidence = float(row.get("confidence", 0.0))
        if confidence >= args.easy_threshold:
            item = dict(row)
            item["sample_weight"] = 0.4
            item["source_group"] = "easy_pseudo"
            manifest.append(item)
        elif confidence >= args.hard_threshold:
            item = dict(row)
            item["sample_weight"] = 0.7
            item["source_group"] = "hard_llm_checked"
            manifest.append(item)

    write_jsonl(args.output, manifest)
    print(json.dumps({"base_rows": len(base_rows), "pseudo_rows": len(pseudo_rows), "manifest_rows": len(manifest), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
