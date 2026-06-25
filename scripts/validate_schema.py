from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from magkg.schema import MAGKGSchema


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the MAGKG entity hierarchy and relation schema.")
    parser.add_argument("--schema", type=Path, default=PROJECT_ROOT / "configs" / "magkg_schema.json")
    args = parser.parse_args()

    payload = load_json(args.schema)
    schema = MAGKGSchema(payload)

    errors: list[str] = []
    if not schema.allowed_entity_types:
        errors.append("No entity types were loaded from entity_schema.")
    if not schema.relations:
        errors.append("No relation definitions were loaded from relation_schema.")

    for relation, spec in schema.relations.items():
        missing_heads = sorted(spec.head_types - schema.allowed_entity_types)
        missing_tails = sorted(spec.tail_types - schema.allowed_entity_types)
        if missing_heads:
            errors.append(f"{relation}: unknown head types: {missing_heads}")
        if missing_tails:
            errors.append(f"{relation}: unknown tail types: {missing_tails}")

    coverage: dict[str, dict[str, list[str]]] = {
        entity_type: {"as_head": [], "as_tail": []}
        for entity_type in sorted(schema.allowed_entity_types)
    }
    for relation, spec in schema.relations.items():
        for entity_type in spec.head_types:
            if entity_type in coverage:
                coverage[entity_type]["as_head"].append(relation)
        for entity_type in spec.tail_types:
            if entity_type in coverage:
                coverage[entity_type]["as_tail"].append(relation)

    uncovered = [
        entity_type
        for entity_type, roles in coverage.items()
        if not roles["as_head"] and not roles["as_tail"]
    ]
    if uncovered:
        errors.append(f"Entity types without relation coverage: {uncovered}")

    summary = {
        "level1_types": len(schema.level1_to_level2),
        "level2_types": len(schema.allowed_entity_types),
        "relations": len(schema.relations),
        "uncovered_entity_types": uncovered,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
