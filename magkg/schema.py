from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RelationSpec:
    label: str
    head_types: set[str]
    tail_types: set[str]
    category: str = ""
    rule: str = ""


class MAGKGSchema:
    """Entity hierarchy and relation constraints used by the demo workflow."""

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.level1_to_level2 = self._load_entity_hierarchy(payload)
        self.level2_to_level1 = {
            level2: level1
            for level1, labels in self.level1_to_level2.items()
            for level2 in labels
        }
        self.allowed_entity_types = set(self.level2_to_level1)
        self.relations = self._load_relations(payload)

    @classmethod
    def from_file(cls, path: str | Path) -> "MAGKGSchema":
        with Path(path).open("r", encoding="utf-8-sig") as handle:
            return cls(json.load(handle))

    @staticmethod
    def _load_entity_hierarchy(payload: dict[str, Any]) -> dict[str, list[str]]:
        hierarchy: dict[str, list[str]] = {}
        for level1, entries in payload.get("entity_schema", {}).items():
            labels: list[str] = []
            for entry in entries:
                label = entry.get("type")
                if label:
                    labels.append(str(label))
            if labels:
                hierarchy[str(level1)] = labels
        return hierarchy

    @staticmethod
    def _load_relations(payload: dict[str, Any]) -> dict[str, RelationSpec]:
        relations: dict[str, RelationSpec] = {}
        for item in payload.get("relation_schema", []):
            label = str(item["relation"])
            relations[label] = RelationSpec(
                label=label,
                head_types={str(value) for value in item.get("head_types", [])},
                tail_types={str(value) for value in item.get("tail_types", [])},
                category=str(item.get("category", "")),
                rule=str(item.get("rule", "")),
            )
        return relations

    def level1(self, level2_type: str) -> str:
        return self.level2_to_level1[level2_type]

    def validate_entity_type(self, level2_type: str) -> None:
        if level2_type not in self.allowed_entity_types:
            raise ValueError(f"Unknown MAGKG entity type: {level2_type}")

    def validate_relation(self, relation: str, head_type: str, tail_type: str) -> None:
        if relation not in self.relations:
            raise ValueError(f"Unknown MAGKG relation: {relation}")
        spec = self.relations[relation]
        if head_type not in spec.head_types or tail_type not in spec.tail_types:
            raise ValueError(
                f"Invalid relation schema: ({head_type})-[{relation}]->({tail_type})"
            )
