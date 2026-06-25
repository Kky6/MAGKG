from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import MAGKGSchema


@dataclass
class Entity:
    mention: str
    entity_type: str
    canonical: str
    evidence: dict[str, Any]


@dataclass
class Relation:
    head: str
    relation: str
    tail: str
    evidence: dict[str, Any]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_name(text: str, aliases: dict[str, str]) -> str:
    key = re.sub(r"\s+", " ", text.strip())
    return aliases.get(key, key)


def evidence_record(row: dict[str, Any], evidence_text: str) -> dict[str, Any]:
    """Build a compact provenance record for node or edge evidence."""
    record = {
        "document_id": str(row.get("document_id", "")),
        "paragraph_id": str(row.get("paragraph_id", "")),
        "chunk_id": str(row.get("chunk_id", row.get("id", ""))),
        "sentence_ids": list(row.get("sentence_ids", [])),
        "section": str(row.get("section", "")),
        "text": evidence_text,
    }
    if row.get("paragraph_text"):
        record["paragraph_text"] = str(row["paragraph_text"])
    if row.get("document_title"):
        record["document_title"] = str(row["document_title"])
    return record


def collect_entities(row: dict[str, Any], schema: MAGKGSchema) -> list[Entity]:
    aliases = row.get("aliases", {})
    entities: list[Entity] = []
    for item in row.get("entities", []):
        entity_type = str(item["type"])
        schema.validate_entity_type(entity_type)
        mention = str(item["mention"])
        entities.append(
            Entity(
                mention=mention,
                entity_type=entity_type,
                canonical=normalize_name(str(item.get("canonical", mention)), aliases),
                evidence=evidence_record(row, str(item.get("evidence", row["text"]))),
            )
        )
    return entities


def collect_relations(row: dict[str, Any], entities: list[Entity], schema: MAGKGSchema) -> list[Relation]:
    by_canonical = {entity.canonical: entity for entity in entities}
    relations: list[Relation] = []
    for item in row.get("relations", []):
        head = normalize_name(str(item["head"]), row.get("aliases", {}))
        tail = normalize_name(str(item["tail"]), row.get("aliases", {}))
        relation = str(item["relation"])
        if head not in by_canonical or tail not in by_canonical:
            raise ValueError(f"Relation references unknown entities: {head}, {tail}")
        schema.validate_relation(
            relation,
            by_canonical[head].entity_type,
            by_canonical[tail].entity_type,
        )
        relations.append(
            Relation(
                head=head,
                relation=relation,
                tail=tail,
                evidence=evidence_record(row, str(item.get("evidence", row["text"]))),
            )
        )
    return relations


def build_graph(rows: list[dict[str, Any]], schema: MAGKGSchema) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in rows:
        row_entities = collect_entities(row, schema)
        row_relations = collect_relations(row, row_entities, schema)

        for entity in row_entities:
            node = nodes.setdefault(
                entity.canonical,
                {
                    "id": entity.canonical,
                    "name": entity.canonical,
                    "type": entity.entity_type,
                    "level1": schema.level1(entity.entity_type),
                    "aliases": set(),
                    "evidence": [],
                },
            )
            node["aliases"].add(entity.mention)
            node["evidence"].append(entity.evidence)

        for relation in row_relations:
            edge_key = (relation.head, relation.relation, relation.tail)
            edge = edges.setdefault(
                edge_key,
                {
                    "source": relation.head,
                    "relation": relation.relation,
                    "target": relation.tail,
                    "evidence": [],
                },
            )
            edge["evidence"].append(relation.evidence)

    node_list = []
    for node in nodes.values():
        node = dict(node)
        node["aliases"] = sorted(node["aliases"])
        node_list.append(node)

    return {
        "nodes": sorted(node_list, key=lambda item: item["id"]),
        "edges": sorted(edges.values(), key=lambda item: (item["source"], item["relation"], item["target"])),
        "stats": {"nodes": len(nodes), "edges": len(edges)},
    }


def run(input_path: str | Path, schema_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    schema = MAGKGSchema.from_file(schema_path)
    rows = read_jsonl(input_path)
    graph = build_graph(rows, schema)
    write_json(output_path, graph)
    return graph
