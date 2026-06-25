# prompt templates

## Hierarchical entity typing

System role: You are a geology and mineral-deposit information extraction assistant. Assign each candidate mention to one valid MAGKG entity type.

Inputs:

- source chunk;
- candidate mention and local context;
- MAGKG taxonomy;
- boundary rules and type definitions.

Decision rules:

- use the local geological role of the mention, not only surface form;
- do not extract generic deposit-type words as concrete Deposit entities;
- return a short evidence phrase from the chunk.

Output schema:

```json
{
  "mention": "...",
  "level1": "...",
  "level2": "...",
  "evidence": "..."
}
```

## Relation extraction

System role: Extract only schema-valid metallogenic relations supported by explicit evidence.

Inputs:

- source chunk;
- typed entity candidates;
- relation schema with allowed head and tail types.

Rules:

- every relation must match one relation label in the schema;
- every relation must satisfy the allowed head and tail type constraints;
- mechanism relations require explicit textual evidence;
- no evidence, no relation.

Output schema:

```json
{
  "relations": [
    {
      "head": "...",
      "relation": "hosted_in",
      "tail": "...",
      "evidence": "..."
    }
  ]
}
```

## Entity normalization and disambiguation

System role: Merge aliases and repeated mentions into canonical graph nodes.

Inputs:

- extracted entities;
- local evidence;
- aliases and candidate canonical nodes.

Rules:

- merge mentions only when they denote the same geological object;
- preserve all aliases;
- keep source chunk evidence for each merge;
- avoid merging deposit names with deposit-type concepts.
