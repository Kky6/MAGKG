# MAGKG Local Pipeline

This document maps the released files to the main MAGKG workflow.

## 1. Schema Validation

Input:

- `configs/magkg_schema.json`

Command:

```bash
python scripts/validate_schema.py --schema configs/magkg_schema.json
```

Output:

- number of Level-1 entity groups;
- number of active Level-2 entity types;
- number of relation types;
- uncovered entity types, if any.

## 2. Curriculum Boundary Manifest

Inputs:

- `data/synthetic/synthetic_boundary_sample.jsonl`
- `data/sample_pseudo_pool.jsonl`

Command:

```bash
python scripts/curriculum_boundary_demo.py \
  --base data/synthetic/synthetic_boundary_sample.jsonl \
  --pseudo data/sample_pseudo_pool.jsonl \
  --output outputs/train_manifest_round1.jsonl
```

The script demonstrates the training-data assembly logic used by the boundary-oriented extraction stage: base synthetic examples are retained with full weight, high-confidence pseudo labels are down-weighted, and hard cases above the lower threshold are retained as LLM-checked candidates.

## 3. Optional LLM Hooks

The local pipeline does not require an API key. To connect an OpenAI-compatible LLM endpoint, copy `.env.example` to `.env` and fill:

```env
KG_API_BASE=https://your-llm-endpoint.example.com/v1
KG_API_KEY=your_api_key_here
KG_MODEL=your-model-name
```

The same configuration can be used by downstream scripts for hierarchical type assignment, schema-constrained relation extraction, normalization, or disambiguation experiments.

## 4. Canonical Graph Construction

Input:

- `data/kg/kg_trace_chunks.jsonl`

Command:

```bash
python scripts/run_demo.py \
  --input data/kg/kg_trace_chunks.jsonl \
  --schema configs/magkg_schema.json \
  --output outputs/demo_graph.json
```

The workflow validates each relation against the MAGKG schema, normalizes aliases to canonical names, merges repeated nodes and edges, and preserves source evidence for both nodes and edges.

## 5. Evidence Table Export

Input:

- `outputs/demo_graph.json`

Command:

```bash
python scripts/export_evidence_table.py \
  --graph outputs/demo_graph.json \
  --output outputs/evidence_trace.jsonl
```

The exported table flattens graph edges into inspectable evidence rows containing the triple, evidence text, document ID, paragraph ID, chunk ID, sentence IDs, section, and paragraph text.

## 6. One-Command Run

```bash
python scripts/run_all.py
```

This command executes all four runnable stages in sequence.
