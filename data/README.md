# MAGKG Data Examples

This directory contains JSONL examples used by the executable MAGKG workflow.

- `synthetic/synthetic_boundary_sample.jsonl`: 240 synthetic boundary examples used by the curriculum manifest demo.
- `sample_pseudo_pool.jsonl`: pseudo-labeled examples for curriculum-style sample selection.
- `kg/kg_trace_chunks.jsonl`: chunk-level KG examples with entities, aliases, relations, and document/paragraph/sentence/chunk provenance.
- `kg/kg_trace_triples.jsonl`: flattened triples with evidence text, chunk text, and paragraph text.
- `kg/canonical_graph_subset.json`: canonical graph generated from `kg_trace_chunks.jsonl`.
- `kg/evidence_trace.jsonl`: edge-level evidence table exported from the canonical graph.
- `sample_chunks.jsonl`, `sample_boundary_train.jsonl`, and `sample_triples.jsonl`: compact legacy examples retained for quick format inspection.
