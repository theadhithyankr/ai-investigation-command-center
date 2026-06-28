# EvidenceIQ Architecture

EvidenceIQ uses a citation-first pipeline:

1. Parse raw evidence into normalized `EvidenceItem` records.
2. Extract entities and risk terms.
3. Build an entity index and relationship graph.
4. Search evidence before answering any question.
5. Generate answers and memos only from retrieved evidence.

The current engine is dependency-light on purpose. Heavier models can be added
behind the same tool interfaces later.

## Agent Tools

- `search_evidence`: rank documents for a question.
- `get_entity_profile`: summarize an entity's evidence footprint.
- `find_relationship_path`: connect two entities through shared evidence.
- `build_timeline`: separate dated and unknown-date evidence.
- `scan_risk_signals`: produce explainable risk leads.
- `generate_cited_memo`: draft a memo with citation IDs.
