from __future__ import annotations

import hashlib

from evidenceiq.case import InvestigationCase
from evidenceiq.entities import all_entity_names, leadable_people
from evidenceiq.models import EvidenceItem, WallEdge, WallNode


ENTITY_NODE_TYPES = {
    "people": "person",
    "organizations": "organization",
    "emails": "email",
    "locations": "location",
}


def build_investigation_wall(case: InvestigationCase) -> tuple[list[WallNode], list[WallEdge]]:
    nodes: dict[str, WallNode] = {}
    edges: dict[tuple[str, str, str], WallEdge] = {}
    evidence_by_id = {item.id: item for item in case.items}

    for signal in case.risk_signals():
        risk_id = _risk_id(signal.label, signal.reason, [citation.evidence_id for citation in signal.citations])
        nodes[risk_id] = WallNode(
            id=risk_id,
            label=signal.label,
            node_type="risk",
            severity=signal.severity,
            metadata={"score": str(signal.score), "reason": signal.reason},
        )
        for citation in signal.citations:
            evidence = evidence_by_id.get(citation.evidence_id)
            if not evidence:
                continue
            _add_evidence_node(nodes, evidence)
            _add_edge(edges, risk_id, _evidence_id(evidence.id), "cites", {"excerpt": citation.excerpt})
            _add_entity_connections(nodes, edges, evidence)

    risk_cited_ids = {
        edge.target.replace("evidence:", "", 1)
        for edge in edges.values()
        if edge.source.startswith("risk:") and edge.target.startswith("evidence:")
    }
    for evidence in case.items:
        if evidence.id in risk_cited_ids:
            continue
        if all_entity_names(evidence):
            _add_evidence_node(nodes, evidence)
            _add_entity_connections(nodes, edges, evidence)

    return list(nodes.values()), list(edges.values())


def _add_evidence_node(nodes: dict[str, WallNode], evidence: EvidenceItem) -> None:
    node_id = _evidence_id(evidence.id)
    date_label = evidence.timestamp.date().isoformat() if evidence.timestamp else "Unknown date"
    nodes[node_id] = WallNode(
        id=node_id,
        label=evidence.title,
        node_type="evidence",
        metadata={"evidence_id": evidence.id, "source_type": evidence.source_type, "date": date_label},
    )


def _add_entity_connections(
    nodes: dict[str, WallNode],
    edges: dict[tuple[str, str, str], WallEdge],
    evidence: EvidenceItem,
) -> None:
    evidence_node_id = _evidence_id(evidence.id)
    for entity_key, node_type in ENTITY_NODE_TYPES.items():
        entities = leadable_people(evidence) if entity_key == "people" else evidence.entities.get(entity_key, [])
        for entity in entities:
            entity_id = _entity_id(node_type, entity)
            nodes.setdefault(entity_id, WallNode(id=entity_id, label=entity, node_type=node_type))
            _add_edge(edges, evidence_node_id, entity_id, "mentions")
    if evidence.sender:
        sender_id = _entity_id("email", evidence.sender)
        nodes.setdefault(sender_id, WallNode(id=sender_id, label=evidence.sender, node_type="email"))
        _add_edge(edges, evidence_node_id, sender_id, "source")
    for recipient in evidence.recipients:
        recipient_id = _entity_id("email", recipient)
        nodes.setdefault(recipient_id, WallNode(id=recipient_id, label=recipient, node_type="email"))
        _add_edge(edges, evidence_node_id, recipient_id, "recipient")


def _add_edge(
    edges: dict[tuple[str, str, str], WallEdge],
    source: str,
    target: str,
    relationship: str,
    metadata: dict[str, str] | None = None,
) -> None:
    edges.setdefault((source, target, relationship), WallEdge(source, target, relationship, metadata or {}))


def _risk_id(label: str, reason: str, evidence_ids: list[str]) -> str:
    digest = hashlib.sha256(f"{label}\n{reason}\n{','.join(sorted(evidence_ids))}".encode("utf-8")).hexdigest()
    return f"risk:{digest[:12]}"


def _evidence_id(evidence_id: str) -> str:
    return f"evidence:{evidence_id}"


def _entity_id(node_type: str, label: str) -> str:
    digest = hashlib.sha256(label.lower().encode("utf-8")).hexdigest()
    return f"{node_type}:{digest[:12]}"
