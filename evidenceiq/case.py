from __future__ import annotations

from collections import Counter, defaultdict, deque

from evidenceiq.entities import all_entity_names
from evidenceiq.models import EvidenceItem, PersonLead, RiskSignal


SCENE_PROXIMITY_TYPES = {"scene_note", "forensic_note", "phone_log", "interview", "report"}
NON_PERSON_TOKENS = {
    "access",
    "analyst",
    "biologics",
    "budget",
    "case",
    "desk",
    "document",
    "email",
    "entrance",
    "evidence",
    "finance",
    "financial",
    "forensic",
    "internal",
    "interview",
    "lab",
    "laboratory",
    "loading",
    "log",
    "manager",
    "meridian",
    "note",
    "office",
    "officer",
    "report",
    "research",
    "scene",
    "security",
    "statement",
    "timeline",
    "unknown",
    "vendor",
    "witness",
}
VICTIM_CONTEXT_TERMS = {"victim", "deceased", "death", "died", "body", "killed", "murdered"}


class InvestigationCase:
    def __init__(self, items: list[EvidenceItem]):
        self.items = items
        self.entity_index = self._build_entity_index()
        self.graph = self._build_graph()

    def _build_entity_index(self) -> dict[str, list[EvidenceItem]]:
        index: dict[str, list[EvidenceItem]] = defaultdict(list)
        for item in self.items:
            for name in all_entity_names(item):
                index[name].append(item)
        return dict(index)

    def _build_graph(self) -> dict[str, set[str]]:
        graph: dict[str, set[str]] = defaultdict(set)
        for item in self.items:
            entities = sorted(all_entity_names(item))
            for left in entities:
                for right in entities:
                    if left != right:
                        graph[left].add(right)
            graph[item.id].update(entities)
            for entity in entities:
                graph[entity].add(item.id)
        return {node: edges for node, edges in graph.items()}

    def entity_profile(self, entity: str) -> dict:
        matches = {name: docs for name, docs in self.entity_index.items() if entity.lower() in name.lower()}
        docs = [doc for group in matches.values() for doc in group]
        return {
            "query": entity,
            "matched_entities": sorted(matches),
            "evidence_count": len({doc.id for doc in docs}),
            "citations": [doc.citation() for doc in docs[:5]],
        }

    def relationship_path(self, start: str, end: str) -> list[str]:
        start_node = self._best_node(start)
        end_node = self._best_node(end)
        if not start_node or not end_node:
            return []
        queue = deque([(start_node, [start_node])])
        seen = {start_node}
        while queue:
            node, path = queue.popleft()
            if node == end_node:
                return path
            if len(path) > 5:
                continue
            for neighbor in self.graph.get(node, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return []

    def timeline(self, query: str | None = None) -> tuple[list[EvidenceItem], list[EvidenceItem]]:
        filtered = self.items
        if query:
            q = query.lower()
            filtered = [item for item in self.items if q in item.title.lower() or q in item.body.lower()]
        known = sorted([item for item in filtered if item.timestamp], key=lambda item: item.timestamp)
        unknown = [item for item in filtered if not item.timestamp]
        return known, unknown

    def risk_signals(self) -> list[RiskSignal]:
        signals: list[RiskSignal] = []
        for item in self.items:
            terms = item.entities.get("risk_terms", [])
            if terms:
                score = min(100, 25 + len(terms) * 15)
                signals.append(
                    RiskSignal(
                        "Sensitive language",
                        "high" if score >= 55 else "medium",
                        score,
                        f"Found risk terms: {', '.join(terms)}",
                        (item.citation(),),
                    )
                )
        entity_counts = Counter(entity for item in self.items for entity in all_entity_names(item))
        for entity, count in entity_counts.items():
            if count >= 3:
                docs = self.entity_index.get(entity, [])[:3]
                signals.append(
                    RiskSignal(
                        "Repeated entity concentration",
                        "medium",
                        min(100, 20 + count * 10),
                        f"{entity} appears in {count} evidence items.",
                        tuple(doc.citation() for doc in docs),
                    )
                )
        return sorted(signals, key=lambda signal: signal.score, reverse=True)

    def leadboard(self, limit: int = 10) -> list[PersonLead]:
        leads: list[PersonLead] = []
        people = sorted(
            {
                person
                for item in self.items
                for person in item.entities.get("people", [])
                if self._is_lead_candidate(person, item)
            }
        )
        for person in people:
            docs = self.entity_index.get(person, [])
            unique_docs = list({doc.id: doc for doc in docs}.values())
            if not unique_docs:
                continue
            risk_docs = [doc for doc in unique_docs if doc.entities.get("risk_terms")]
            known_docs = [doc for doc in unique_docs if doc.timestamp]
            scene_docs = [doc for doc in unique_docs if doc.source_type in SCENE_PROXIMITY_TYPES]
            unknown_docs = [doc for doc in unique_docs if not doc.timestamp]
            related_entities = set()
            for doc in unique_docs:
                related_entities.update(all_entity_names(doc))
            related_entities.discard(person)
            relationship_density = min(10, len(related_entities))
            source_types = {doc.source_type for doc in unique_docs}
            raw_score = (
                min(4, len(unique_docs)) * 8
                + min(3, len(risk_docs)) * 7
                + min(3, len(known_docs)) * 5
                + min(3, len(scene_docs)) * 6
                + relationship_density * 2
                + min(3, len(source_types)) * 3
                + min(3, len(unknown_docs)) * 2
            )
            score = min(
                95,
                raw_score,
            )
            if score >= 70:
                priority = "high"
            elif score >= 40:
                priority = "medium"
            else:
                priority = "low"
            reasons = self._lead_reasons(
                person,
                unique_docs,
                risk_docs,
                known_docs,
                scene_docs,
                related_entities,
                unknown_docs,
            )
            leads.append(
                PersonLead(
                    name=person,
                    priority=priority,
                    score=score,
                    evidence_count=len(unique_docs),
                    risk_proximity=len(risk_docs),
                    timeline_proximity=len(known_docs),
                    scene_proximity=len(scene_docs),
                    relationship_density=relationship_density,
                    unresolved_items=len(unknown_docs),
                    reasons=tuple(reasons),
                    citations=tuple(doc.citation() for doc in unique_docs[:5]),
                )
            )
        return sorted(
            leads,
            key=lambda lead: (
                lead.score,
                lead.evidence_count,
                lead.risk_proximity,
                lead.relationship_density,
            ),
            reverse=True,
        )[:limit]

    def _is_lead_candidate(self, person: str, item: EvidenceItem) -> bool:
        clean = person.strip()
        if not clean:
            return False
        tokens = [part.lower().strip(".,:;()[]") for part in clean.split()]
        if len(tokens) != 2:
            return False
        if any(token in NON_PERSON_TOKENS for token in tokens):
            return False
        if clean in item.entities.get("organizations", []):
            return False
        if self._appears_as_victim(clean, item):
            return False
        return True

    def _appears_as_victim(self, person: str, item: EvidenceItem) -> bool:
        haystack = "\n".join([item.title, item.source, item.body])
        for sentence in haystack.replace("\n", ". ").split("."):
            lowered = sentence.lower()
            if person.lower() in lowered and any(term in lowered for term in VICTIM_CONTEXT_TERMS):
                return True
        return False

    def _lead_reasons(
        self,
        person: str,
        docs: list[EvidenceItem],
        risk_docs: list[EvidenceItem],
        known_docs: list[EvidenceItem],
        scene_docs: list[EvidenceItem],
        related_entities: set[str],
        unknown_docs: list[EvidenceItem],
    ) -> list[str]:
        reasons = [f"{person} appears in {len(docs)} cited evidence item(s)."]
        if risk_docs:
            terms = sorted({term for doc in risk_docs for term in doc.entities.get("risk_terms", [])})
            reasons.append(f"Appears near configured risk terms: {', '.join(terms)}.")
        if known_docs:
            reasons.append(f"Has {len(known_docs)} known-date timeline overlap(s).")
        if scene_docs:
            types = sorted({doc.source_type for doc in scene_docs})
            reasons.append(f"Appears in scene-adjacent evidence types: {', '.join(types)}.")
        if related_entities:
            reasons.append(f"Connected to {len(related_entities)} other extracted entity/entities.")
        if unknown_docs:
            reasons.append(f"{len(unknown_docs)} linked item(s) have unknown dates and need review.")
        reasons.append("Priority is an investigative triage score, not a guilt or liability finding.")
        return reasons

    def _best_node(self, query: str) -> str | None:
        q = query.lower()
        for node in self.graph:
            if q == node.lower():
                return node
        for node in self.graph:
            if q in node.lower():
                return node
        return None
