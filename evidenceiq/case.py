from __future__ import annotations

from collections import Counter, defaultdict, deque

from evidenceiq.entities import all_entity_names
from evidenceiq.models import EvidenceItem, RiskSignal


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

    def _best_node(self, query: str) -> str | None:
        q = query.lower()
        for node in self.graph:
            if q == node.lower():
                return node
        for node in self.graph:
            if q in node.lower():
                return node
        return None
