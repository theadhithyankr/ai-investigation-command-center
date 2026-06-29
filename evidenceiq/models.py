from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class CaseRecord:
    id: str
    name: str
    case_type: str
    description: str
    created_at: datetime
    updated_at: datetime
    llm_enabled: bool = False


@dataclass(frozen=True)
class Citation:
    evidence_id: str
    title: str
    source: str
    excerpt: str


@dataclass
class EvidenceItem:
    id: str
    source: str
    source_type: str
    title: str
    body: str
    timestamp: datetime | None
    sender: str | None = None
    recipients: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    content_hash: str = ""

    def citation(self, excerpt: str | None = None) -> Citation:
        clean = " ".join((excerpt or self.body[:260]).split())
        return Citation(self.id, self.title, self.source, clean)


@dataclass(frozen=True)
class SearchResult:
    evidence: EvidenceItem
    score: float
    matched_terms: tuple[str, ...]
    excerpt: str


@dataclass(frozen=True)
class RiskSignal:
    label: str
    severity: str
    score: int
    reason: str
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class PersonLead:
    name: str
    priority: str
    score: int
    evidence_count: int
    risk_proximity: int
    timeline_proximity: int
    scene_proximity: int
    relationship_density: int
    unresolved_items: int
    reasons: tuple[str, ...]
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class VictimRecord:
    name: str
    evidence_count: int
    reasons: tuple[str, ...]
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class MapPin:
    evidence_id: str
    title: str
    source: str
    timestamp: datetime | None
    location_label: str
    latitude: float
    longitude: float
    people: tuple[str, ...]
    excerpt: str


@dataclass(frozen=True)
class WallNode:
    id: str
    label: str
    node_type: str
    severity: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WallEdge:
    source: str
    target: str
    relationship: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentAnswer:
    question: str
    answer: str
    confidence: str
    citations: tuple[Citation, ...]
    mode: str = "local"
    fallback_reason: str | None = None


@dataclass(frozen=True)
class MemoResult:
    memo: str
    mode: str
    fallback_reason: str | None = None
