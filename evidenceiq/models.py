from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


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
class AgentAnswer:
    question: str
    answer: str
    confidence: str
    citations: tuple[Citation, ...]
