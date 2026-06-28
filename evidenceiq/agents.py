from __future__ import annotations

from evidenceiq.case import InvestigationCase
from evidenceiq.llm import LLMClient, validate_cited_text
from evidenceiq.models import AgentAnswer
from evidenceiq.search import EvidenceSearch


class InvestigationAgent:
    def __init__(self, case: InvestigationCase, llm_client: LLMClient | None = None):
        self.case = case
        self.search_engine = EvidenceSearch(case.items)
        self.llm_client = llm_client

    def answer(self, question: str) -> AgentAnswer:
        if asks_for_legal_conclusion(question):
            return AgentAnswer(
                question,
                "EvidenceIQ cannot determine guilt, fraud, criminality, or legal liability. It can only surface cited risk signals and investigation leads.",
                "none",
                (),
            )
        results = self.search_engine.search(question, limit=4)
        supported = [result for result in results if result.matched_terms or result.score >= 0.18]
        if not supported:
            return AgentAnswer(
                question,
                "No supporting evidence found. EvidenceIQ will not make an unsupported claim.",
                "none",
                (),
            )
        citations = tuple(result.evidence.citation(result.excerpt) for result in supported)
        confidence = "high" if supported[0].score >= 0.5 and len(supported) >= 2 else "medium"
        if self.llm_client:
            llm_answer = self.llm_client.answer(question, supported)
            allowed_ids = {citation.evidence_id for citation in citations}
            validated = validate_cited_text(llm_answer, allowed_ids)
            if validated:
                return AgentAnswer(question, validated, confidence, citations)
        facts = []
        for result in supported[:3]:
            facts.append(f"- {result.excerpt} [{result.evidence.id}]")
        answer = "Based only on retrieved evidence:\n" + "\n".join(facts)
        return AgentAnswer(question, answer, confidence, citations)

    def memo(self, case_name: str = "Investigation Case") -> str:
        known, unknown = self.case.timeline()
        risks = self.case.risk_signals()[:5]
        lines = [
            f"# {case_name} Investigation Memo",
            "",
            "## Executive Summary",
            f"Evidence reviewed: {len(self.case.items)} items.",
            f"Known-date timeline events: {len(known)}. Unknown-date evidence items: {len(unknown)}.",
            f"Risk signals surfaced: {len(risks)}.",
            "",
            "## Timeline Highlights",
        ]
        for item in known[:6]:
            lines.append(f"- {item.timestamp.date()}: {item.title} [{item.id}]")
        if unknown:
            lines.append(f"- Unknown date: {len(unknown)} evidence items require date review.")
        lines.extend(["", "## Risk Signals"])
        if not risks:
            lines.append("- No configured risk signals found.")
        for signal in risks:
            citation_ids = ", ".join(c.evidence_id for c in signal.citations)
            lines.append(f"- {signal.label} ({signal.severity}, {signal.score}/100): {signal.reason} [{citation_ids}]")
        lines.extend(
            [
                "",
                "## Limitations",
                "This memo supports investigation triage only. It does not determine guilt, fraud, or legal liability.",
            ]
        )
        return "\n".join(lines)

    def generate_llm_memo(self, case_name: str = "Investigation Case") -> str:
        fallback = self.memo(case_name)
        if not self.llm_client:
            return fallback
        known, _unknown = self.case.timeline()
        risks = self.case.risk_signals()[:5]
        evidence = []
        seen_ids = set()
        for signal in risks:
            for citation in signal.citations:
                if citation.evidence_id not in seen_ids:
                    seen_ids.add(citation.evidence_id)
                    evidence.append(citation)
        for item in known[:6]:
            if item.id not in seen_ids:
                seen_ids.add(item.id)
                evidence.append(item.citation())
        risk_lines = []
        for signal in risks:
            citation_ids = ", ".join(f"[{c.evidence_id}]" for c in signal.citations)
            risk_lines.append(f"- {signal.label}: {signal.reason} {citation_ids}")
        timeline_lines = [f"- {item.timestamp.date()}: {item.title} [{item.id}]" for item in known[:6]]
        llm_memo = self.llm_client.memo(case_name, evidence, risk_lines, timeline_lines)
        validated = validate_cited_text(llm_memo, {citation.evidence_id for citation in evidence})
        return validated or fallback


def asks_for_legal_conclusion(question: str) -> bool:
    lowered = question.lower()
    conclusion_terms = ("guilty", "criminal", "crime", "fraud", "liable", "illegal", "proof")
    return any(term in lowered for term in conclusion_terms)
