from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

from evidenceiq.models import Citation, SearchResult


DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
SAFE_GROQ_MODELS = (
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
)


class LLMClient(Protocol):
    model: str

    def answer(self, question: str, evidence: list[SearchResult]) -> str | None:
        ...

    def memo(self, case_name: str, evidence: list[Citation], risks: list[str], timeline: list[str]) -> str | None:
        ...


@dataclass(frozen=True)
class PromptPayload:
    system: str
    user: str


class GroqLLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 12.0):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        requested_model = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self.model = requested_model if requested_model in SAFE_GROQ_MODELS else DEFAULT_GROQ_MODEL
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def answer(self, question: str, evidence: list[SearchResult]) -> str | None:
        if not self.is_configured:
            return None
        payload = build_answer_prompt(question, evidence)
        return self._complete(payload)

    def memo(self, case_name: str, evidence: list[Citation], risks: list[str], timeline: list[str]) -> str | None:
        if not self.is_configured:
            return None
        payload = build_memo_prompt(case_name, evidence, risks, timeline)
        return self._complete(payload)

    def _complete(self, payload: PromptPayload) -> str | None:
        try:
            from groq import Groq

            client = Groq(api_key=self.api_key, timeout=self.timeout)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": payload.system},
                    {"role": "user", "content": payload.user},
                ],
                temperature=0.1,
            )
            content = response.choices[0].message.content
        except Exception:
            return None
        if not content:
            return None
        return content.strip()


def build_answer_prompt(question: str, evidence: list[SearchResult]) -> PromptPayload:
    snippets = "\n".join(f"[{result.evidence.id}] {result.excerpt}" for result in evidence)
    return PromptPayload(
        system=(
            "You are EvidenceIQ, an investigation copilot. Answer only from the provided evidence snippets. "
            "Do not infer guilt, fraud, criminality, illegality, or legal liability. "
            "Every factual claim must include one or more citation IDs in square brackets."
        ),
        user=f"Question: {question}\n\nRetrieved evidence snippets:\n{snippets}\n\nCited answer:",
    )


def build_memo_prompt(case_name: str, evidence: list[Citation], risks: list[str], timeline: list[str]) -> PromptPayload:
    snippets = "\n".join(f"[{citation.evidence_id}] {citation.excerpt}" for citation in evidence)
    risk_lines = "\n".join(risks) or "- No configured risk signals found."
    timeline_lines = "\n".join(timeline) or "- No known-date timeline events found."
    return PromptPayload(
        system=(
            "You are EvidenceIQ, an investigation memo assistant. Write concise markdown sections: "
            "Executive Summary, Key Findings, Evidence Gaps, Next Steps, Limitations. "
            "Use only the provided evidence snippets, risk lines, and timeline lines. "
            "Every finding must include citation IDs in square brackets. "
            "Do not declare guilt, fraud, criminality, illegality, or legal liability."
        ),
        user=(
            f"Case name: {case_name}\n\nEvidence snippets:\n{snippets}\n\n"
            f"Risk lines:\n{risk_lines}\n\nTimeline lines:\n{timeline_lines}\n\nMemo:"
        ),
    )


def validate_cited_text(text: str | None, allowed_ids: set[str]) -> str | None:
    if not text or not allowed_ids:
        return None
    cited_ids = _extract_citation_ids(text)
    if not cited_ids:
        return None
    if not cited_ids.issubset(allowed_ids):
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and not _extract_citation_ids(stripped):
            return None
    return text.strip()


def _extract_citation_ids(text: str) -> set[str]:
    ids = set()
    for bracketed in re.findall(r"\[([^\]]+)\]", text):
        ids.update(part.strip() for part in re.split(r"[,;\s]+", bracketed) if part.strip())
    return ids
