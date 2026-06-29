from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from evidenceiq.entities import is_plausible_person_name
from evidenceiq.models import Citation, SearchResult
from pydantic import BaseModel, ConfigDict, Field, field_validator


DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
SAFE_GROQ_MODELS = (
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
)


class LLMClient(Protocol):
    model: str

    def answer(self, question: str, evidence: list[SearchResult]) -> str | None:
        ...

    def memo(self, case_name: str, evidence: list[Citation], risks: list[str], timeline: list[str]) -> str | None:
        ...

    def extract_entities(self, text: str) -> dict[str, list[str]] | None:
        ...


@dataclass(frozen=True)
class PromptPayload:
    system: str
    user: str
    json_mode: bool = False


class EntityExtractionSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    people: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    victims: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    money: list[str] = Field(default_factory=list)
    risk_terms: list[str] = Field(default_factory=list)
    forensic_concepts: list[str] = Field(default_factory=list)

    @field_validator("*", mode="before")
    @classmethod
    def _list_of_strings(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


class GroqLLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 12.0):
        self.api_key = api_key or _env_value("GROQ_API_KEY", "")
        requested_model = model or _env_value("GROQ_MODEL", DEFAULT_GROQ_MODEL)
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

    def extract_entities(self, text: str) -> dict[str, list[str]] | None:
        if not self.is_configured:
            return None
        payload = build_entity_prompt(text)
        response = self._complete(payload)
        return parse_entity_response(response)

    def _complete(self, payload: PromptPayload) -> str | None:
        try:
            from groq import Groq

            client = Groq(api_key=self.api_key, timeout=self.timeout)
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": payload.system},
                    {"role": "user", "content": payload.user},
                ],
                "temperature": 0.1,
            }
            if payload.json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
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


def build_entity_prompt(text: str) -> PromptPayload:
    return PromptPayload(
        system=(
            "Extract structured entities from investigation evidence. Return JSON only with keys: "
            "people, organizations, locations, roles, victims, dates, money, risk_terms, forensic_concepts. "
            "Use this exact schema: "
            '{"people":["Human Full Name"],"organizations":["Organization Name"],"locations":["Facility or Place"],'
            '"roles":["Role Label"],"victims":["Human Full Name"],"dates":["Date Text"],"money":["Money Text"],'
            '"risk_terms":["Exact Risk Term"],"forensic_concepts":["DNA Link"]}. '
            "People must be human full names only, not organizations, job titles, document titles, budgets, labs, or evidence types. "
            "Never classify churches, businesses, schools, or government bodies as people, even if they are mentioned in close proximity to a suspect's actions. "
            "Buildings, facilities, churches, businesses, schools, and government bodies belong in organizations or locations, never people. "
            "Forensic ideas like DNA Link, fingerprint match, blood evidence, toxicology result, or weapon type belong in forensic_concepts, never people. "
            "Ignore document titles and headers when deciding people; phrases like Upstairs Guest, Guest Room, Second Street, or Scene Note are locations/titles, not people. "
            "Do not put physical evidence, relationship labels, descriptors, or official roles in people; examples: Failed Toxin, Family Friend, The Burned, Detective Seaver. "
            "If a phrase is a role like Finance Officer, Family Friend, Detective Seaver, or Security Contractor, put it in roles, not people. "
            "If a named person is described as the victim, deceased, body found, murdered, or killed, include them in victims as well as people. "
            "risk_terms must only include exact suspicious terms present in the text."
        ),
        user=f"Evidence text:\n{text[:6000]}\n\nJSON:",
        json_mode=True,
    )


def parse_entity_response(text: str | None) -> dict[str, list[str]] | None:
    if not text:
        return None
    cleaned = text.strip()
    match = re.search(r"\{.*\}", cleaned, re.S)
    if match:
        cleaned = match.group(0)
    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    try:
        parsed = EntityExtractionSchema.model_validate(raw)
    except Exception:
        return None
    entities = {
        "people": sorted(set(parsed.people)),
        "organizations": sorted(set(parsed.organizations)),
        "locations": sorted(set(parsed.locations)),
        "roles": sorted(set(parsed.roles)),
        "victims": sorted(set(parsed.victims)),
        "dates": sorted(set(parsed.dates)),
        "money": sorted(set(parsed.money)),
        "risk_terms": sorted(set(parsed.risk_terms)),
    }
    non_people = (
        set(parsed.organizations)
        | set(parsed.locations)
        | set(parsed.roles)
        | set(parsed.forensic_concepts)
    )
    entities["people"] = [
        person
        for person in entities["people"]
        if (
            len(person.split()) >= 2
            and person not in non_people
            and is_plausible_person_name(person)
        )
    ]
    entities["victims"] = [victim for victim in entities["victims"] if len(victim.split()) >= 2]
    return entities


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


def _env_value(name: str, default: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    for env_path in (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            if key.strip() == name:
                return raw_value.strip().strip('"').strip("'")
    return default
