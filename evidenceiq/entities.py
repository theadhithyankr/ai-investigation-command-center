from __future__ import annotations

import re

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
MONEY_RE = re.compile(r"\$ ?\d[\d,]*(?:\.\d{2})?|\b\d+(?:\.\d+)? ?(?:million|billion|crore|lakh)\b", re.I)
DATE_RE = re.compile(r"\b(?:20\d{2}|19\d{2})[-/]\d{1,2}[-/]\d{1,2}\b")
PERSON_RE = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
ORG_RE = re.compile(r"\b[A-Z][A-Za-z&.\- ]+ (?:Inc|LLC|Ltd|Limited|Corp|Corporation|Bank|Partners|Energy)\b")
RISK_TERMS = {
    "off book",
    "side letter",
    "backdate",
    "delete",
    "personal email",
    "urgent wire",
    "do not share",
    "cash only",
    "override",
    "confidential",
}


def extract_entities(text: str) -> dict[str, list[str]]:
    people = sorted(set(PERSON_RE.findall(text)))
    orgs = sorted(set(match.strip() for match in ORG_RE.findall(text)))
    emails = sorted(set(EMAIL_RE.findall(text)))
    money = sorted(set(MONEY_RE.findall(text)))
    dates = sorted(set(DATE_RE.findall(text)))
    risks = sorted(term for term in RISK_TERMS if term in text.lower())
    return {
        "people": people,
        "organizations": orgs,
        "emails": emails,
        "money": money,
        "dates": dates,
        "risk_terms": risks,
    }


def enrich_entities(items):
    for item in items:
        combined = "\n".join([item.title, item.sender or "", " ".join(item.recipients), item.body])
        item.entities = extract_entities(combined)
    return items


def all_entity_names(item) -> set[str]:
    names: set[str] = set()
    for key in ("people", "organizations", "emails"):
        names.update(item.entities.get(key, []))
    if item.sender:
        names.add(item.sender)
    names.update(item.recipients)
    return names
