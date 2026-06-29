from __future__ import annotations

import re
from typing import Any

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
MONEY_RE = re.compile(r"\$ ?\d[\d,]*(?:\.\d{2})?|\b\d+(?:\.\d+)? ?(?:million|billion|crore|lakh)\b", re.I)
DATE_RE = re.compile(r"\b(?:20\d{2}|19\d{2})[-/]\d{1,2}[-/]\d{1,2}\b")
PERSON_SEQUENCE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")
ORG_RE = re.compile(r"\b[A-Z][A-Za-z&.\- ]+ (?:Inc|LLC|Ltd|Limited|Corp|Corporation|Bank|Partners|Energy)\b")
SOURCE_PERSON_RE = re.compile(r"^\s*source/person\s*:\s*(?P<name>.+?)\s*$", re.I)
TAGS_RE = re.compile(r"^\s*tags\s*:\s*(?P<tags>.+?)\s*$", re.I)
NARRATIVE_PREFIX_TOKENS = {"found", "failed", "saw", "seen", "met", "told", "asked", "reported", "noted"}
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
ENTITY_KEYS = ("people", "organizations", "emails", "money", "dates", "risk_terms", "locations", "roles", "victims")
PERSON_BLOCKING_TYPES = {"ORGANIZATION", "FACILITY", "ROLE"}
NON_PERSON_NAME_TOKENS = {
    "access",
    "acid",
    "analyst",
    "aunt",
    "bedroom",
    "biologics",
    "blood",
    "bloody",
    "burned",
    "budget",
    "case",
    "church",
    "city",
    "company",
    "cousin",
    "courthouse",
    "daughter",
    "detective",
    "desk",
    "document",
    "dispute",
    "email",
    "entry",
    "entrance",
    "evidence",
    "failed",
    "family",
    "father",
    "finance",
    "financial",
    "friend",
    "forensic",
    "found",
    "guest",
    "hall",
    "hospital",
    "husband",
    "internal",
    "interview",
    "lab",
    "laboratory",
    "loading",
    "log",
    "manager",
    "manual",
    "meridian",
    "mother",
    "murder",
    "note",
    "office",
    "officer",
    "poison",
    "poisoning",
    "police",
    "property",
    "report",
    "research",
    "room",
    "scene",
    "security",
    "school",
    "sister",
    "son",
    "statement",
    "street",
    "suspect",
    "the",
    "timeline",
    "toxin",
    "unknown",
    "upstairs",
    "vendor",
    "victim",
    "witness",
    "wife",
}
NON_PERSON_PREFIX_TOKENS = {
    "captain",
    "chief",
    "detective",
    "doctor",
    "dr",
    "inspector",
    "lieutenant",
    "lt",
    "officer",
    "professor",
    "sergeant",
    "sgt",
}


def extract_entities(text: str) -> dict[str, list[str]]:
    people = sorted(_extract_local_people(text))
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
        "locations": [],
        "roles": [],
    }


def enrich_entities(items, llm_client: Any | None = None):
    for item in items:
        combined = "\n".join([item.sender or "", " ".join(item.recipients), item.body])
        local_entities = extract_entities(combined)
        llm_entities = _extract_with_llm(llm_client, combined)
        item.entities = merge_entities(local_entities, llm_entities)
    return items


def all_entity_names(item) -> set[str]:
    names: set[str] = set()
    names.update(leadable_people(item))
    for key in ("organizations", "emails", "locations"):
        names.update(item.entities.get(key, []))
    if item.sender:
        names.add(item.sender)
    names.update(item.recipients)
    return names


def merge_entities(local_entities: dict[str, list[str]], llm_entities: dict[str, list[str]] | None) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for key in ENTITY_KEYS:
        values = set(local_entities.get(key, []))
        if llm_entities:
            values.update(llm_entities.get(key, []))
        merged[key] = sorted(value for value in values if value)
    if llm_entities and llm_entities.get("people"):
        # Trust structured extraction for people. Regex extraction is intentionally broad and can catch document titles.
        merged["people"] = sorted(set(llm_entities.get("people", [])))
    non_person_values = set(merged.get("organizations", [])) | set(merged.get("roles", [])) | set(merged.get("locations", []))
    merged["people"] = [
        person
        for person in merged.get("people", [])
        if person not in non_person_values and is_plausible_person_name(person)
    ]
    return merged


def leadable_people(item) -> list[str]:
    title_phrases = _candidate_phrases(item.title)
    people: set[str] = set()
    for person in item.entities.get("people", []):
        clean = _clean_person_candidate(person)
        if _is_leadable_person_candidate(clean, item, title_phrases, require_body_or_source=False):
            people.add(clean)
    for person in _extract_source_people(item.body):
        if _is_leadable_person_candidate(person, item, title_phrases, require_body_or_source=False):
            people.add(person)
    for person in _extract_narrative_people(item.body):
        if _is_leadable_person_candidate(person, item, title_phrases, require_body_or_source=True):
            people.add(person)
    for tag in _extract_tag_values(item.body):
        clean = _clean_person_candidate(tag)
        if _is_leadable_person_candidate(clean, item, title_phrases, require_body_or_source=True):
            people.add(clean)
    return sorted(people)


def entity_classifications(item, entity: str) -> tuple[str, ...]:
    clean = _clean_person_candidate(entity)
    classifications: set[str] = set()
    if clean in {_clean_person_candidate(value) for value in item.entities.get("people", [])}:
        classifications.add("PERSON")
    if clean in {_clean_person_candidate(value) for value in item.entities.get("organizations", [])}:
        classifications.add("ORGANIZATION")
    if clean in {_clean_person_candidate(value) for value in item.entities.get("locations", [])}:
        classifications.add("FACILITY")
    if clean in {_clean_person_candidate(value) for value in item.entities.get("roles", [])}:
        classifications.add("ROLE")
    if clean in _extract_source_people(item.body) or clean in _extract_narrative_people(item.body):
        classifications.add("PERSON")
    tag_people = {_clean_person_candidate(tag) for tag in _extract_tag_values(item.body) if is_plausible_person_name(tag)}
    if clean in tag_people:
        classifications.add("PERSON")
    return tuple(sorted(classifications))


def has_strict_person_type(item, entity: str) -> bool:
    classifications = set(entity_classifications(item, entity))
    return "PERSON" in classifications and not classifications.intersection(PERSON_BLOCKING_TYPES)


def is_plausible_person_name(value: str) -> bool:
    clean = _clean_person_candidate(value)
    parts = [part.strip(".,:;()[]'\"").lower() for part in clean.split()]
    if len(parts) != 2:
        return False
    if parts[0] in NON_PERSON_PREFIX_TOKENS:
        return False
    if any(part in NON_PERSON_NAME_TOKENS for part in parts):
        return False
    if any(part.isdigit() for part in parts):
        return False
    return all(part and part[0].isalpha() for part in parts)


def _extract_local_people(text: str) -> set[str]:
    people = set(_extract_source_people(text))
    people.update(_extract_narrative_people(text))
    for tag in _extract_tag_values(text):
        clean = _clean_person_candidate(tag)
        if is_plausible_person_name(clean):
            people.add(clean)
    return people


def _extract_source_people(text: str) -> set[str]:
    people: set[str] = set()
    for line in text.splitlines():
        match = SOURCE_PERSON_RE.match(line)
        if not match:
            continue
        clean = _clean_person_candidate(match.group("name"))
        if is_plausible_person_name(clean):
            people.add(clean)
    return people


def _extract_tag_values(text: str) -> list[str]:
    tags: list[str] = []
    for line in text.splitlines():
        match = TAGS_RE.match(line)
        if match:
            tags.extend(part.strip() for part in re.split(r"[,;]", match.group("tags")) if part.strip())
    return tags


def _extract_narrative_people(text: str) -> set[str]:
    people: set[str] = set()
    for raw_line in text.splitlines():
        if SOURCE_PERSON_RE.match(raw_line) or TAGS_RE.match(raw_line) or ":" in raw_line[:24]:
            continue
        for phrase in _candidate_phrases(raw_line):
            words = phrase.split()
            if len(words) > 2 and words[0].lower() in NARRATIVE_PREFIX_TOKENS:
                phrase = " ".join(words[1:3])
            else:
                phrase = " ".join(words[:2])
            clean = _clean_person_candidate(phrase)
            if is_plausible_person_name(clean):
                people.add(clean)
    return people


def _candidate_phrases(text: str) -> set[str]:
    return {" ".join(match.group(0).split()) for match in PERSON_SEQUENCE_RE.finditer(text or "")}


def _clean_person_candidate(value: str) -> str:
    clean = re.sub(r"\s*\([^)]*\)\s*$", "", value or "").strip()
    clean = clean.strip(".,:;[]'\"")
    return " ".join(clean.split())


def _is_leadable_person_candidate(
    person: str,
    item,
    title_phrases: set[str],
    *,
    require_body_or_source: bool,
) -> bool:
    if not person or not is_plausible_person_name(person):
        return False
    if person in title_phrases or person.lower() == item.title.strip().lower():
        return False
    if person in item.entities.get("organizations", []):
        return False
    if person in item.entities.get("locations", []):
        return False
    if person in item.entities.get("roles", []):
        return False
    if not has_strict_person_type(item, person):
        return False
    if require_body_or_source:
        haystack = "\n".join([item.source or "", item.body or ""]).lower()
        if person.lower() not in haystack:
            return False
    return True


def _extract_with_llm(llm_client: Any | None, text: str) -> dict[str, list[str]] | None:
    if not llm_client or not hasattr(llm_client, "extract_entities"):
        return None
    try:
        entities = llm_client.extract_entities(text)
    except Exception:
        return None
    if not isinstance(entities, dict):
        return None
    return {
        key: [str(value).strip() for value in entities.get(key, []) if str(value).strip()]
        for key in ENTITY_KEYS
    }
