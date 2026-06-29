from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime
from email import policy
from email.parser import Parser
from pathlib import Path
from tempfile import NamedTemporaryFile

from evidenceiq.models import EvidenceItem

DATE_PATTERNS = [
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%a, %d %b %Y %H:%M:%S %z",
    "%d %b %Y %H:%M:%S %z",
]


def stable_id(source: str, body: str) -> str:
    digest = hashlib.sha256(f"{source}\n{body}".encode("utf-8", errors="ignore")).hexdigest()
    return f"ev-{digest[:12]}"


def content_hash(body: str) -> str:
    normalized = " ".join(body.lower().split())
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    cleaned = value.strip()
    for pattern in DATE_PATTERNS:
        try:
            parsed = datetime.strptime(cleaned, pattern)
            return parsed.replace(tzinfo=None)
        except ValueError:
            continue
    match = re.search(r"\b(20\d{2}|19\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", cleaned)
    if match:
        year, month, day = map(int, match.groups())
        try:
            return datetime(year, month, day)
        except ValueError:
            return None
    return None


def parse_text_file(path: Path) -> EvidenceItem:
    text = path.read_text(encoding="utf-8", errors="ignore")
    parsed = Parser(policy=policy.default).parsestr(text)
    has_headers = bool(parsed["from"] or parsed["to"] or parsed["date"] or parsed["subject"])
    body = parsed.get_body(preferencelist=("plain",))
    body_text = body.get_content() if body else (parsed.get_payload() if has_headers else text)
    title = str(parsed["subject"] or path.stem).strip()
    sender = str(parsed["from"]).strip() if parsed["from"] else None
    recipients = _split_recipients(str(parsed["to"] or ""))
    timestamp = parse_date(str(parsed["date"] or "")) or parse_date(text[:500])
    item = EvidenceItem(
        id=stable_id(str(path), body_text),
        source=str(path),
        source_type=path.suffix.lower().lstrip(".") or "txt",
        title=title,
        body=body_text.strip(),
        timestamp=timestamp,
        sender=sender,
        recipients=recipients,
    )
    item.content_hash = content_hash(item.body)
    return item


def parse_csv_file(path: Path) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            body = "\n".join(f"{key}: {value}" for key, value in row.items() if value)
            title = row.get("subject") or row.get("title") or f"{path.stem} row {index}"
            timestamp = parse_date(row.get("date") or row.get("timestamp"))
            item = EvidenceItem(
                id=stable_id(f"{path}:{index}", body),
                source=f"{path}:{index}",
                source_type="csv",
                title=title,
                body=body,
                timestamp=timestamp,
                sender=row.get("from") or row.get("sender"),
                recipients=_split_recipients(row.get("to") or row.get("recipients") or ""),
            )
            item.content_hash = content_hash(item.body)
            items.append(item)
    return items


def load_evidence(path: Path) -> list[EvidenceItem]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        items: list[EvidenceItem] = []
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in {".txt", ".eml", ".csv", ".md", ".pdf"}:
                items.extend(load_evidence(child))
        return deduplicate(items)
    if path.suffix.lower() == ".csv":
        return parse_csv_file(path)
    if path.suffix.lower() in {".txt", ".eml", ".md"}:
        return [parse_text_file(path)]
    if path.suffix.lower() == ".pdf":
        return [parse_pdf_file(path)]
    raise ValueError(f"Unsupported evidence type: {path.suffix}")


def parse_uploaded_bytes(name: str, data: bytes) -> list[EvidenceItem]:
    suffix = Path(name).suffix.lower() or ".txt"
    with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(data)
        temp_path = Path(handle.name)
    try:
        return load_evidence(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def create_manual_evidence(
    title: str,
    evidence_type: str,
    body: str,
    *,
    date_value: str | None = None,
    source_person: str | None = None,
    tags: list[str] | None = None,
    location: str | None = None,
    latitude: float | str | None = None,
    longitude: float | str | None = None,
) -> EvidenceItem:
    clean_title = title.strip() or "Untitled note"
    clean_body = body.strip()
    clean_source = source_person.strip() if source_person else "Manual entry"
    clean_tags = [tag.strip() for tag in tags or [] if tag.strip()]
    metadata = [
        f"Type: {evidence_type}",
        f"Source/person: {clean_source}",
    ]
    if clean_tags:
        metadata.append(f"Tags: {', '.join(clean_tags)}")
    clean_location = location.strip() if location else ""
    if clean_location:
        metadata.append(f"Location: {clean_location}")
    coordinates = _format_coordinates(latitude, longitude)
    if coordinates:
        metadata.append(f"Coordinates: {coordinates}")
    full_body = "\n".join(metadata + ["", clean_body]).strip()
    item = EvidenceItem(
        id=stable_id(f"manual:{clean_title}:{clean_source}", full_body),
        source=clean_source,
        source_type=evidence_type.strip() or "manual_note",
        title=clean_title,
        body=full_body,
        timestamp=parse_date(date_value),
        sender=clean_source if clean_source != "Manual entry" else None,
    )
    item.content_hash = content_hash(item.body)
    return item


def parse_manual_note_payload(text: str) -> dict[str, object]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Paste JSON or a labeled note first.")
    if stripped.startswith("{"):
        return _parse_manual_note_json(stripped)
    return _parse_labeled_manual_note(stripped)


def parse_manual_note_payloads(text: str) -> list[dict[str, object]]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Paste one or more notes first.")
    if stripped.startswith("["):
        return _parse_manual_note_json_array(stripped)
    if stripped.startswith("{"):
        return [parse_manual_note_payload(stripped)]
    parts = [part.strip() for part in re.split(r"(?im)^\s*---\s*EVIDENCE ITEM\s*---\s*$", stripped) if part.strip()]
    if len(parts) <= 1:
        return [parse_manual_note_payload(stripped)]
    return [parse_manual_note_payload(part) for part in parts]


def _parse_manual_note_json_array(text: str) -> list[dict[str, object]]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc.msg}") from exc
    if not isinstance(raw, list):
        raise ValueError("Bulk JSON must be an array of evidence objects.")
    payloads = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Evidence item {index} must be a JSON object.")
        payloads.append(_normalize_manual_note_dict(item))
    return payloads


def _parse_manual_note_json(text: str) -> dict[str, object]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Manual note JSON must be an object.")
    return _normalize_manual_note_dict(raw)


def _normalize_manual_note_dict(raw: dict) -> dict[str, object]:
    tags = raw.get("tags", [])
    if isinstance(tags, str):
        tags = _split_recipients(tags)
    elif not isinstance(tags, list):
        tags = []
    latitude = str(raw.get("latitude") or raw.get("lat") or "").strip()
    longitude = str(raw.get("longitude") or raw.get("lon") or raw.get("lng") or "").strip()
    if raw.get("coordinates") and (not latitude or not longitude):
        latitude, longitude = _split_coordinates(str(raw.get("coordinates") or ""))
    return {
        "title": str(raw.get("title") or "").strip(),
        "evidence_type": str(raw.get("evidence_type") or raw.get("type") or "other").strip(),
        "date": str(raw.get("date") or "").strip(),
        "unknown_date": _is_unknown_date(raw.get("date")),
        "source_person": str(raw.get("source_person") or raw.get("source") or "").strip(),
        "tags": [str(tag).strip() for tag in tags if str(tag).strip()],
        "location": str(raw.get("location") or "").strip(),
        "latitude": latitude,
        "longitude": longitude,
        "body": str(raw.get("body") or "").strip(),
    }


def _parse_labeled_manual_note(text: str) -> dict[str, object]:
    labels = {
        "title": ("title", "itle"),
        "evidence_type": ("evidence type", "type"),
        "date": ("date",),
        "source_person": ("source/person", "source", "source person"),
        "tags": ("tags",),
        "location": ("location",),
        "latitude": ("latitude", "lat"),
        "longitude": ("longitude", "lon", "lng"),
        "coordinates": ("coordinates", "coords"),
        "body": ("body",),
    }
    found: dict[str, str] = {}
    current_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        matched_key, value = _match_labeled_line(line, labels)
        if matched_key:
            current_key = matched_key
            found[current_key] = value
            continue
        if current_key:
            separator = "\n" if found[current_key] else ""
            found[current_key] = f"{found[current_key]}{separator}{line}".strip()
    tags = _split_recipients(found.get("tags", ""))
    date_value = found.get("date", "").strip()
    latitude = found.get("latitude", "").strip()
    longitude = found.get("longitude", "").strip()
    if found.get("coordinates") and (not latitude or not longitude):
        latitude, longitude = _split_coordinates(found.get("coordinates", ""))
    return {
        "title": found.get("title", "").strip(),
        "evidence_type": found.get("evidence_type", "other").strip() or "other",
        "date": date_value,
        "unknown_date": _is_unknown_date(date_value),
        "source_person": found.get("source_person", "").strip(),
        "tags": tags,
        "location": found.get("location", "").strip(),
        "latitude": latitude,
        "longitude": longitude,
        "body": found.get("body", "").strip(),
    }


def _match_labeled_line(line: str, labels: dict[str, tuple[str, ...]]) -> tuple[str | None, str]:
    for key, aliases in labels.items():
        for alias in aliases:
            prefix = f"{alias}:"
            if line.lower().startswith(prefix):
                return key, line[len(prefix) :].strip()
    return None, ""


def _is_unknown_date(value: object) -> bool:
    if value is None:
        return True
    cleaned = str(value).strip().lower()
    return not cleaned or cleaned in {"unknown", "unknown date", "n/a", "na", "none"}


def _split_coordinates(value: str) -> tuple[str, str]:
    parts = [part.strip() for part in value.split(",", 1)]
    if len(parts) != 2:
        return "", ""
    return parts[0], parts[1]


def _format_coordinates(latitude: float | str | None, longitude: float | str | None) -> str:
    if latitude in (None, "") or longitude in (None, ""):
        return ""
    try:
        lat_value = float(latitude)
        lon_value = float(longitude)
    except (TypeError, ValueError):
        return ""
    if not (-90 <= lat_value <= 90 and -180 <= lon_value <= 180):
        return ""
    return f"{lat_value:.6f}, {lon_value:.6f}".rstrip("0").rstrip(".")


def parse_pdf_file(path: Path) -> EvidenceItem:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise ValueError("PDF parsing requires optional dependency: pip install pypdf") from exc
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    item = EvidenceItem(
        id=stable_id(str(path), text),
        source=str(path),
        source_type="pdf",
        title=path.stem,
        body=text.strip(),
        timestamp=parse_date(text[:1000]),
    )
    item.content_hash = content_hash(item.body)
    return item


def deduplicate(items: list[EvidenceItem]) -> list[EvidenceItem]:
    seen: set[str] = set()
    unique: list[EvidenceItem] = []
    for item in items:
        if item.content_hash not in seen:
            seen.add(item.content_hash)
            unique.append(item)
    return unique


def _split_recipients(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,;]", value or "") if part.strip()]
