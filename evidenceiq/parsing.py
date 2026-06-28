from __future__ import annotations

import csv
import hashlib
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
