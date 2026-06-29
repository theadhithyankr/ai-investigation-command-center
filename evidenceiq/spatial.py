from __future__ import annotations

import re

from evidenceiq.entities import leadable_people
from evidenceiq.models import EvidenceItem, MapPin


COORDINATE_RE = re.compile(
    r"^\s*(?:coordinates?|coords?|lat(?:itude)?\s*/\s*lon(?:gitude)?|lat(?:itude)?\s*,\s*lon(?:gitude)?)\s*:\s*"
    r"(?P<lat>[+-]?\d+(?:\.\d+)?)\s*,\s*(?P<lon>[+-]?\d+(?:\.\d+)?)\s*$",
    re.I,
)
LOCATION_RE = re.compile(r"^\s*location\s*:\s*(?P<label>.+?)\s*$", re.I)


def extract_map_pins(items: list[EvidenceItem]) -> list[MapPin]:
    pins = []
    for item in items:
        parsed = extract_coordinate_metadata(item.body)
        if not parsed:
            continue
        label, latitude, longitude = parsed
        pins.append(
            MapPin(
                evidence_id=item.id,
                title=item.title,
                source=item.source,
                timestamp=item.timestamp,
                location_label=label or "Explicit coordinates",
                latitude=latitude,
                longitude=longitude,
                people=tuple(leadable_people(item)),
                excerpt=_excerpt(item.body),
            )
        )
    return sorted(pins, key=lambda pin: (pin.timestamp is None, pin.timestamp or "", pin.title, pin.evidence_id))


def extract_coordinate_metadata(text: str) -> tuple[str, float, float] | None:
    location_label = ""
    for line in text.splitlines():
        location_match = LOCATION_RE.match(line)
        if location_match:
            location_label = location_match.group("label").strip()
            continue
        coordinate_match = COORDINATE_RE.match(line)
        if not coordinate_match:
            continue
        latitude = float(coordinate_match.group("lat"))
        longitude = float(coordinate_match.group("lon"))
        if -90 <= latitude <= 90 and -180 <= longitude <= 180:
            return location_label, latitude, longitude
    return None


def _excerpt(text: str, limit: int = 220) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return f"{clean[: limit - 3].rstrip()}..."
