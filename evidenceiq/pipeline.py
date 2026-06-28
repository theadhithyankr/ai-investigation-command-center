from __future__ import annotations

from pathlib import Path

from evidenceiq.case import InvestigationCase
from evidenceiq.entities import enrich_entities
from evidenceiq.parsing import load_evidence


def build_case_from_folder(path: str | Path) -> InvestigationCase:
    items = load_evidence(Path(path))
    return InvestigationCase(enrich_entities(items))
