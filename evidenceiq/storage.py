from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from evidenceiq.entities import enrich_entities
from evidenceiq.models import CaseRecord, EvidenceItem
from evidenceiq.parsing import load_evidence


SAMPLE_CASE_ID = "case-aster-bridge-sample"

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    case_type TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    llm_enabled INTEGER NOT NULL DEFAULT 0,
    archived_at TEXT
);

CREATE TABLE IF NOT EXISTS evidence (
    id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    source TEXT NOT NULL,
    source_type TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    timestamp TEXT,
    sender TEXT,
    recipients TEXT NOT NULL,
    entities TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    PRIMARY KEY(case_id, id),
    FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE,
    UNIQUE(case_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_evidence_case_id ON evidence(case_id);
CREATE INDEX IF NOT EXISTS idx_evidence_case_time ON evidence(case_id, timestamp);
"""


class EvidenceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init(self) -> None:
        conn = self._connect()
        try:
            self._migrate_if_needed(conn)
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _migrate_if_needed(self, conn: sqlite3.Connection) -> None:
        evidence_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'evidence'"
        ).fetchone()
        if not evidence_exists:
            return
        table_info = conn.execute("PRAGMA table_info(evidence)").fetchall()
        columns = {row[1] for row in table_info}
        primary_key_columns = {row[1]: row[5] for row in table_info if row[5]}
        if "case_id" in columns and primary_key_columns.get("case_id") and primary_key_columns.get("id"):
            return
        if "case_id" in columns:
            conn.execute("ALTER TABLE evidence RENAME TO evidence_legacy")
            conn.executescript(SCHEMA)
            legacy_rows = conn.execute(
                """
                SELECT id, case_id, source, source_type, title, body, timestamp, sender, recipients, entities, content_hash
                FROM evidence_legacy
                """
            ).fetchall()
            for row in legacy_rows:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO evidence
                    (id, case_id, source, source_type, title, body, timestamp, sender, recipients, entities, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    row,
                )
            conn.execute("DROP TABLE evidence_legacy")
            conn.commit()
            return

        now = _utc_now()
        conn.execute("ALTER TABLE evidence RENAME TO evidence_legacy")
        conn.executescript(SCHEMA)
        conn.execute(
            """
            INSERT OR IGNORE INTO cases
            (id, name, case_type, description, created_at, updated_at, llm_enabled)
            VALUES (?, ?, ?, ?, ?, ?, 0)
            """,
            (
                SAMPLE_CASE_ID,
                "Aster Bridge Sample",
                "demo",
                "Built-in demo case seeded from data/sample_case.",
                now,
                now,
            ),
        )
        legacy_rows = conn.execute(
            """
            SELECT id, source, source_type, title, body, timestamp, sender, recipients, entities, content_hash
            FROM evidence_legacy
            """
        ).fetchall()
        for row in legacy_rows:
            conn.execute(
                """
                INSERT OR IGNORE INTO evidence
                (id, case_id, source, source_type, title, body, timestamp, sender, recipients, entities, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (row[0], SAMPLE_CASE_ID, row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]),
            )
        conn.execute("DROP TABLE evidence_legacy")
        conn.commit()

    def create_case(
        self,
        name: str,
        case_type: str = "custom",
        description: str = "",
        llm_enabled: bool = False,
        case_id: str | None = None,
    ) -> CaseRecord:
        now = _utc_now()
        record = CaseRecord(
            id=case_id or f"case-{uuid.uuid4().hex[:12]}",
            name=name.strip() or "Untitled Case",
            case_type=case_type.strip() or "custom",
            description=description.strip(),
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
            llm_enabled=llm_enabled,
        )
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO cases
                (id, name, case_type, description, created_at, updated_at, llm_enabled)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.name,
                    record.case_type,
                    record.description,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    int(record.llm_enabled),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return record

    def list_cases(self, include_archived: bool = False) -> list[CaseRecord]:
        conn = self._connect()
        try:
            where = "" if include_archived else "WHERE archived_at IS NULL"
            rows = conn.execute(
                f"""
                SELECT id, name, case_type, description, created_at, updated_at, llm_enabled
                FROM cases
                {where}
                ORDER BY updated_at DESC, created_at DESC
                """
            ).fetchall()
        finally:
            conn.close()
        return [_case_from_row(row) for row in rows]

    def get_case(self, case_id: str) -> CaseRecord | None:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT id, name, case_type, description, created_at, updated_at, llm_enabled
                FROM cases
                WHERE id = ? AND archived_at IS NULL
                """,
                (case_id,),
            ).fetchone()
        finally:
            conn.close()
        return _case_from_row(row) if row else None

    def update_case(
        self,
        case_id: str,
        *,
        name: str | None = None,
        case_type: str | None = None,
        description: str | None = None,
        llm_enabled: bool | None = None,
    ) -> CaseRecord | None:
        current = self.get_case(case_id)
        if not current:
            return None
        updated = {
            "name": current.name if name is None else name.strip() or current.name,
            "case_type": current.case_type if case_type is None else case_type.strip() or current.case_type,
            "description": current.description if description is None else description.strip(),
            "llm_enabled": current.llm_enabled if llm_enabled is None else llm_enabled,
            "updated_at": _utc_now(),
        }
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE cases
                SET name = ?, case_type = ?, description = ?, llm_enabled = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    updated["name"],
                    updated["case_type"],
                    updated["description"],
                    int(updated["llm_enabled"]),
                    updated["updated_at"],
                    case_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return self.get_case(case_id)

    def delete_case(self, case_id: str) -> bool:
        conn = self._connect()
        try:
            before = conn.total_changes
            conn.execute(
                "UPDATE cases SET archived_at = ?, updated_at = ? WHERE id = ?",
                (_utc_now(), _utc_now(), case_id),
            )
            conn.commit()
            return conn.total_changes > before
        finally:
            conn.close()

    def upsert_many(self, case_id: str, items: list[EvidenceItem]) -> int:
        inserted = 0
        conn = self._connect()
        try:
            for item in items:
                before = conn.total_changes
                conn.execute(
                    """
                    INSERT OR IGNORE INTO evidence
                    (id, case_id, source, source_type, title, body, timestamp, sender, recipients, entities, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.id,
                        case_id,
                        item.source,
                        item.source_type,
                        item.title,
                        item.body,
                        item.timestamp.isoformat() if item.timestamp else None,
                        item.sender,
                        json.dumps(item.recipients),
                        json.dumps(item.entities),
                        item.content_hash,
                    ),
                )
                if conn.total_changes > before:
                    inserted += 1
            if inserted:
                conn.execute("UPDATE cases SET updated_at = ? WHERE id = ?", (_utc_now(), case_id))
            conn.commit()
        finally:
            conn.close()
        return inserted

    def all(self, case_id: str) -> list[EvidenceItem]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id, source, source_type, title, body, timestamp, sender, recipients, entities, content_hash
                FROM evidence
                WHERE case_id = ?
                ORDER BY COALESCE(timestamp, '9999-12-31'), title
                """,
                (case_id,),
            ).fetchall()
        finally:
            conn.close()
        return [_evidence_from_row(row) for row in rows]

    def seed_sample_case(self, sample_path: str | Path) -> CaseRecord:
        case = self.get_case(SAMPLE_CASE_ID)
        if not case:
            case = self.create_case(
                "Aster Bridge Sample",
                case_type="demo",
                description="Built-in demo case seeded from data/sample_case.",
                llm_enabled=False,
                case_id=SAMPLE_CASE_ID,
            )
        if not self.all(case.id):
            self.upsert_many(case.id, enrich_entities(load_evidence(Path(sample_path))))
        return self.get_case(case.id) or case


def _case_from_row(row) -> CaseRecord:
    return CaseRecord(
        id=row[0],
        name=row[1],
        case_type=row[2],
        description=row[3],
        created_at=datetime.fromisoformat(row[4]),
        updated_at=datetime.fromisoformat(row[5]),
        llm_enabled=bool(row[6]),
    )


def _evidence_from_row(row) -> EvidenceItem:
    return EvidenceItem(
        id=row[0],
        source=row[1],
        source_type=row[2],
        title=row[3],
        body=row[4],
        timestamp=datetime.fromisoformat(row[5]) if row[5] else None,
        sender=row[6],
        recipients=json.loads(row[7]),
        entities=json.loads(row[8]),
        content_hash=row[9],
    )


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0, tzinfo=None).isoformat()
