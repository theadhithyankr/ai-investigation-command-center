from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from evidenceiq.models import EvidenceItem


SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_type TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    timestamp TEXT,
    sender TEXT,
    recipients TEXT NOT NULL,
    entities TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE
);
"""


class EvidenceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _init(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def upsert_many(self, items: list[EvidenceItem]) -> int:
        inserted = 0
        conn = self._connect()
        try:
            for item in items:
                before = conn.total_changes
                conn.execute(
                    """
                    INSERT OR IGNORE INTO evidence
                    (id, source, source_type, title, body, timestamp, sender, recipients, entities, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.id,
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
            conn.commit()
        finally:
            conn.close()
        return inserted

    def all(self) -> list[EvidenceItem]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id, source, source_type, title, body, timestamp, sender, recipients, entities, content_hash
                FROM evidence
                ORDER BY COALESCE(timestamp, '9999-12-31'), title
                """
            ).fetchall()
        finally:
            conn.close()
        items: list[EvidenceItem] = []
        for row in rows:
            timestamp = datetime.fromisoformat(row[5]) if row[5] else None
            item = EvidenceItem(
                id=row[0],
                source=row[1],
                source_type=row[2],
                title=row[3],
                body=row[4],
                timestamp=timestamp,
                sender=row[6],
                recipients=json.loads(row[7]),
                entities=json.loads(row[8]),
                content_hash=row[9],
            )
            items.append(item)
        return items
