from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.live_osint.extraction import ExtractedEvent, parse_datetime


SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_messages (
    raw_id TEXT PRIMARY KEY,
    channel TEXT NOT NULL,
    message_id TEXT NOT NULL,
    message_time TEXT NOT NULL,
    text TEXT NOT NULL,
    url TEXT,
    raw_json TEXT NOT NULL,
    inserted_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS extracted_events (
    event_id TEXT PRIMARY KEY,
    raw_id TEXT NOT NULL,
    is_conflict_related INTEGER NOT NULL,
    country TEXT,
    location_name TEXT,
    latitude REAL,
    longitude REAL,
    location_precision TEXT DEFAULT 'missing',
    event_type TEXT NOT NULL,
    severity REAL NOT NULL,
    confidence REAL NOT NULL,
    summary TEXT NOT NULL,
    matched_keywords TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(raw_id) REFERENCES raw_messages(raw_id)
);

CREATE INDEX IF NOT EXISTS idx_raw_messages_time ON raw_messages(message_time);
CREATE INDEX IF NOT EXISTS idx_events_country_time ON extracted_events(country, updated_at);
CREATE INDEX IF NOT EXISTS idx_events_conflict ON extracted_events(is_conflict_related, confidence);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    ensure_columns(conn)
    return conn


def ensure_columns(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(extracted_events)").fetchall()}
    if "location_precision" not in columns:
        conn.execute("ALTER TABLE extracted_events ADD COLUMN location_precision TEXT DEFAULT 'missing'")


def upsert_raw_message(conn: sqlite3.Connection, raw: dict) -> str:
    channel = str(raw.get("channel") or "unknown")
    message_id = str(raw.get("message_id") or "")
    message_time = parse_datetime(raw.get("date")).isoformat()
    text = str(raw.get("text") or "")
    url = raw.get("url")
    raw_id = f"{channel}:{message_id}" if message_id else f"{channel}:{hash(text)}"

    conn.execute(
        """
        INSERT INTO raw_messages (raw_id, channel, message_id, message_time, text, url, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(raw_id) DO UPDATE SET
            message_time=excluded.message_time,
            text=excluded.text,
            url=excluded.url,
            raw_json=excluded.raw_json
        """,
        (raw_id, channel, message_id, message_time, text, url, json.dumps(raw, ensure_ascii=False)),
    )
    return raw_id


def upsert_event(conn: sqlite3.Connection, raw_id: str, event: ExtractedEvent) -> None:
    conn.execute(
        """
        INSERT INTO extracted_events (
            event_id, raw_id, is_conflict_related, country, location_name, latitude, longitude,
            event_type, severity, confidence, summary, matched_keywords, location_precision
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            is_conflict_related=excluded.is_conflict_related,
            country=excluded.country,
            location_name=excluded.location_name,
            latitude=excluded.latitude,
            longitude=excluded.longitude,
            location_precision=excluded.location_precision,
            event_type=excluded.event_type,
            severity=excluded.severity,
            confidence=excluded.confidence,
            summary=excluded.summary,
            matched_keywords=excluded.matched_keywords,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            event.event_id,
            raw_id,
            int(event.is_conflict_related),
            event.country,
            event.location_name,
            event.latitude,
            event.longitude,
            event.event_type,
            event.severity,
            event.confidence,
            event.summary,
            json.dumps(event.matched_keywords, ensure_ascii=False),
            event.location_precision,
        ),
    )


def fetch_events(
    conn: sqlite3.Connection,
    limit: int = 200,
    conflict_only: bool = True,
    since_days: int | None = None,
    min_confidence: float | None = None,
) -> list[dict]:
    clauses = []
    params: list[object] = []
    if conflict_only:
        clauses.append("e.is_conflict_related = 1")
    if since_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
        clauses.append("r.message_time >= ?")
        params.append(cutoff.isoformat())
    if min_confidence is not None:
        clauses.append("e.confidence >= ?")
        params.append(min_confidence)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    rows = conn.execute(
        f"""
        SELECT
            e.event_id,
            e.is_conflict_related,
            e.country,
            e.location_name,
            e.latitude,
            e.longitude,
            e.location_precision,
            e.event_type,
            e.severity,
            e.confidence,
            e.summary,
            e.matched_keywords,
            r.channel,
            r.message_id,
            r.message_time,
            r.text,
            r.url,
            r.raw_json
        FROM extracted_events e
        JOIN raw_messages r ON r.raw_id = e.raw_id
        {where}
        ORDER BY r.message_time DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    keys = [
        "event_id",
        "is_conflict_related",
        "country",
        "location_name",
        "latitude",
        "longitude",
        "location_precision",
        "event_type",
        "severity",
        "confidence",
        "summary",
        "matched_keywords",
        "channel",
        "message_id",
        "message_time",
        "raw_text",
        "url",
        "raw_json",
    ]
    out = []
    for row in rows:
        item = dict(zip(keys, row))
        item["is_conflict_related"] = bool(item["is_conflict_related"])
        item["matched_keywords"] = json.loads(item["matched_keywords"] or "[]")
        raw_json = json.loads(item.pop("raw_json") or "{}")
        item["source_reliability"] = raw_json.get("source_reliability")
        item["region_hint"] = raw_json.get("region_hint")
        out.append(item)
    return out
