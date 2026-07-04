import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.live_osint.extraction import extract_event
from src.live_osint.storage import connect, upsert_event


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "artifacts" / "live_osint" / "live_events.db"


def parse_args():
    parser = argparse.ArgumentParser(description="Re-run event extraction for all stored raw Telegram messages.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    return parser.parse_args()


def main():
    args = parse_args()
    conn = connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT raw_id, raw_json
        FROM raw_messages
        ORDER BY message_time DESC
        """
    ).fetchall()

    event_count = 0
    with conn:
        for row in rows:
            raw = json.loads(row["raw_json"])
            event = extract_event(raw)
            upsert_event(conn, row["raw_id"], event)
            if event.is_conflict_related:
                event_count += 1
    conn.close()
    print(f"raw_messages_reprocessed: {len(rows)}")
    print(f"conflict_events_detected: {event_count}")


if __name__ == "__main__":
    main()
