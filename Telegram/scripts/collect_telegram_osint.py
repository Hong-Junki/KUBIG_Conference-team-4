import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.live_osint.extraction import extract_event
from src.live_osint.storage import connect, upsert_event, upsert_raw_message
from src.live_osint.telegram_client import collect_with_telethon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "artifacts" / "live_osint" / "live_events.db"
DEFAULT_CONFIG = ROOT / "config" / "telegram_channels.json"
DEFAULT_SAMPLE = ROOT / "examples" / "telegram_sample_messages.json"


def load_sample(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def ingest_messages(messages: list[dict], db_path: Path, since_days: int | None = None) -> tuple[int, int, int]:
    conn = connect(db_path)
    raw_count = event_count = skipped_old = 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days) if since_days else None
    with conn:
        for raw in messages:
            if cutoff and parse_dt(str(raw.get("date"))) < cutoff:
                skipped_old += 1
                continue
            raw_id = upsert_raw_message(conn, raw)
            raw_count += 1
            event = extract_event(raw)
            upsert_event(conn, raw_id, event)
            if event.is_conflict_related:
                event_count += 1
    conn.close()
    return raw_count, event_count, skipped_old


def parse_args():
    parser = argparse.ArgumentParser(description="Collect Telegram OSINT messages into a local SQLite event store.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit-per-channel", type=int, default=50)
    parser.add_argument("--since-days", type=int, default=30, help="Skip collected messages older than N days. Use 0 to disable.")
    parser.add_argument("--demo", action="store_true", help="Use examples/telegram_sample_messages.json instead of Telegram API.")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.demo:
        messages = load_sample(args.sample)
    else:
        messages = asyncio.run(collect_with_telethon(args.config, args.limit_per_channel))

    since_days = args.since_days if args.since_days and args.since_days > 0 else None
    raw_count, event_count, skipped_old = ingest_messages(messages, args.db, since_days=since_days)
    print(f"db: {args.db}")
    print(f"raw_messages_ingested: {raw_count}")
    print(f"conflict_events_detected: {event_count}")
    print(f"skipped_old_messages: {skipped_old}")


if __name__ == "__main__":
    main()
