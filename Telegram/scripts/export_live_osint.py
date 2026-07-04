import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.live_osint.storage import connect, fetch_events


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "artifacts" / "live_osint" / "live_events.db"
DEFAULT_OUT = ROOT / "artifacts" / "live_osint" / "live_events.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Export live OSINT events as dashboard-ready JSON.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--since-days", type=int, default=30, help="Only export messages from the last N days. Use 0 to disable.")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--include-non-conflict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    conn = connect(args.db)
    since_days = args.since_days if args.since_days and args.since_days > 0 else None
    events = fetch_events(
        conn,
        limit=args.limit,
        conflict_only=not args.include_non_conflict,
        since_days=since_days,
        min_confidence=args.min_confidence,
    )
    conn.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "telegram_allowlist",
        "filters": {
            "since_days": since_days,
            "min_confidence": args.min_confidence,
            "conflict_only": not args.include_non_conflict,
        },
        "event_count": len(events),
        "events": events,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {args.out}")
    print(f"events: {len(events)}")


if __name__ == "__main__":
    main()
