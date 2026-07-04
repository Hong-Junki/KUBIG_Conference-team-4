import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = ROOT / "artifacts" / "live_osint" / "live_events.json"
DEFAULT_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4.1-mini")


def parse_args():
    parser = argparse.ArgumentParser(description="Enhance Telegram raw-message summaries with the OpenAI API.")
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--out", type=Path, default=None, help="Defaults to overwriting --events.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-events", type=int, default=120)
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save progress every N enriched events.")
    parser.add_argument("--force", action="store_true", help="Regenerate summaries even when a model summary exists.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def compact_text(value: str, limit: int = 2200) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def build_prompt(event: dict) -> str:
    keywords = ", ".join(event.get("matched_keywords") or [])
    return f"""
Telegram channel: {event.get("channel") or "unknown"}
Message time: {event.get("message_time") or "unknown"}
Country: {event.get("country") or "unknown"}
Location: {event.get("location_name") or "unknown"}
Detected event type: {event.get("event_type") or "signal"}
Severity score: {event.get("severity") or 0}
Confidence score: {event.get("confidence") or 0}
Matched keywords: {keywords or "none"}
Raw message:
{compact_text(event.get("raw_text") or event.get("summary") or "")}
""".strip()


def summarize_event(client, model: str, event: dict) -> dict:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ko_raw_summary": {
                "type": "string",
                "description": "2-3 Korean sentences summarizing only the provided Telegram raw message.",
            },
            "ko_raw_brief": {
                "type": "string",
                "description": "A short Korean one-line headline for the message.",
            },
            "source_limits": {
                "type": "string",
                "description": "A short Korean caveat about uncertainty, claims, or source limits.",
            },
        },
        "required": ["ko_raw_summary", "ko_raw_brief", "source_limits"],
    }
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You summarize Telegram OSINT messages in Korean for a conflict-monitoring dashboard. "
                    "Use only the provided raw message and metadata. Do not invent casualty numbers, locations, "
                    "attackers, victims, or verification status. If the message is a claim, allegation, denial, "
                    "political statement, or commentary rather than a confirmed incident, say so clearly. "
                    "Keep the tone sober and analytical."
                ),
            },
            {"role": "user", "content": build_prompt(event)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "telegram_raw_summary",
                "strict": True,
                "schema": schema,
            }
        },
    )
    return json.loads(response.output_text)


def main():
    args = parse_args()
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / ".env.local")

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Add it to .env.local or your shell environment.")

    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("Missing dependency: openai. Run: pip install -r requirements.txt")

    payload = json.loads(args.events.read_text(encoding="utf-8"))
    events = payload.get("events") or []
    client = OpenAI()
    enriched = 0
    skipped = 0
    out = args.out or args.events

    def save_progress() -> None:
        payload["telegram_summary_source"] = "openai_responses_api"
        payload["telegram_summary_model"] = args.model
        payload["telegram_summary_enriched_events"] = enriched
        payload["telegram_summary_skipped_events"] = skipped
        if not args.dry_run:
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for event in events[: args.max_events]:
        if event.get("ko_raw_summary_model") and not args.force:
            skipped += 1
            continue
        result = summarize_event(client, args.model, event)
        if event.get("ko_raw_summary"):
            event["ko_raw_summary_rule_based"] = event.get("ko_raw_summary")
        event["ko_raw_summary"] = result["ko_raw_summary"]
        event["ko_raw_brief"] = result["ko_raw_brief"]
        event["ko_raw_source_limits"] = result["source_limits"]
        event["ko_raw_summary_model"] = args.model
        enriched += 1
        print(f"enriched: {event.get('event_id')} {event.get('country') or 'UNK'} {event.get('channel')}", flush=True)
        if args.checkpoint_every > 0 and enriched % args.checkpoint_every == 0:
            save_progress()
            print(f"checkpoint_saved: {out} enriched={enriched} skipped={skipped}", flush=True)
        if args.dry_run:
            break

    if args.dry_run:
        print(f"dry_run_enriched_events: {enriched}")
        for event in events[: args.max_events]:
            if event.get("ko_raw_brief"):
                safe = event["ko_raw_brief"].encode("ascii", errors="backslashreplace").decode("ascii")
                print(f"{event.get('event_id')}: {safe}")
                break
    else:
        save_progress()
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
