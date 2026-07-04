import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "artifacts" / "live_osint" / "live_events.db"
DEFAULT_OUT = ROOT / "artifacts" / "live_osint" / "audit_report.md"

NEGATION_HINTS = [
    "no casualties",
    "no attack",
    "no clashes",
    "peaceful",
    "unconfirmed",
    "denies",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create a quality audit report for the live Telegram OSINT feed.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--fresh-days", type=int, default=14)
    parser.add_argument("--duplicate-window-hours", type=int, default=12)
    return parser.parse_args()


def fetch_rows(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT
            r.raw_id,
            r.channel,
            r.message_id,
            r.message_time,
            r.text,
            r.url,
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
            e.matched_keywords
        FROM raw_messages r
        LEFT JOIN extracted_events e ON e.raw_id = r.raw_id
        ORDER BY r.message_time DESC
        """
    ).fetchall()
    conn.close()
    out = []
    for row in rows:
        item = dict(row)
        item["is_conflict_related"] = bool(item.get("is_conflict_related"))
        item["matched_keywords"] = json.loads(item.get("matched_keywords") or "[]")
        out.append(item)
    return out


def pct(num: int, den: int) -> str:
    return "0.0%" if den == 0 else f"{num / den * 100:.1f}%"


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def text_key(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    stop = {"the", "and", "with", "from", "that", "this", "have", "has", "for", "are", "was", "were", "after"}
    tokens = [t for t in tokens if len(t) > 2 and t not in stop]
    return " ".join(tokens[:12])


def duplicate_groups(events: list[dict], window_hours: int) -> list[list[dict]]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for event in events:
        dt = parse_dt(event.get("message_time"))
        if not dt:
            continue
        bucket_time = int(dt.timestamp() // (window_hours * 3600))
        keywords = tuple(sorted(event.get("matched_keywords") or [])[:3])
        key = (event.get("country"), event.get("event_type"), keywords, bucket_time)
        buckets[key].append(event)

    groups = []
    for items in buckets.values():
        channels = {item.get("channel") for item in items}
        summaries = {text_key(item.get("summary") or item.get("text") or "") for item in items}
        if len(items) >= 2 and (len(channels) > 1 or len(summaries) < len(items)):
            groups.append(items)
    return sorted(groups, key=len, reverse=True)


def md_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def sample_event(event: dict) -> str:
    summary = (event.get("summary") or event.get("text") or "").replace("\n", " ")
    if len(summary) > 180:
        summary = summary[:177].rstrip() + "..."
    url = event.get("url") or ""
    return (
        f"- `{event.get('channel')}` `{event.get('message_time')}` "
        f"{event.get('country') or 'NO_COUNTRY'} / {event.get('event_type') or 'no_event'} "
        f"conf={float(event.get('confidence') or 0):.2f}, sev={float(event.get('severity') or 0):.2f}: "
        f"{summary} {url}"
    )


def build_report(rows: list[dict], sample_size: int, duplicate_window_hours: int, fresh_days: int) -> str:
    now = datetime.now(timezone.utc).isoformat()
    raw_total = len(rows)
    events = [r for r in rows if r.get("event_id")]
    conflict_events = [r for r in events if r.get("is_conflict_related")]
    non_conflict = [r for r in events if not r.get("is_conflict_related")]
    fresh_cutoff = datetime.now(timezone.utc) - timedelta(days=fresh_days)
    fresh_conflict_events = [
        e for e in conflict_events if (parse_dt(e.get("message_time")) or datetime.min.replace(tzinfo=timezone.utc)) >= fresh_cutoff
    ]

    channel_counts = Counter(r.get("channel") or "unknown" for r in rows)
    channel_event_counts = Counter(r.get("channel") or "unknown" for r in conflict_events)
    channel_rows = []
    for channel, raw_count in channel_counts.most_common():
        detected = channel_event_counts[channel]
        channel_rows.append([channel, raw_count, detected, pct(detected, raw_count)])

    missing_country = [e for e in conflict_events if not e.get("country")]
    missing_location = [e for e in conflict_events if not e.get("location_name")]
    missing_coords = [e for e in conflict_events if e.get("latitude") is None or e.get("longitude") is None]
    country_fallback = [e for e in conflict_events if e.get("location_precision") == "country"]
    low_conf = [e for e in conflict_events if float(e.get("confidence") or 0) < 0.75]
    high_sev_low_conf = [
        e for e in conflict_events if float(e.get("severity") or 0) >= 0.75 and float(e.get("confidence") or 0) < 0.75
    ]
    negation_suspects = [
        e for e in conflict_events if any(hint in (e.get("text") or "").lower() for hint in NEGATION_HINTS)
    ]
    old_events = []
    for e in conflict_events:
        dt = parse_dt(e.get("message_time"))
        if dt and dt < fresh_cutoff:
            old_events.append(e)

    keyword_counts = Counter()
    for e in conflict_events:
        keyword_counts.update(e.get("matched_keywords") or [])

    dupes = duplicate_groups(conflict_events, duplicate_window_hours)

    lines = [
        "# Live OSINT Feed Audit Report",
        "",
        f"- Generated at: `{now}`",
        f"- Freshness threshold: `{fresh_days}` days",
        f"- Raw messages: `{raw_total}`",
        f"- Extracted rows: `{len(events)}`",
        f"- Conflict events: `{len(conflict_events)}`",
        f"- Fresh conflict events: `{len(fresh_conflict_events)}`",
        f"- Non-conflict extracted rows: `{len(non_conflict)}`",
        "",
        "## Channel Detection Rates",
        "",
        *md_table(["channel", "raw_messages", "conflict_events", "detection_rate"], channel_rows),
        "",
        "## Quality Flags",
        "",
        *md_table(
            ["flag", "count", "rate_among_conflict_events"],
            [
                ["missing_country", len(missing_country), pct(len(missing_country), len(conflict_events))],
                ["missing_location", len(missing_location), pct(len(missing_location), len(conflict_events))],
                ["missing_coordinates", len(missing_coords), pct(len(missing_coords), len(conflict_events))],
                ["country_centroid_fallback", len(country_fallback), pct(len(country_fallback), len(conflict_events))],
                ["low_confidence_lt_0.75", len(low_conf), pct(len(low_conf), len(conflict_events))],
                ["high_severity_low_confidence", len(high_sev_low_conf), pct(len(high_sev_low_conf), len(conflict_events))],
                ["negation_or_uncertainty_suspects", len(negation_suspects), pct(len(negation_suspects), len(conflict_events))],
                [f"older_than_{fresh_days}_days", len(old_events), pct(len(old_events), len(conflict_events))],
                ["possible_duplicate_groups", len(dupes), "-"],
            ],
        ),
        "",
        "## Top Matched Keywords",
        "",
        *md_table(["keyword", "count"], [[k, v] for k, v in keyword_counts.most_common(15)]),
        "",
        "## Samples To Review",
        "",
        "### Missing Country",
        "",
    ]

    lines.extend(sample_event(e) for e in missing_country[:sample_size])
    lines.extend(["", "### Missing Coordinates", ""])
    lines.extend(sample_event(e) for e in missing_coords[:sample_size])
    lines.extend(["", "### Low Confidence", ""])
    lines.extend(sample_event(e) for e in low_conf[:sample_size])
    lines.extend(["", "### Negation Or Uncertainty Suspects", ""])
    lines.extend(sample_event(e) for e in negation_suspects[:sample_size])
    lines.extend(["", f"### Older Than {fresh_days} Days", ""])
    lines.extend(sample_event(e) for e in old_events[:sample_size])
    lines.extend(["", "### Possible Duplicate Groups", ""])
    if not dupes:
        lines.append("- No duplicate groups detected by the current heuristic.")
    for idx, group in enumerate(dupes[:sample_size], start=1):
        lines.append(f"- Group {idx}: {len(group)} events")
        lines.extend("  " + sample_event(e) for e in group[:5])

    lines.extend(
        [
            "",
            "## Suggested Next Actions",
            "",
            "- If missing coordinates are high, expand `CITY_COORDS` and country aliases in `src/live_osint/extraction.py`.",
            "- If old events are high, add a time-window filter during export or collection.",
            "- If low-confidence events are useful, lower display threshold; if noisy, raise it.",
            "- If one keyword dominates false positives, reduce its weight or require a country/location match.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    rows = fetch_rows(args.db)
    report = build_report(rows, args.sample_size, args.duplicate_window_hours, args.fresh_days)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
