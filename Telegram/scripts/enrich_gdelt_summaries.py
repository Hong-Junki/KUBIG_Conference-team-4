import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTEXT = ROOT / "artifacts" / "live_osint" / "gdelt_context.json"
DEFAULT_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4.1-mini")


def parse_args():
    parser = argparse.ArgumentParser(description="Enhance GDELT country summaries with the OpenAI API.")
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--out", type=Path, default=None, help="Defaults to overwriting --context.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-countries", type=int, default=40)
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


def title_lines(titles: list[dict]) -> str:
    lines = []
    for item in titles[:5]:
        title = str(item.get("title") or "").strip()
        domain = str(item.get("domain") or "").strip()
        date = str(item.get("date") or "").strip()
        if title:
            lines.append(f"- {date} {domain}: {title}")
    return "\n".join(lines) if lines else "- representative titles unavailable"


def build_prompt(country: str, item: dict) -> str:
    keywords = ", ".join(item.get("top_keywords") or [])
    return f"""
Country ISO3: {country}
GDELT anchor date: {item.get("anchor_date") or "unknown"}
Latest-day title count: {item.get("gdelt_24h") or 0}
Lookback title count: {item.get("gdelt_7d") or 0}
Top repeated terms: {keywords or "none"}
Representative titles:
{title_lines(item.get("top_titles") or [])}
""".strip()


def summarize_country(client, model: str, country: str, item: dict) -> dict:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ko_summary": {
                "type": "string",
                "description": "2-3 Korean sentences explaining the country-level news context from the titles.",
            },
            "ko_brief": {
                "type": "string",
                "description": "A short Korean one-line panel headline.",
            },
            "source_limits": {
                "type": "string",
                "description": "A short Korean caveat about title-only summarization and source limits.",
            },
        },
        "required": ["ko_summary", "ko_brief", "source_limits"],
    }
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You write Korean geopolitical news context for an OSINT dashboard. "
                    "Use only the provided GDELT article titles, domains, counts, and keywords. "
                    "Do not invent facts, casualty numbers, actors, or locations that are not supported by the titles. "
                    "If the titles are broad or noisy, say that the signal is broad/noisy. "
                    "Keep the tone sober, analytical, and concise."
                ),
            },
            {"role": "user", "content": build_prompt(country, item)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "gdelt_country_summary",
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

    payload = json.loads(args.context.read_text(encoding="utf-8"))
    countries = payload.get("countries") or {}
    client = OpenAI()
    enriched = 0

    for country, item in list(countries.items())[: args.max_countries]:
        if int(item.get("gdelt_7d") or 0) <= 0 and int(item.get("gdelt_24h") or 0) <= 0:
            continue
        result = summarize_country(client, args.model, country, item)
        item["ko_summary_rule_based"] = item.get("ko_summary")
        item["ko_summary"] = result["ko_summary"]
        item["ko_brief"] = result["ko_brief"]
        item["source_limits"] = result["source_limits"]
        item["summary_model"] = args.model
        enriched += 1
        print(f"enriched: {country}", flush=True)
        if args.dry_run:
            break

    payload["summary_source"] = "openai_responses_api"
    payload["summary_model"] = args.model
    payload["summary_enriched_countries"] = enriched
    out = args.out or args.context
    if not args.dry_run:
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {out}")
    else:
        print(f"dry_run_enriched_countries: {enriched}")
        for country, item in list(countries.items())[: args.max_countries]:
            if item.get("ko_brief"):
                print(f"{country}: {item['ko_brief'].encode('ascii', errors='backslashreplace').decode('ascii')}")


if __name__ == "__main__":
    main()
