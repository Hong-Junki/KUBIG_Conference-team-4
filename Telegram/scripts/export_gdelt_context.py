import argparse
import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = ROOT / "artifacts" / "live_osint" / "live_events.json"
DEFAULT_OUT = ROOT / "artifacts" / "live_osint" / "gdelt_context.json"
DEFAULT_PROJECT = "conflict-early-warning"
DEFAULT_DATASET = "conflict_ew"
DEFAULT_TABLE = "gdelt_titles"

sys.path.append(str(ROOT))
from src.live_osint.extraction import COUNTRY_CENTROIDS


def parse_args():
    parser = argparse.ArgumentParser(description="Export country-level GDELT title context for the Telegram OSINT UI.")
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--project", default=os.getenv("GCP_PROJECT_ID", DEFAULT_PROJECT))
    parser.add_argument("--dataset", default=os.getenv("BIGQUERY_DATASET", DEFAULT_DATASET))
    parser.add_argument("--table", default=os.getenv("GDELT_TITLES_TABLE", DEFAULT_TABLE))
    parser.add_argument("--credentials", type=Path, default=None, help="Optional service-account JSON path.")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--top-titles", type=int, default=3)
    parser.add_argument("--max-countries", type=int, default=40)
    return parser.parse_args()


def load_countries(events_path: Path, max_countries: int) -> list[str]:
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    seen = []
    for event in payload.get("events", []):
        country = event.get("country")
        if country and country != "UNK" and country not in seen:
            seen.append(country)
        if len(seen) >= max_countries:
            break
    for country in COUNTRY_CENTROIDS:
        if country not in seen:
            seen.append(country)
        if len(seen) >= max_countries:
            break
    return seen


def query_counts(client, table_ref: str, countries: list[str], days: int) -> dict[str, dict]:
    anchor_date = query_anchor_date(client, table_ref, countries)
    if not anchor_date:
        return {}
    query = f"""
    SELECT
      iso3,
      COUNTIF(date = @anchor_date) AS gdelt_24h,
      COUNTIF(date >= DATE_SUB(@anchor_date, INTERVAL @lookback_days DAY)) AS gdelt_7d
    FROM `{table_ref}`
    WHERE iso3 IN UNNEST(@countries)
      AND date >= DATE_SUB(@anchor_date, INTERVAL @lookback_days DAY)
    GROUP BY iso3
    """
    job_config = bigquery_job_config(
        [
            ("countries", "STRING", countries, True),
            ("lookback_days", "INT64", max(days - 1, 0), False),
            ("anchor_date", "DATE", anchor_date.isoformat(), False),
        ]
    )
    out = {}
    for row in client.query(query, job_config=job_config):
        out[row.iso3] = {
            "gdelt_24h": int(row.gdelt_24h or 0),
            "gdelt_7d": int(row.gdelt_7d or 0),
            "top_keywords": [],
            "top_titles": [],
            "anchor_date": anchor_date.isoformat(),
        }
    return out


def query_anchor_date(client, table_ref: str, countries: list[str]):
    query = f"""
    SELECT MAX(date) AS anchor_date
    FROM `{table_ref}`
    WHERE iso3 IN UNNEST(@countries)
    """
    job_config = bigquery_job_config([("countries", "STRING", countries, True)])
    rows = list(client.query(query, job_config=job_config))
    return rows[0].anchor_date if rows and rows[0].anchor_date else None


def query_titles(client, table_ref: str, country: str, days: int, limit: int) -> list[dict]:
    anchor_date = query_anchor_date(client, table_ref, [country])
    if not anchor_date:
        return []
    query = f"""
    SELECT date, title, domain, url
    FROM `{table_ref}`
    WHERE iso3 = @country
      AND date >= DATE_SUB(@anchor_date, INTERVAL @lookback_days DAY)
      AND title IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (PARTITION BY url ORDER BY date DESC) = 1
    ORDER BY date DESC
    LIMIT @limit
    """
    job_config = bigquery_job_config(
        [
            ("country", "STRING", country, False),
            ("lookback_days", "INT64", max(days - 1, 0), False),
            ("limit", "INT64", limit, False),
            ("anchor_date", "DATE", anchor_date.isoformat(), False),
        ]
    )
    return [
        {
            "date": row.date.isoformat() if row.date else None,
            "title": html.unescape(row.title or ""),
            "domain": html.unescape(row.domain or ""),
            "url": row.url,
        }
        for row in client.query(query, job_config=job_config)
    ]


def query_keywords(client, table_ref: str, country: str, days: int, limit: int = 6) -> list[str]:
    anchor_date = query_anchor_date(client, table_ref, [country])
    if not anchor_date:
        return []
    query = f"""
    WITH tokens AS (
      SELECT LOWER(token) AS token
      FROM `{table_ref}`,
      UNNEST(REGEXP_EXTRACT_ALL(COALESCE(title, ''), r'[A-Za-z]{{4,}}')) AS token
      WHERE iso3 = @country
        AND date >= DATE_SUB(@anchor_date, INTERVAL @lookback_days DAY)
        AND title IS NOT NULL
    )
    SELECT token, COUNT(*) AS n
    FROM tokens
    WHERE token NOT IN UNNEST(@stopwords)
    GROUP BY token
    ORDER BY n DESC, token
    LIMIT @limit
    """
    stopwords = [
        "after",
        "amid",
        "from",
        "have",
        "into",
        "over",
        "said",
        "says",
        "that",
        "their",
        "this",
        "with",
    ]
    job_config = bigquery_job_config(
        [
            ("country", "STRING", country, False),
            ("lookback_days", "INT64", max(days - 1, 0), False),
            ("limit", "INT64", limit, False),
            ("stopwords", "STRING", stopwords, True),
            ("anchor_date", "DATE", anchor_date.isoformat(), False),
        ]
    )
    return [row.token for row in client.query(query, job_config=job_config)]


def korean_context_summary(country: str, item: dict) -> str:
    latest = int(item.get("gdelt_24h") or 0)
    recent = int(item.get("gdelt_7d") or 0)
    keywords = item.get("top_keywords") or []
    titles = item.get("top_titles") or []
    anchor = item.get("anchor_date")

    if recent <= 0:
        return f"{country} 관련 GDELT 기사 제목 컨텍스트는 최근 수집 구간에서 충분히 확인되지 않았습니다."

    keyword_text = ", ".join(keywords[:4]) if keywords else "주요 키워드 없음"
    title_context = infer_title_context(country, titles, keywords)
    if latest > 0:
        volume = f"GDELT 최신 수집일({anchor})에는 {country} 관련 제목이 {latest:,}건, 최근 7일 기준 {recent:,}건 확인됩니다."
    else:
        volume = f"GDELT 최근 7일 기준 {country} 관련 제목이 {recent:,}건 확인됩니다."

    if titles:
        domains = sorted({title.get("domain") for title in titles if title.get("domain")})
        domain_text = ", ".join(domains[:3])
        source_hint = f" 대표 제목은 {domain_text or '여러 매체'} 등에서 확인됩니다."
    else:
        source_hint = " 대표 제목은 제한적입니다."

    return f"{volume} {title_context} 주요 반복 키워드는 {keyword_text}입니다.{source_hint}"


def infer_title_context(country: str, titles: list[dict], keywords: list[str]) -> str:
    title_text = " ".join(str(item.get("title") or "") for item in titles).lower()
    keyword_text = " ".join(keywords).lower()
    text = f"{title_text} {keyword_text}"

    themes = []
    theme_rules = [
        (
            "군사 충돌과 공격 피해",
            [
                "attack",
                "strike",
                "missile",
                "drone",
                "airstrike",
                "shelling",
                "explosion",
                "killed",
                "wounded",
                "casualties",
            ],
        ),
        (
            "외교 협상과 국제 대응",
            [
                "ceasefire",
                "peace",
                "deal",
                "talks",
                "summit",
                "trump",
                "nato",
                "un ",
                "minister",
                "sanctions",
            ],
        ),
        (
            "인도주의 위기와 민간 피해",
            [
                "gaza",
                "hospital",
                "aid",
                "humanitarian",
                "refugee",
                "evacuation",
                "children",
                "food",
            ],
        ),
        (
            "정치·정부 발표와 국내 여론",
            [
                "president",
                "government",
                "election",
                "parliament",
                "protest",
                "officials",
                "governor",
            ],
        ),
        (
            "에너지·인프라 시설 영향",
            [
                "oil",
                "energy",
                "power",
                "plant",
                "infrastructure",
                "port",
                "refinery",
                "pipeline",
            ],
        ),
        (
            "국경 지역과 주변국 파급",
            [
                "border",
                "romania",
                "belarus",
                "poland",
                "lebanon",
                "iran",
                "israel",
                "russia",
            ],
        ),
    ]
    for label, terms in theme_rules:
        if any(term in text for term in terms):
            themes.append(label)

    if not themes:
        return f"대표 제목들은 {country} 관련 사건을 여러 지역·매체가 반복적으로 다루고 있음을 보여줍니다."

    if len(themes) == 1:
        return f"대표 제목들의 맥락은 주로 {themes[0]}에 집중되어 있습니다."

    return f"대표 제목들은 {', '.join(themes[:3])}을 함께 다루는 보도 흐름으로 볼 수 있습니다."


def bigquery_job_config(params):
    from google.cloud import bigquery

    query_params = []
    for name, typ, value, is_array in params:
        if is_array:
            query_params.append(bigquery.ArrayQueryParameter(name, typ, value))
        else:
            query_params.append(bigquery.ScalarQueryParameter(name, typ, value))
    return bigquery.QueryJobConfig(query_parameters=query_params)


def main():
    args = parse_args()
    if args.credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(args.credentials.resolve())

    try:
        from google.cloud import bigquery
    except ImportError:
        print("Missing dependency: google-cloud-bigquery. Run: pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(2)

    countries = load_countries(args.events, args.max_countries)
    if not countries:
        raise SystemExit("No countries found in live events.")

    client = bigquery.Client(project=args.project)
    table_ref = f"{args.project}.{args.dataset}.{args.table}"
    context = query_counts(client, table_ref, countries, args.days)

    for country in countries:
        item = context.setdefault(country, {"gdelt_24h": 0, "gdelt_7d": 0, "top_keywords": [], "top_titles": []})
        item["top_titles"] = query_titles(client, table_ref, country, args.days, args.top_titles)
        item["top_keywords"] = query_keywords(client, table_ref, country, args.days)
        item["ko_summary"] = korean_context_summary(country, item)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "bigquery_gdelt_titles",
        "table": table_ref,
        "days": args.days,
        "countries": context,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {args.out}")
    print(f"countries: {len(context)}")


if __name__ == "__main__":
    main()
