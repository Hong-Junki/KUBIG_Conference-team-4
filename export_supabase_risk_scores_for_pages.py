from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import quote_plus

import psycopg
from psycopg.rows import dict_row

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local convenience only
    load_dotenv = None


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "Telegram" / "risk_scores_live.json"

if load_dotenv:
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT.parent / ".env")


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name) or default


def database_url() -> str:
    direct = env("DATABASE_URL")
    if direct:
        return direct.replace("postgresql+psycopg://", "postgresql://")

    password = env("SUPABASE_DB_PASSWORD") or env("supabase_password")
    host = env("SUPABASE_DB_HOST")
    if not password or not host:
        raise RuntimeError("Set DATABASE_URL or SUPABASE_DB_HOST + SUPABASE_DB_PASSWORD")

    user = env("SUPABASE_DB_USER", "postgres")
    port = env("SUPABASE_DB_PORT", "5432")
    dbname = env("SUPABASE_DB_NAME", "postgres")
    return f"postgresql://{user}:{quote_plus(password)}@{host}:{port}/{dbname}"


def json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def dashboard_tier(row: dict) -> str:
    level = str(row.get("map_risk_level") or row.get("alert_level") or "").upper()
    if level == "ALERT":
        return "critical"
    if level == "WARNING":
        return "high"
    if level == "WATCH":
        return "watch"
    score = float(row.get("current_risk_score") or row.get("raw_score") or 0)
    if score >= 70:
        return "critical"
    if score >= 50:
        return "high"
    if score >= 25:
        return "watch"
    return "low"


def row_to_dashboard(row: dict) -> dict:
    iso3 = row["iso3"]
    onset_prob = row.get("onset_prob")
    top_features = row.get("top_features") or {}
    if onset_prob is None and isinstance(top_features, dict):
        onset_prob = top_features.get("onset_prob")

    risk_score = row.get("current_risk_score")
    if risk_score is None:
        risk_score = row.get("raw_score")

    return {
        "country": iso3,
        "date": json_default(row.get("feature_date")),
        "run_ts": json_default(row.get("scored_at")),
        "base_pred": row.get("base_pred"),
        "onset_prob": float(onset_prob or 0),
        "calm_flag": row.get("calm_flag"),
        "risk_score": round(float(risk_score or 0), 4),
        "onset_alert_score": row.get("onset_alert_score"),
        "watchlist_score": row.get("watchlist_score"),
        "current_risk_score": row.get("current_risk_score"),
        "map_risk_level": row.get("map_risk_level"),
        "score_policy_version": row.get("score_policy_version"),
        "tier": dashboard_tier(row),
    }


def fetch_scores() -> dict:
    with psycopg.connect(database_url(), sslmode="require", row_factory=dict_row) as conn:
        latest = conn.execute(
            """
            select
              l.iso3,
              l.horizon_days,
              l.scored_at,
              l.feature_date,
              l.model_version_id,
              l.raw_score,
              l.alert_level,
              l.onset_alert_score,
              l.watchlist_score,
              l.current_risk_score,
              l.map_risk_level,
              l.score_policy_version,
              l.top_features
            from latest_risk_scores l
            where l.horizon_days = 7
            order by coalesce(l.current_risk_score, l.raw_score) desc
            """
        ).fetchall()

        history = conn.execute(
            """
            select
              h.iso3,
              h.horizon_days,
              h.scored_at,
              h.feature_date,
              h.model_version_id,
              h.raw_score,
              h.alert_level,
              h.onset_alert_score,
              h.watchlist_score,
              h.current_risk_score,
              h.map_risk_level,
              h.score_policy_version
            from risk_score_history h
            where h.horizon_days = 7
              and h.scored_at >= now() - interval '45 days'
            order by h.iso3, h.scored_at
            """
        ).fetchall()

    countries = {row["iso3"]: row_to_dashboard(row) for row in latest}
    grouped_history: dict[str, list[dict]] = defaultdict(list)
    for row in history:
        grouped_history[row["iso3"]].append(row_to_dashboard(row))

    return {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": "supabase.latest_risk_scores",
        "score_system": "dashboard_policy_v1_candidate3_candidate4",
        "countries": countries,
        "history": grouped_history,
    }


def main() -> None:
    payload = fetch_scores()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH} with {len(payload['countries'])} countries")


if __name__ == "__main__":
    main()
