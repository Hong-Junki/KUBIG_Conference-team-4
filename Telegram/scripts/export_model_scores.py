import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "live_osint" / "model_scores.json"
DEFAULT_CREDENTIALS = ROOT.parent / "conflict-ew-mvp-20260604-4af3cecfb588.json"
DEFAULT_PROJECT = "conflict-ew-mvp-20260604"
DEFAULT_DATASET = "conflict_ew"
DEFAULT_TABLE = "model_scores"


def parse_args():
    parser = argparse.ArgumentParser(description="Export latest model onset scores for the static OSINT UI.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    return parser.parse_args()


def tier_from_percentile(percentile: float) -> str:
    if percentile >= 98:
        return "critical"
    if percentile >= 95:
        return "high"
    if percentile >= 85:
        return "watch"
    return "low"


def main():
    args = parse_args()
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
    except ImportError:
        raise SystemExit("Missing dependency: google-cloud-bigquery. Run: pip install -r requirements.txt")

    if not args.credentials.exists():
        raise SystemExit(f"credentials not found: {args.credentials}")

    creds = service_account.Credentials.from_service_account_file(str(args.credentials))
    client = bigquery.Client(project=args.project, credentials=creds)
    table_ref = f"{args.project}.{args.dataset}.{args.table}"
    latest_query = f"""
    WITH latest AS (
      SELECT country, date, run_ts, base_pred, onset_prob, calm_flag
      FROM `{table_ref}`
      QUALIFY ROW_NUMBER() OVER (PARTITION BY country, date ORDER BY run_ts DESC) = 1
    ), max_date AS (
      SELECT MAX(date) AS d FROM latest
    ), selected AS (
      SELECT latest.*
      FROM latest, max_date
      WHERE date = d
    )
    SELECT
      country,
      date,
      run_ts,
      base_pred,
      onset_prob,
      calm_flag,
      PERCENT_RANK() OVER (ORDER BY onset_prob) AS onset_percent_rank,
      CUME_DIST() OVER (ORDER BY onset_prob) AS onset_cume_dist
    FROM selected
    ORDER BY onset_prob DESC
    """
    rows = list(client.query(latest_query))
    countries = {}
    for row in rows:
        percentile = round(float(row.onset_cume_dist or 0) * 100, 2)
        countries[row.country] = {
            "country": row.country,
            "date": row.date.isoformat() if row.date else None,
            "run_ts": row.run_ts,
            "base_pred": float(row.base_pred or 0),
            "onset_prob": float(row.onset_prob or 0),
            "calm_flag": int(row.calm_flag or 0),
            "risk_score": percentile,
            "onset_percentile": percentile,
            "tier": tier_from_percentile(percentile),
        }

    history_query = f"""
    WITH dedup AS (
      SELECT country, date, run_ts, base_pred, onset_prob, calm_flag
      FROM `{table_ref}`
      QUALIFY ROW_NUMBER() OVER (PARTITION BY country, date ORDER BY run_ts DESC) = 1
    ), scored AS (
      SELECT
        country,
        date,
        run_ts,
        base_pred,
        onset_prob,
        calm_flag,
        CUME_DIST() OVER (PARTITION BY date ORDER BY onset_prob) AS onset_cume_dist
      FROM dedup
    )
    SELECT country, date, run_ts, base_pred, onset_prob, calm_flag, onset_cume_dist
    FROM scored
    ORDER BY country, date
    """
    history_rows = list(client.query(history_query))
    history_by_country = defaultdict(list)
    for row in history_rows:
        percentile = round(float(row.onset_cume_dist or 0) * 100, 2)
        history_by_country[row.country].append(
            {
                "date": row.date.isoformat() if row.date else None,
                "run_ts": row.run_ts,
                "base_pred": float(row.base_pred or 0),
                "onset_prob": float(row.onset_prob or 0),
                "calm_flag": int(row.calm_flag or 0),
                "risk_score": percentile,
                "tier": tier_from_percentile(percentile),
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "bigquery_model_scores",
        "table": table_ref,
        "score_system": {
            "primary": "onset_prob",
            "risk_score": "latest-date cross-country percentile of onset_prob, scaled 0-100",
            "period_risk_score": "selected-period average and maximum of daily cross-country percentile scores",
            "tier": "critical >=98, high >=95, watch >=85, low otherwise",
            "note": "onset_prob is a ranking score, not a calibrated absolute probability. No finance/security subscore columns are present in model_scores.",
        },
        "countries": countries,
        "history": dict(history_by_country),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {args.out}")
    print(f"countries: {len(countries)}")
    print(f"history rows: {len(history_rows)}")
    if countries:
        top = sorted(countries.values(), key=lambda item: item["risk_score"], reverse=True)[:10]
        for item in top:
            print(f"{item['country']} risk={item['risk_score']:.2f} onset={item['onset_prob']:.4f} tier={item['tier']} calm={item['calm_flag']}")


if __name__ == "__main__":
    main()
