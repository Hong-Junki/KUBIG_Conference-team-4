from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import psycopg
from google.cloud import bigquery
from psycopg.rows import dict_row


ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL_VERSION_ID = "onset_prod_lstm_v1"
DEFAULT_MODEL_NAME = "onset_prod_lstm"
DEFAULT_TARGET_NAME = "y_onset"
DEFAULT_HORIZON_DAYS = 7
SCORE_POLICY_VERSION = "dashboard_policy_v1_candidate3_candidate4"


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def database_url() -> str:
    direct = env("DATABASE_URL")
    if direct:
        return direct.replace("postgresql+psycopg://", "postgresql://")

    host = env("SUPABASE_DB_HOST")
    password = env("SUPABASE_DB_PASSWORD") or env("supabase_password")
    if not host or not password:
        raise RuntimeError("Set DATABASE_URL or SUPABASE_DB_HOST + SUPABASE_DB_PASSWORD in .env")

    user = env("SUPABASE_DB_USER", "postgres")
    dbname = env("SUPABASE_DB_NAME", "postgres")
    port = env("SUPABASE_DB_PORT", "5432")
    return f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{dbname}"


def connect() -> psycopg.Connection:
    return psycopg.connect(database_url(), sslmode="require")


def bq_context() -> tuple[str, str, str, str]:
    project_id = env("GCP_PROJECT_ID") or env("GOOGLE_CLOUD_PROJECT")
    dataset_id = env("BIGQUERY_DATASET", "conflict_ew")
    scores_table = env("BIGQUERY_MODEL_SCORES_TABLE", "model_scores")
    input_table = env("BIGQUERY_MODEL_INPUT_TABLE", "model_input")
    if not project_id:
        raise RuntimeError("Set GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT in .env")
    return project_id, dataset_id, scores_table, input_table


def choose_column(columns: set[str], candidates: list[str], *, required: bool = True) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for name in candidates:
        if name in columns:
            return name
        if name.lower() in lowered:
            return lowered[name.lower()]
    if required:
        raise RuntimeError(f"None of these columns were found: {candidates}. Available={sorted(columns)}")
    return None


def table_columns(client: bigquery.Client, table_id: str) -> set[str]:
    table = client.get_table(table_id)
    return {field.name for field in table.schema}


def infer_score_mapping(columns: set[str]) -> dict[str, str | None]:
    return {
        "iso3": choose_column(columns, ["iso3", "country", "country_iso3", "ActionGeo_CountryCode"]),
        "feature_date": choose_column(columns, ["feature_date", "date", "score_date", "target_date", "ds"]),
        "raw_score": choose_column(
            columns,
            ["risk_score", "score", "onset_score", "raw_score", "pred_score", "probability", "p_onset", "onset_prob"],
        ),
        "alert_level": choose_column(columns, ["alert_level", "risk_level", "tier", "risk_tier"], required=False),
        "scored_at": choose_column(columns, ["scored_at", "score_timestamp", "created_at", "updated_at", "run_ts"], required=False),
        "model_version_id": choose_column(columns, ["model_version_id", "model_id", "model_name"], required=False),
        "horizon_days": choose_column(columns, ["horizon_days", "horizon", "lead_days"], required=False),
    }


def ensure_onset_model_metadata(conn: psycopg.Connection, project_id: str, dataset_id: str, scores_table: str, input_table: str) -> None:
    model_version_id = env("ONSET_MODEL_VERSION_ID", DEFAULT_MODEL_VERSION_ID)
    model_name = env("ONSET_MODEL_NAME", DEFAULT_MODEL_NAME)
    target_name = env("ONSET_TARGET_NAME", DEFAULT_TARGET_NAME)
    horizon_days = int(env("ONSET_HORIZON_DAYS", str(DEFAULT_HORIZON_DAYS)))
    artifact_uri = env("ONSET_ARTIFACT_URI", "drive://가중치/onset_prod")
    feature_schema_uri = f"bigquery://{project_id}.{dataset_id}.{input_table}"
    metrics = {
        "score_source": f"bigquery://{project_id}.{dataset_id}.{scores_table}",
        "serving_note": "Scores are produced upstream by the onset serving container and synced into Supabase.",
        "score_policy_version": SCORE_POLICY_VERSION,
        "dashboard_policy": {
            "onset_alert_score": "Candidate 3 Hybrid 70/30",
            "watchlist_score": "Candidate 3 Hybrid 70/30",
            "current_risk_score": "Candidate 4 Official Composite",
            "map_risk_level": "derived from current_risk_score",
        },
        "weights_expected_paths": [
            "output/models/onset_prod",
            "output/models/gkg_pca",
            "input/processed/features/anchor_embeddings.npz",
        ],
    }

    with conn.cursor() as cur:
        cur.execute(
            """
            update model_versions
            set is_active = false
            where target_name = %s and horizon_days = %s and model_version_id <> %s
            """,
            (target_name, horizon_days, model_version_id),
        )
        cur.execute(
            """
            insert into model_versions (
              model_version_id, model_name, target_name, horizon_days,
              artifact_uri, feature_schema_uri, metrics_json, is_active
            )
            values (%s, %s, %s, %s, %s, %s, %s::jsonb, true)
            on conflict (model_version_id) do update set
              model_name = excluded.model_name,
              target_name = excluded.target_name,
              horizon_days = excluded.horizon_days,
              artifact_uri = excluded.artifact_uri,
              feature_schema_uri = excluded.feature_schema_uri,
              metrics_json = excluded.metrics_json,
              is_active = true
            """,
            (
                model_version_id,
                model_name,
                target_name,
                horizon_days,
                artifact_uri,
                feature_schema_uri,
                json.dumps(metrics, ensure_ascii=False),
            ),
        )
        cur.execute(
            """
            insert into alert_thresholds (
              model_version_id, iso3, horizon_days,
              low_threshold, medium_threshold, high_threshold, is_active
            )
            values (%s, null, %s, %s, %s, %s, true)
            """,
            (
                model_version_id,
                horizon_days,
                float(env("ONSET_LOW_THRESHOLD", "25")),
                float(env("ONSET_MEDIUM_THRESHOLD", "40")),
                float(env("ONSET_HIGH_THRESHOLD", "70")),
            ),
        )
    conn.commit()


def alert_level_from_score(score: float, medium: float, high: float) -> str:
    if score >= high:
        return "HIGH"
    if score >= medium:
        return "MEDIUM"
    return "LOW"


def map_risk_level(score: float) -> str:
    if score >= 70:
        return "ALERT"
    if score >= 50:
        return "WARNING"
    if score >= 25:
        return "WATCH"
    return "STABLE"


def percentile_score(series: pd.Series) -> pd.Series:
    if len(series) <= 1:
        return pd.Series([100.0] * len(series), index=series.index)
    return series.rank(method="average", pct=True) * 100


def sigmoid_score(value: pd.Series | float) -> pd.Series | float:
    return 100 / (1 + np.exp(-value))


def latest_onset_thresholds(conn: psycopg.Connection, model_version_id: str, horizon_days: int) -> tuple[float, float]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            select medium_threshold, high_threshold
            from alert_thresholds
            where model_version_id = %s
              and horizon_days = %s
              and iso3 is null
              and is_active = true
            order by created_at desc
            limit 1
            """,
            (model_version_id, horizon_days),
        )
        row = cur.fetchone()
    if row:
        return float(row["medium_threshold"]), float(row["high_threshold"])
    return 40.0, 70.0


def query_optional_table(client: bigquery.Client, project_id: str, dataset_id: str, table_name: str) -> pd.DataFrame | None:
    full_table = f"{project_id}.{dataset_id}.{table_name}"
    try:
        client.get_table(full_table)
    except Exception:
        return None
    return client.query(
        f"select * from `{full_table}`",
        job_config=bigquery.QueryJobConfig(use_query_cache=False),
    ).to_dataframe(create_bqstorage_client=False)


def load_latest_model_input_features(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    input_table: str,
) -> pd.DataFrame:
    full_table = f"{project_id}.{dataset_id}.{input_table}"
    query = f"""
    with latest as (
      select max(date) as feature_date
      from `{full_table}`
    )
    select
      country as iso3,
      date as feature_date,
      gcr_gdev_protest_7d_z,
      gcr_gdev_battles_7d_z,
      gcr_gdev_remote_7d_z,
      gcr_gdev_vac_7d_z,
      gcr_gdev_massviol_7d_z,
      gdelt2_quad4_share_7d,
      gdelt_goldstein_mean_7d,
      gkg_article_count_1d,
      gkg_article_count_7d,
      gdev_battles_30d,
      gdev_remote_30d,
      gdev_vac_30d
    from `{full_table}`
    where date = (select feature_date from latest)
    """
    frame = client.query(query, job_config=bigquery.QueryJobConfig(use_query_cache=False)).to_dataframe(
        create_bqstorage_client=False
    )
    frame["feature_date"] = pd.to_datetime(frame["feature_date"]).dt.tz_localize(None)
    return frame


def apply_dashboard_score_policy(
    scores: pd.DataFrame,
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    input_table: str,
) -> pd.DataFrame:
    """Apply the final dashboard policy.

    - onset_alert_score: Candidate 3 Hybrid 70/30
    - watchlist_score: Candidate 3 Hybrid 70/30
    - current_risk_score: Candidate 4 Official Composite
    """
    out = scores.copy()
    out["onset_prob"] = out["raw_score"].astype(float)
    out["probability_score"] = (out["onset_prob"] * 100).clip(lower=0, upper=100)
    out["rank_today"] = out["onset_prob"].rank(ascending=False, method="min").astype(int)
    n = len(out)
    out["rank_score"] = 100.0 if n <= 1 else (1 - (out["rank_today"] - 1) / (n - 1)) * 100
    out["hybrid_score_70_30"] = (0.7 * out["probability_score"] + 0.3 * out["rank_score"]).clip(
        lower=0, upper=100
    )
    out["onset_alert_score"] = out["hybrid_score_70_30"]
    out["watchlist_score"] = out["hybrid_score_70_30"]

    features = load_latest_model_input_features(client, project_id, dataset_id, input_table)
    baseline = query_optional_table(client, project_id, dataset_id, "baseline_scores")
    neighbors = query_optional_table(client, project_id, dataset_id, "country_neighbors")
    latest_features = features.copy()

    if baseline is not None and {"iso3", "b_score"}.issubset(set(baseline.columns)):
        latest_features = latest_features.merge(
            baseline[["iso3", "b_score"]].rename(columns={"b_score": "baseline_B"}),
            on="iso3",
            how="left",
        )
    else:
        baseline_raw = (
            features.assign(
                baseline_raw=lambda d: np.log1p(
                    d[["gdev_battles_30d", "gdev_remote_30d", "gdev_vac_30d"]].fillna(0).sum(axis=1)
                )
            )
            .groupby("iso3", as_index=False)["baseline_raw"]
            .mean()
        )
        baseline_raw["baseline_B"] = percentile_score(baseline_raw["baseline_raw"])
        latest_features = latest_features.merge(baseline_raw[["iso3", "baseline_B"]], on="iso3", how="left")

    latest_features["U_score"] = sigmoid_score(latest_features["gcr_gdev_protest_7d_z"].fillna(0))
    conflict_z = latest_features[
        ["gcr_gdev_battles_7d_z", "gcr_gdev_remote_7d_z", "gcr_gdev_vac_7d_z", "gcr_gdev_massviol_7d_z"]
    ].fillna(0).mean(axis=1)
    latest_features["C_score"] = sigmoid_score(conflict_z)
    quad4_score = percentile_score(latest_features["gdelt2_quad4_share_7d"].fillna(0))
    neg_goldstein_score = percentile_score((-latest_features["gdelt_goldstein_mean_7d"]).fillna(0))
    latest_features["S_score"] = 0.6 * quad4_score + 0.4 * neg_goldstein_score
    denom = latest_features["gkg_article_count_7d"].fillna(0) / 7
    spike = latest_features["gkg_article_count_1d"].fillna(0) / denom.replace(0, np.nan)
    latest_features["I_score"] = ((spike.fillna(1) - 1) * 50).clip(lower=0, upper=100)
    latest_features["C_state"] = latest_features[["U_score", "C_score", "S_score", "I_score"]].mean(axis=1)

    p99 = float(out["onset_prob"].quantile(0.99))
    if p99 <= 0:
        p99 = 1.0
    out["F_spread"] = (out["onset_prob"] * 100 / p99).clip(lower=0, upper=100)

    out = out.merge(
        latest_features[["iso3", "baseline_B", "U_score", "C_score", "S_score", "I_score", "C_state"]],
        on="iso3",
        how="left",
    )

    if neighbors is not None and {"iso3", "neighbor_iso3"}.issubset(set(neighbors.columns)):
        hot_neighbors = latest_features.loc[latest_features["C_score"].ge(50), ["iso3"]].rename(
            columns={"iso3": "neighbor_iso3"}
        )
        hotspot = (
            neighbors.merge(hot_neighbors, on="neighbor_iso3", how="inner")
            .groupby("iso3", as_index=False)
            .size()
            .rename(columns={"size": "hot_neighbor_count"})
        )
        hotspot["hotspot_bonus"] = (1.5 * hotspot["hot_neighbor_count"]).clip(upper=5)
        out = out.merge(hotspot[["iso3", "hotspot_bonus"]], on="iso3", how="left")
    else:
        out["hotspot_bonus"] = 0.0
    out["hotspot_bonus"] = out["hotspot_bonus"].fillna(0)
    out["current_risk_score"] = (
        0.2 * out["baseline_B"].fillna(0)
        + 0.4 * out["C_state"].fillna(0)
        + 0.4 * out["F_spread"].fillna(0)
        + out["hotspot_bonus"]
    ).clip(lower=0, upper=100)
    out["map_risk_level"] = out["current_risk_score"].map(map_risk_level)
    out["score_policy_version"] = SCORE_POLICY_VERSION
    out["raw_score"] = out["onset_alert_score"]
    return out.sort_values("current_risk_score", ascending=False).reset_index(drop=True)


def ensure_score_policy_columns(conn: psycopg.Connection) -> None:
    statements = [
        "alter table latest_risk_scores add column if not exists onset_alert_score double precision",
        "alter table latest_risk_scores add column if not exists watchlist_score double precision",
        "alter table latest_risk_scores add column if not exists current_risk_score double precision",
        "alter table latest_risk_scores add column if not exists map_risk_level text",
        "alter table latest_risk_scores add column if not exists score_policy_version text",
        "alter table risk_score_history add column if not exists onset_alert_score double precision",
        "alter table risk_score_history add column if not exists watchlist_score double precision",
        "alter table risk_score_history add column if not exists current_risk_score double precision",
        "alter table risk_score_history add column if not exists map_risk_level text",
        "alter table risk_score_history add column if not exists score_policy_version text",
    ]
    with conn.cursor() as cur:
        for statement in statements:
            cur.execute(statement)
    conn.commit()


def load_latest_scores(client: bigquery.Client, table_id: str, mapping: dict[str, str | None]) -> pd.DataFrame:
    iso3_col = mapping["iso3"]
    date_col = mapping["feature_date"]
    score_col = mapping["raw_score"]
    alert_col = mapping["alert_level"]
    scored_at_col = mapping["scored_at"]
    model_col = mapping["model_version_id"]
    horizon_col = mapping["horizon_days"]

    select_parts = [
        f"cast(`{iso3_col}` as string) as iso3",
        f"date(`{date_col}`) as feature_date",
        f"cast(`{score_col}` as float64) as raw_score",
    ]
    select_parts.append(f"cast(`{alert_col}` as string) as source_alert_level" if alert_col else "cast(null as string) as source_alert_level")
    select_parts.append(f"timestamp(`{scored_at_col}`) as source_scored_at" if scored_at_col else "current_timestamp() as source_scored_at")
    select_parts.append(f"cast(`{model_col}` as string) as source_model_version_id" if model_col else "cast(null as string) as source_model_version_id")
    select_parts.append(f"cast(`{horizon_col}` as int64) as source_horizon_days" if horizon_col else "cast(null as int64) as source_horizon_days")

    query = f"""
    with latest as (
      select max(date(`{date_col}`)) as feature_date
      from `{table_id}`
    )
    select
      {", ".join(select_parts)}
    from `{table_id}`
    where date(`{date_col}`) = (select feature_date from latest)
    qualify row_number() over (
      partition by cast(`{iso3_col}` as string)
      order by {f'timestamp(`{scored_at_col}`)' if scored_at_col else f'date(`{date_col}`)'} desc
    ) = 1
    order by raw_score desc
    """
    return client.query(query, job_config=bigquery.QueryJobConfig(use_query_cache=False)).to_dataframe(
        create_bqstorage_client=False
    )


def write_scores_to_supabase(conn: psycopg.Connection, scores: pd.DataFrame, feature_source: str) -> uuid.UUID:
    model_version_id = env("ONSET_MODEL_VERSION_ID", DEFAULT_MODEL_VERSION_ID)
    horizon_days = int(env("ONSET_HORIZON_DAYS", str(DEFAULT_HORIZON_DAYS)))
    medium, high = latest_onset_thresholds(conn, model_version_id, horizon_days)
    run_id = uuid.uuid4()
    feature_date = pd.to_datetime(scores["feature_date"].max()).date()

    scores = scores.copy()
    scores["model_version_id"] = model_version_id
    scores["horizon_days"] = scores["source_horizon_days"].fillna(horizon_days).astype(int)
    scores["alert_level"] = scores["source_alert_level"].fillna("")
    blank = scores["alert_level"].eq("")
    scores.loc[blank, "alert_level"] = scores.loc[blank, "raw_score"].map(lambda s: alert_level_from_score(float(s), medium, high))

    with conn.cursor() as cur:
        cur.executemany(
            """
            insert into countries (iso3, country_name, is_active)
            values (%s, %s, true)
            on conflict (iso3) do update set is_active = excluded.is_active
            """,
            [(str(iso3), str(iso3)) for iso3 in scores["iso3"].tolist()],
        )
        cur.execute(
            """
            insert into scoring_runs (
              scoring_run_id, model_version_id, feature_source, feature_date,
              status, row_count
            )
            values (%s, %s, %s, %s, 'running', %s)
            """,
            (run_id, model_version_id, feature_source, feature_date, len(scores)),
        )

        rows = [
            (
                r.iso3,
                int(r.horizon_days),
                pd.to_datetime(r.feature_date).date(),
                r.model_version_id,
                float(r.raw_score),
                None,
                r.alert_level,
                json.dumps(
                    {
                        "score_source": "bigquery.model_scores",
                        "score_policy_version": r.score_policy_version,
                        "onset_prob": round(float(r.onset_prob), 6),
                        "probability_score": round(float(r.probability_score), 6),
                        "rank_score": round(float(r.rank_score), 6),
                        "hybrid_score_70_30": round(float(r.hybrid_score_70_30), 6),
                        "current_risk_score": round(float(r.current_risk_score), 6),
                        "map_risk_level": r.map_risk_level,
                        "components": {
                            "baseline_B": None if pd.isna(r.baseline_B) else round(float(r.baseline_B), 6),
                            "C_state": None if pd.isna(r.C_state) else round(float(r.C_state), 6),
                            "F_spread": None if pd.isna(r.F_spread) else round(float(r.F_spread), 6),
                            "hotspot_bonus": round(float(r.hotspot_bonus), 6),
                        },
                    },
                    ensure_ascii=False,
                ),
                run_id,
                float(r.onset_alert_score),
                float(r.watchlist_score),
                float(r.current_risk_score),
                r.map_risk_level,
                r.score_policy_version,
            )
            for r in scores.itertuples(index=False)
        ]
        cur.executemany(
            """
            insert into risk_score_history (
              iso3, horizon_days, scored_at, feature_date, model_version_id,
              raw_score, calibrated_score, alert_level, scoring_run_id,
              onset_alert_score, watchlist_score, current_risk_score,
              map_risk_level, score_policy_version
            )
            values (%s, %s, now(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (a, b, c, d, e, f, g, i, j, k, l, m, n)
                for a, b, c, d, e, f, g, _h, i, j, k, l, m, n in rows
            ],
        )
        cur.executemany(
            """
            insert into latest_risk_scores (
              iso3, horizon_days, scored_at, feature_date, model_version_id,
              raw_score, calibrated_score, alert_level, top_features,
              scoring_run_id, updated_at,
              onset_alert_score, watchlist_score, current_risk_score,
              map_risk_level, score_policy_version
            )
            values (%s, %s, now(), %s, %s, %s, %s, %s, %s::jsonb, %s, now(), %s, %s, %s, %s, %s)
            on conflict (iso3, horizon_days) do update set
              scored_at = excluded.scored_at,
              feature_date = excluded.feature_date,
              model_version_id = excluded.model_version_id,
              raw_score = excluded.raw_score,
              calibrated_score = excluded.calibrated_score,
              alert_level = excluded.alert_level,
              top_features = excluded.top_features,
              scoring_run_id = excluded.scoring_run_id,
              onset_alert_score = excluded.onset_alert_score,
              watchlist_score = excluded.watchlist_score,
              current_risk_score = excluded.current_risk_score,
              map_risk_level = excluded.map_risk_level,
              score_policy_version = excluded.score_policy_version,
              updated_at = now()
            """,
            rows,
        )
        cur.execute(
            """
            update scoring_runs
            set status = 'success', finished_at = now(), row_count = %s
            where scoring_run_id = %s
            """,
            (len(scores), run_id),
        )
    conn.commit()
    return run_id


def main() -> None:
    load_dotenv()
    project_id, dataset_id, scores_table, input_table = bq_context()
    table_id = f"{project_id}.{dataset_id}.{scores_table}"
    feature_source = f"bigquery://{table_id}"
    client = bigquery.Client(project=project_id)
    columns = table_columns(client, table_id)
    mapping = infer_score_mapping(columns)

    with connect() as conn:
        ensure_onset_model_metadata(conn, project_id, dataset_id, scores_table, input_table)
        ensure_score_policy_columns(conn)
        scores = load_latest_scores(client, table_id, mapping)
        scores = apply_dashboard_score_policy(scores, client, project_id, dataset_id, input_table)
        run_id = write_scores_to_supabase(conn, scores, feature_source)

    print(f"Synced BigQuery model scores to Supabase: {len(scores)} rows")
    print(f"source={feature_source}")
    print(f"scoring_run_id={run_id}")
    print(
        scores.head(10)[
            ["iso3", "feature_date", "onset_alert_score", "watchlist_score", "current_risk_score", "map_risk_level"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
