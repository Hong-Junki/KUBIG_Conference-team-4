-- Final dashboard scoring policy migration.
-- Policy:
--   onset_alert_score = Candidate 3 Hybrid 70/30
--   watchlist_score = Candidate 3 Hybrid 70/30
--   current_risk_score = Candidate 4 Official Composite
--   map_risk_level = tier derived from current_risk_score

alter table latest_risk_scores
  add column if not exists onset_alert_score double precision,
  add column if not exists watchlist_score double precision,
  add column if not exists current_risk_score double precision,
  add column if not exists map_risk_level text,
  add column if not exists score_policy_version text;

alter table risk_score_history
  add column if not exists onset_alert_score double precision,
  add column if not exists watchlist_score double precision,
  add column if not exists current_risk_score double precision,
  add column if not exists map_risk_level text,
  add column if not exists score_policy_version text;

create or replace view dashboard_latest_risk as
select
  l.iso3,
  c.country_name,
  c.region,
  l.horizon_days,
  l.scored_at,
  l.feature_date,
  l.raw_score,
  l.calibrated_score,
  l.alert_level,
  l.onset_alert_score,
  l.watchlist_score,
  l.current_risk_score,
  l.map_risk_level,
  l.score_policy_version,
  l.model_version_id,
  l.top_features
from latest_risk_scores l
left join countries c using (iso3)
where c.is_active is distinct from false;
