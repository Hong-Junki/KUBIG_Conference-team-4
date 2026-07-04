# Dashboard Scoring Policy

## Final Decision

The dashboard uses two different score concepts instead of forcing one score to serve every purpose.

| Dashboard use case | Selected score | Candidate |
|---|---|---|
| Onset early-warning score | `onset_alert_score` | Candidate 3: Hybrid 70/30 |
| Dashboard watchlist | `watchlist_score` | Candidate 3: Hybrid 70/30 |
| Map color / country risk tier | `current_risk_score` | Candidate 4: Official Composite |

## Candidate 3: Hybrid 70/30

Candidate 3 combines the model probability and the daily rank score.

```text
probability_score = onset_prob * 100
rank_score = daily percentile/rank score from onset_prob
hybrid_score_70_30 = 0.7 * probability_score + 0.3 * rank_score
```

This is used for:

- `onset_alert_score`
- `watchlist_score`

Reason:

- It preserves the onset model signal.
- It avoids the overly narrow raw-probability scale.
- It is less noisy than using pure daily rank.
- In the one-year backtest, it gave a better precision/recall balance than pure rank.

## Candidate 4: Official Composite

Candidate 4 combines baseline risk, current state, ML onset signal, and hotspot context.

```text
current_risk_score = 0.2 * B + 0.4 * C_state + 0.4 * F + hotspot
```

Where:

| Component | Meaning |
|---|---|
| `B` | Country baseline risk from `baseline_scores.b_score` |
| `C_state` | Current state score from U/C/S/I components |
| `F` | Spread onset probability, mapped to 0-100 |
| `hotspot` | Neighbor-country hotspot bonus from `country_neighbors` |

This is used for:

- map color
- country risk tier
- current risk monitoring
- briefing priority

Reason:

- It is better for current/ongoing risk monitoring than strict onset detection.
- It incorporates baseline and current situation rather than only the model probability.

## Backtest Summary

Evaluation window: `2024-04-01` to `2025-03-31`

Target: `y_onset`

| Candidate | Precision@5 | Lift@5 | Precision@10 | Lift@10 | Alerts/day at score >= 70 | Alert precision | Alert recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Candidate 1 Probability | 0.0685 | 6.9378 | 0.0427 | 4.3292 | 2.2552 | 0.1333 | 0.4115 |
| Candidate 2 Rank | 0.0685 | 6.9378 | 0.0427 | 4.3292 | 18.0000 | 0.0263 | 0.8278 |
| Candidate 3 Hybrid 70/30 | 0.0685 | 6.9378 | 0.0427 | 4.3292 | 6.7370 | 0.0549 | 0.6459 |
| Candidate 4 Official Composite | 0.0104 | 1.0545 | 0.0110 | 1.1100 | 1.9095 | 0.0190 | 0.0383 |

Interpretation:

- Candidate 3 is selected for early-warning/watchlist because it balances alert volume and recall.
- Candidate 4 is not selected for strict onset alerts because it dilutes onset signal with baseline and current-state components.
- Candidate 4 is selected for map/current risk because it is designed to represent overall risk, not only new onset.

## Supabase Columns

The sync job writes these service-facing columns:

| Column | Meaning |
|---|---|
| `raw_score` | Backward-compatible score, set to `onset_alert_score` |
| `onset_alert_score` | Candidate 3 Hybrid 70/30 |
| `watchlist_score` | Candidate 3 Hybrid 70/30 |
| `current_risk_score` | Candidate 4 Official Composite |
| `map_risk_level` | `STABLE`, `WATCH`, `WARNING`, `ALERT` from `current_risk_score` |
| `score_policy_version` | Current scoring policy identifier |

## Scheduled Sync

GitHub Actions workflow:

```text
.github/workflows/sync_scores_to_supabase.yml
```

Schedule:

```text
*/15 * * * *
```

The workflow reads latest BigQuery `model_scores` and `model_input`, computes dashboard scores, and writes them to Supabase.

Required GitHub Secrets:

| Secret | Purpose |
|---|---|
| `GOOGLE_CLOUD_PROJECT` | BigQuery project ID |
| `GCP_SERVICE_ACCOUNT_JSON` | Service account JSON for BigQuery access |
| `DATABASE_URL` | Supabase/Postgres connection URL |
