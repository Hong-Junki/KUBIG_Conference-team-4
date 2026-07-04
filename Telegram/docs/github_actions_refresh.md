# GitHub Actions Dashboard Refresh

This workflow refreshes the Telegram dashboard HTML on a 15-minute schedule.

Flow:

```text
Telegram collection
-> event reprocessing/export
-> BigQuery model score export
-> BigQuery GDELT context export
-> OpenAI Korean summaries
-> site/live_osint.html build
-> index.html and live_osint.html update
-> commit to main
```

When `Telegram/index.html` is committed to `main`, the Supabase ingestion workflow owned by the other team can read the embedded `eventData`, `gdeltContextData`, and `modelScoreData`.

## Required GitHub Secrets

Set these in the GitHub repository:

| Secret | Purpose |
|---|---|
| `TELEGRAM_API_ID` | Telegram API ID |
| `TELEGRAM_API_HASH` | Telegram API hash |
| `TELEGRAM_STRING_SESSION` | Non-interactive Telethon login session |
| `OPENAI_API_KEY` | Generates Korean summaries already embedded in `index.html` |
| `GCP_SERVICE_ACCOUNT_JSON` | Full service-account JSON for BigQuery reads |

Optional repository variable:

| Variable | Default |
|---|---|
| `OPENAI_SUMMARY_MODEL` | `gpt-4.1-mini` |

## Create `TELEGRAM_STRING_SESSION`

Run locally once:

```powershell
pip install -r requirements.txt
$env:TELEGRAM_API_ID="123456"
$env:TELEGRAM_API_HASH="your_api_hash"
python scripts\create_telegram_string_session.py
```

Follow the Telegram login prompt, then copy the printed session string into GitHub Secrets as `TELEGRAM_STRING_SESSION`.

Do not commit `.session`, `.env.local`, service-account JSON, or the printed string session.

## Workflow Location

GitHub only recognizes workflows under the repository root:

```text
.github/workflows/telegram-dashboard-refresh.yml
```

If this Telegram workspace is copied into the upstream repository as `Telegram/`, keep the workflow file at the upstream repository root, not inside `Telegram/.github/`.

The workflow auto-detects whether the scripts are in `Telegram/` or in the repository root.

## Schedule

The schedule is:

```yaml
cron: "*/15 * * * *"
```

GitHub Actions schedules are not guaranteed to start at the exact minute, but this is the intended 15-minute refresh cadence.

## What Gets Committed

The workflow commits only when files changed:

- `Telegram/index.html`
- `Telegram/live_osint.html`
- `Telegram/site/live_osint.html`
- `Telegram/artifacts/live_osint/live_events.db`
- `Telegram/artifacts/live_osint/live_events.json`
- `Telegram/artifacts/live_osint/gdelt_context.json`
- `Telegram/artifacts/live_osint/model_scores.json`
- `Telegram/artifacts/live_osint/audit_report.md`

Committing the SQLite DB preserves collection state across scheduled runs.
