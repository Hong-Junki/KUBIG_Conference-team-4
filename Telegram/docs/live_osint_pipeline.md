# Live Telegram OSINT Pipeline

This folder contains a small operational pipeline for showing conflict-related Telegram signals on a dashboard.

## What It Does

1. Collect messages from an allowlist of public or authorized Telegram channels.
2. Store raw messages in SQLite.
3. Extract conflict-related signals with a transparent keyword/geography heuristic.
4. Export dashboard-ready JSON.
5. Build a static HTML feed that can be embedded or replaced by a later frontend.

This is separate from the model training data. The feed should be treated as an unverified monitoring layer.

## Demo Run

Run this first to verify the pipeline without Telegram credentials.

```powershell
python scripts\collect_telegram_osint.py --demo
python scripts\reprocess_live_events.py
python scripts\export_live_osint.py
python scripts\build_live_osint_site.py
python scripts\audit_live_events.py
```

Or run all steps at once:

```powershell
python scripts\run_live_pipeline.py --demo
```

Outputs:

- `artifacts/live_osint/live_events.db`
- `artifacts/live_osint/live_events.json`
- `site/live_osint.html`
- `artifacts/live_osint/audit_report.md`

## Real Telegram Run

1. Install dependencies.

```powershell
pip install -r requirements.txt
```

2. Edit `config/telegram_channels.json`.

Only add public channels or channels where we have permission to monitor. Keep experimental channels disabled until reviewed.

3. Set Telegram API credentials.

```powershell
$env:TELEGRAM_API_ID="123456"
$env:TELEGRAM_API_HASH="your_api_hash"
$env:TELEGRAM_SESSION="kubig_conflict_monitor"
```

If your IDE terminal and script runner do not share environment variables, create `telegram_osint/.env.local` instead:

```text
TELEGRAM_API_ID=123456
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_SESSION=kubig_conflict_monitor
```

`.env.local` is ignored by git.

4. Collect and export.

```powershell
python scripts\collect_telegram_osint.py --limit-per-channel 100
python scripts\reprocess_live_events.py
python scripts\export_live_osint.py
python scripts\build_live_osint_site.py
python scripts\audit_live_events.py
```

Or run all steps at once:

```powershell
python scripts\run_live_pipeline.py --limit-per-channel 100 --since-days 14
```

The first Telethon login may ask for a phone login code in the terminal.

## Files

- `config/telegram_channels.json`: monitored channel allowlist.
- `src/live_osint/telegram_client.py`: Telethon collection layer.
- `src/live_osint/extraction.py`: conflict keyword, country, location, severity, and confidence extraction.
- `src/live_osint/storage.py`: SQLite schema and upsert helpers.
- `scripts/collect_telegram_osint.py`: ingestion CLI.
- `scripts/export_live_osint.py`: JSON export CLI.
- `scripts/build_live_osint_site.py`: static dashboard feed builder.
- `scripts/audit_live_events.py`: quality audit report builder.
- `scripts/reprocess_live_events.py`: re-runs extraction for stored raw messages after changing extraction logic.
- `scripts/run_live_pipeline.py`: runs collect, reprocess, export, site build, and audit in order.

## Current Limitations

- Keyword extraction is intentionally simple and auditable; it is not a verified event database.
- Location inference uses a small alias table and should be expanded before production use.
- Duplicates across channels are not semantically merged yet.
- Telegram terms, channel permissions, privacy, and source safety need review before deployment.
