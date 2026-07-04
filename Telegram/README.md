# Telegram OSINT Monitor

This is the live Telegram monitoring workspace for the conflict early-warning dashboard.

## Quick Start

```powershell
pip install -r requirements.txt
python scripts\run_live_pipeline.py --limit-per-channel 50 --since-days 14
```

Outputs:

- `artifacts/live_osint/live_events.db`
- `artifacts/live_osint/live_events.json`
- `site/live_osint.html`
- `artifacts/live_osint/audit_report.md`

## Credentials

Use environment variables:

```powershell
$env:TELEGRAM_API_ID="123456"
$env:TELEGRAM_API_HASH="your_api_hash"
$env:TELEGRAM_SESSION="kubig_conflict_monitor"
```

Or create `.env.local` in this folder:

```text
TELEGRAM_API_ID=123456
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_SESSION=kubig_conflict_monitor
```

Do not commit `.env.local` or `.session` files.

## Useful Commands

Rebuild from the existing DB without contacting Telegram:

```powershell
python scripts\run_live_pipeline.py --skip-collect
```

Use sample messages only:

```powershell
python scripts\run_live_pipeline.py --demo
```

## GitHub Actions Refresh

For the 15-minute GitHub Actions automation that refreshes `index.html`, see `docs/github_actions_refresh.md`.
