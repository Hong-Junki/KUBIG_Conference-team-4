# VPS + Cron Deployment

This guide runs the Telegram OSINT pipeline on a VPS every 12 hours and serves the generated static dashboard.

## 1. Server Layout

Recommended path:

```bash
/opt/kubig-telegram-osint
```

Expected files:

```text
/opt/kubig-telegram-osint
+-- .env.local
+-- kubig_conflict_monitor.session
+-- logs/
+-- scripts/run_live_pipeline_job.sh
+-- artifacts/live_osint/live_events.json
+-- site/live_osint.html
```

Keep `.env.local` and `*.session` files private. Do not commit or share them.

## 2. Install

```bash
cd /opt
git clone <repo-url> kubig-telegram-osint
cd /opt/kubig-telegram-osint
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
chmod +x scripts/run_live_pipeline_job.sh
```

Create `.env.local`:

```bash
cat > .env.local <<'EOF'
TELEGRAM_API_ID=123456
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_SESSION=kubig_conflict_monitor
EOF
```

Copy the existing Telegram session file into the app directory:

```bash
cp /secure/source/kubig_conflict_monitor.session /opt/kubig-telegram-osint/
chmod 600 /opt/kubig-telegram-osint/.env.local
chmod 600 /opt/kubig-telegram-osint/kubig_conflict_monitor.session
```

## 3. Test Once

```bash
cd /opt/kubig-telegram-osint
APP_DIR=/opt/kubig-telegram-osint PYTHON_BIN=/opt/kubig-telegram-osint/.venv/bin/python \
  scripts/run_live_pipeline_job.sh
```

Expected outputs:

```text
artifacts/live_osint/live_events.db
artifacts/live_osint/live_events.json
site/live_osint.html
artifacts/live_osint/audit_report.md
```

## 4. Cron: Every 12 Hours

Open crontab:

```bash
crontab -e
```

Add:

```cron
0 */12 * * * APP_DIR=/opt/kubig-telegram-osint PYTHON_BIN=/opt/kubig-telegram-osint/.venv/bin/python /opt/kubig-telegram-osint/scripts/run_live_pipeline_job.sh >> /opt/kubig-telegram-osint/logs/pipeline.log 2>&1
```

This runs at minute 0 every 12 hours.

## 5. Serve With Nginx

Example site config:

```nginx
server {
    listen 80;
    server_name your-domain.example;

    root /opt/kubig-telegram-osint/site;
    index live_osint.html;

    location / {
        try_files $uri $uri/ /live_osint.html;
    }
}
```

Reload:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

## 6. Operations

Check logs:

```bash
tail -n 100 /opt/kubig-telegram-osint/logs/pipeline.log
```

Run manually:

```bash
APP_DIR=/opt/kubig-telegram-osint PYTHON_BIN=/opt/kubig-telegram-osint/.venv/bin/python \
  /opt/kubig-telegram-osint/scripts/run_live_pipeline_job.sh
```

Rebuild without Telegram collection:

```bash
cd /opt/kubig-telegram-osint
.venv/bin/python scripts/run_live_pipeline.py --skip-collect --since-days 14
```

## 7. Default Parameters

The job defaults to:

```text
LIMIT_PER_CHANNEL=100
SINCE_DAYS=14
```

Override in cron if needed:

```cron
0 */12 * * * APP_DIR=/opt/kubig-telegram-osint PYTHON_BIN=/opt/kubig-telegram-osint/.venv/bin/python LIMIT_PER_CHANNEL=150 SINCE_DAYS=14 /opt/kubig-telegram-osint/scripts/run_live_pipeline_job.sh >> /opt/kubig-telegram-osint/logs/pipeline.log 2>&1
```
