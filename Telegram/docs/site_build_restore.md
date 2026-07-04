# Site build and restore notes

This note records the current static-site build flow before major UI redesign work.

## Main local files

- Source builder: `scripts/build_live_osint_site.py`
- Model score export: `scripts/export_model_scores.py`
- Local generated site: `site/live_osint.html`
- Public GitHub Pages files:
  - `Telegram/index.html`
  - `Telegram/live_osint.html`

## Rebuild locally

```powershell
cd "C:\Users\Hong JunKi\Desktop\학회.동아리\KUBIG\Conf_project\New_analyze\telegram_osint"
python scripts\build_live_osint_site.py
python -m http.server 8000
```

Open:

```text
http://localhost:8000/site/live_osint.html
```

## Refresh model scores from BigQuery

```powershell
cd "C:\Users\Hong JunKi\Desktop\학회.동아리\KUBIG\Conf_project\New_analyze\telegram_osint"
python scripts\export_model_scores.py
python scripts\build_live_osint_site.py
```

The service-account JSON must not be copied into the public `Telegram` folder.

## Last known GitHub Pages baseline before redesign

Public Pages repo commit:

```text
1004a89 Clarify score text and unknown signals
```

Public URL:

```text
https://hong-junki.github.io/KUBIG_Conference-team-4/Telegram/?v=1004a89
```

