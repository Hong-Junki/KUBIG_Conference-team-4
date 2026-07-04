import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GCP_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


def parse_args():
    parser = argparse.ArgumentParser(description="Refresh Telegram/GDELT dashboard data and publish index.html.")
    parser.add_argument("--limit-per-channel", type=int, default=int(os.getenv("LIMIT_PER_CHANNEL", "100")))
    parser.add_argument("--since-days", type=int, default=int(os.getenv("SINCE_DAYS", "30")))
    parser.add_argument("--min-confidence", type=float, default=float(os.getenv("MIN_CONFIDENCE", "0.0")))
    parser.add_argument("--gdelt-days", type=int, default=int(os.getenv("GDELT_DAYS", "7")))
    parser.add_argument("--gdelt-max-countries", type=int, default=int(os.getenv("GDELT_MAX_COUNTRIES", "58")))
    parser.add_argument("--gdelt-top-titles", type=int, default=int(os.getenv("GDELT_TOP_TITLES", "3")))
    parser.add_argument("--telegram-max-events", type=int, default=int(os.getenv("TELEGRAM_SUMMARY_MAX_EVENTS", "120")))
    parser.add_argument("--gdelt-summary-max-countries", type=int, default=int(os.getenv("GDELT_SUMMARY_MAX_COUNTRIES", "40")))
    parser.add_argument("--publish-root", type=Path, default=ROOT, help="Where index.html/live_osint.html should be copied.")
    parser.add_argument("--skip-collect", action="store_true", help="Rebuild from stored SQLite data without Telegram API calls.")
    parser.add_argument("--skip-model-scores", action="store_true")
    parser.add_argument("--skip-gdelt", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


def run_step(name: str, args: list[str], *, optional: bool = False) -> bool:
    print(f"\n=== {name} ===", flush=True)
    completed = subprocess.run([sys.executable, *args], cwd=ROOT)
    if completed.returncode == 0:
        return True
    if optional:
        print(f"optional_step_failed: {name} exit={completed.returncode}", flush=True)
        return False
    raise SystemExit(completed.returncode)


def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def has_gcp_credentials() -> bool:
    value = DEFAULT_GCP_CREDENTIALS
    return bool(value and Path(value).exists())


def publish_html(publish_root: Path) -> None:
    source = ROOT / "site" / "live_osint.html"
    if not source.exists():
        raise SystemExit(f"site html not found: {source}")
    publish_root.mkdir(parents=True, exist_ok=True)
    for name in ["index.html", "live_osint.html"]:
        target = publish_root / name
        shutil.copyfile(source, target)
        print(f"published: {target}", flush=True)


def main():
    args = parse_args()
    since = str(args.since_days)

    if not args.skip_collect:
        collect_args = [
            "scripts/collect_telegram_osint.py",
            "--limit-per-channel",
            str(args.limit_per_channel),
            "--since-days",
            since,
        ]
        if args.demo:
            collect_args.append("--demo")
        run_step("collect telegram", collect_args)

    run_step("reprocess events", ["scripts/reprocess_live_events.py"])
    run_step(
        "export telegram events",
        [
            "scripts/export_live_osint.py",
            "--since-days",
            since,
            "--min-confidence",
            str(args.min_confidence),
        ],
    )

    if not args.skip_model_scores:
        if has_gcp_credentials():
            run_step(
                "export model scores",
                ["scripts/export_model_scores.py", "--credentials", str(Path(DEFAULT_GCP_CREDENTIALS).resolve())],
                optional=True,
            )
        else:
            print("skip model scores: GOOGLE_APPLICATION_CREDENTIALS is not set or file does not exist.", flush=True)

    if not args.skip_gdelt:
        if has_gcp_credentials():
            run_step(
                "export gdelt context",
                [
                    "scripts/export_gdelt_context.py",
                    "--credentials",
                    str(Path(DEFAULT_GCP_CREDENTIALS).resolve()),
                    "--days",
                    str(args.gdelt_days),
                    "--top-titles",
                    str(args.gdelt_top_titles),
                    "--max-countries",
                    str(args.gdelt_max_countries),
                ],
                optional=True,
            )
        else:
            print("skip gdelt context: GOOGLE_APPLICATION_CREDENTIALS is not set or file does not exist.", flush=True)

    if not args.skip_llm:
        if has_openai_key():
            run_step(
                "enrich gdelt summaries",
                [
                    "scripts/enrich_gdelt_summaries.py",
                    "--max-countries",
                    str(args.gdelt_summary_max_countries),
                ],
                optional=True,
            )
            run_step(
                "enrich telegram summaries",
                [
                    "scripts/enrich_telegram_summaries.py",
                    "--max-events",
                    str(args.telegram_max_events),
                ],
                optional=True,
            )
        else:
            print("skip llm summaries: OPENAI_API_KEY is not set.", flush=True)

    run_step("build site", ["scripts/build_live_osint_site.py"])
    run_step("audit", ["scripts/audit_live_events.py", "--fresh-days", since], optional=True)
    publish_html(args.publish_root.resolve())
    print("\nrefresh done.", flush=True)


if __name__ == "__main__":
    main()
