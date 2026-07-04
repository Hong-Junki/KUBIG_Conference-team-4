import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full Telegram OSINT pipeline.")
    parser.add_argument("--limit-per-channel", type=int, default=50)
    parser.add_argument("--since-days", type=int, default=30)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--skip-collect", action="store_true", help="Skip Telegram collection and rebuild from stored DB.")
    parser.add_argument("--demo", action="store_true", help="Use sample messages for collection.")
    return parser.parse_args()


def run_step(name: str, args: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    completed = subprocess.run([sys.executable, *args], cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


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
        run_step("collect", collect_args)

    run_step("reprocess", ["scripts/reprocess_live_events.py"])
    run_step(
        "export",
        [
            "scripts/export_live_osint.py",
            "--since-days",
            since,
            "--min-confidence",
            str(args.min_confidence),
        ],
    )
    run_step("build site", ["scripts/build_live_osint_site.py"])
    run_step("audit", ["scripts/audit_live_events.py", "--fresh-days", since])

    print("\nDone.", flush=True)
    print(f"HTML: {ROOT / 'site' / 'live_osint.html'}", flush=True)
    print(f"JSON: {ROOT / 'artifacts' / 'live_osint' / 'live_events.json'}", flush=True)
    print(f"Audit: {ROOT / 'artifacts' / 'live_osint' / 'audit_report.md'}", flush=True)


if __name__ == "__main__":
    main()
