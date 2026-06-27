"""GKG submit 주기 실행 wrapper.

03_collect --watch 와 별도로, pending 이 남아있는 한 02_submit 을 주기적으로 실행.
OpenAI enqueue cap (3M tok) 회피를 위해 한 사이클에 소량만 제출하고 충분히 대기.

종료 조건:
  pending == 0 (남은 submitted 는 03_collect 가 처리)

CLI:
  python scripts/gkg_embed/04_loop_submit.py
  python scripts/gkg_embed/04_loop_submit.py --limit 10 --interval 900
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

STATE_PATH = Path("output/gkg_embeddings/state.json")
SUBMIT_SCRIPT = Path("scripts/gkg_embed/02_submit.py")


def status_counts() -> dict[str, int]:
    s = json.loads(STATE_PATH.read_text())
    counts = {"done": 0, "pending": 0, "submitted": 0, "failed": 0}
    for v in s["chunks"].values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20, help="한 사이클당 제출할 chunk 수")
    ap.add_argument("--interval", type=int, default=900, help="사이클 간 대기 (초)")
    args = ap.parse_args()

    cycle = 0
    while True:
        cycle += 1
        sc = status_counts()
        print(f"[loop {cycle}] status={sc}", flush=True)

        if sc["pending"] == 0:
            print(f"[loop {cycle}] pending 0 — 종료. submitted={sc['submitted']} 는 03_collect 가 처리.", flush=True)
            break

        try:
            subprocess.run(
                [sys.executable, str(SUBMIT_SCRIPT), "--limit", str(args.limit)],
                check=False,
            )
        except Exception as e:
            print(f"[loop {cycle}] submit 호출 실패: {e}", flush=True)

        print(f"[loop {cycle}] {args.interval}s 대기", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
