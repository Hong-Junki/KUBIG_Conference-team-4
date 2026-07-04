"""Telegram/index.html에 이미 baked된 gpt-4.1-mini 요약(eventData, gdeltContextData)을
Supabase public.llm_briefings에 누적 적재한다.

새로운 LLM API 호출은 하지 않는다 — HTML에 이미 구워진 요약 텍스트만 재사용한다.

사용 예:
    python scripts/load_static_briefings.py \\
        --html Telegram/index.html \\
        --limit 10 \\
        --alert-level HIGH \\
        --prompt-version static-export-v1 \\
        --dry-run

환경변수:
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY  (dry-run에서도 risk_score_history 조회에 필요)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

SCRIPT_TAG_RE = re.compile(
    r'<script id="(?P<id>eventData|gdeltContextData|modelScoreData)"[^>]*>'
    r'(?P<body>.*?)</script>',
    re.DOTALL,
)

REQUIRED_BLOCKS = {"eventData", "gdeltContextData", "modelScoreData"}


def extract_embedded_json(html_path: Path) -> dict[str, Any]:
    """Telegram/index.html에서 eventData/gdeltContextData/modelScoreData 3개
    <script type="application/json"> 블록을 추출해 파싱한다."""
    text = html_path.read_text(encoding="utf-8")
    found: dict[str, Any] = {}
    for m in SCRIPT_TAG_RE.finditer(text):
        found[m.group("id")] = json.loads(m.group("body"))
    missing = REQUIRED_BLOCKS - found.keys()
    if missing:
        raise ValueError(f"HTML에서 다음 script 블록을 찾지 못함: {sorted(missing)}")
    return found


def group_events_by_iso3(event_data: list[dict]) -> dict[str, list[dict]]:
    """country가 빈 문자열인 이벤트는 iso3에 연결할 수 없으므로 제외한다."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for ev in event_data:
        iso3 = (ev.get("country") or "").strip().upper()
        if not iso3:
            continue
        grouped[iso3].append(ev)
    return grouped


def build_main_causes(gdelt_ctx: dict | None, events: list[dict]) -> list[dict]:
    causes: list[dict] = []
    if gdelt_ctx:
        for kw in gdelt_ctx.get("top_keywords", []):
            causes.append({"source": "gdelt", "keyword": kw})
    keyword_counter: Counter[str] = Counter()
    event_type_counter: Counter[str] = Counter()
    for ev in events:
        for kw in ev.get("matched_keywords") or []:
            keyword_counter[kw] += 1
        et = ev.get("event_type")
        if et:
            event_type_counter[et] += 1
    for kw, cnt in keyword_counter.most_common():
        causes.append({"source": "telegram", "keyword": kw, "count": cnt})
    for et, cnt in event_type_counter.most_common():
        causes.append({"source": "telegram", "event_type": et, "count": cnt})
    return causes


def build_top_risk_drivers(gdelt_ctx: dict | None, events: list[dict]) -> list[dict]:
    """top_risk_drivers는 NOT NULL 컬럼. 매핑 명세에 없어 main_causes와 동일 근거를
    신호 강도(GDELT 기사량, Telegram severity) 기준으로 재구성한 값이다 — 필요시 조정."""
    drivers: list[dict] = []
    if gdelt_ctx:
        drivers.append({
            "source": "gdelt",
            "driver": "article_volume",
            "gdelt_24h": gdelt_ctx.get("gdelt_24h"),
            "gdelt_7d": gdelt_ctx.get("gdelt_7d"),
        })
        for kw in gdelt_ctx.get("top_keywords", [])[:5]:
            drivers.append({"source": "gdelt", "driver": "keyword", "value": kw})
    top_events = sorted(events, key=lambda e: e.get("severity") or 0, reverse=True)[:5]
    for ev in top_events:
        drivers.append({
            "source": "telegram",
            "driver": "event",
            "event_type": ev.get("event_type"),
            "severity": ev.get("severity"),
            "confidence": ev.get("confidence"),
            "channel": ev.get("channel"),
        })
    return drivers


def build_evidence_summary(gdelt_ctx: dict | None, events: list[dict]) -> dict:
    telegram_summaries = [
        {
            "event_id": ev.get("event_id"),
            "channel": ev.get("channel"),
            "message_time": ev.get("message_time"),
            "ko_summary": ev.get("ko_summary"),
            "ko_raw_summary": ev.get("ko_raw_summary"),
        }
        for ev in events
    ]
    gdelt_summary = None
    if gdelt_ctx:
        gdelt_summary = {
            "ko_summary": gdelt_ctx.get("ko_summary"),
            "top_titles": gdelt_ctx.get("top_titles", []),
        }
    return {"telegram": telegram_summaries, "gdelt": gdelt_summary}


def build_alert_evidence(
    gdelt_ctx: dict | None, events: list[dict], gdelt_generated_at: str | None
) -> dict:
    return {
        "source_snapshot_generated_at": gdelt_generated_at,
        "telegram_events": [
            {
                "event_id": ev.get("event_id"),
                "channel": ev.get("channel"),
                "message_id": ev.get("message_id"),
                "message_time": ev.get("message_time"),
                "url": ev.get("url"),
                "raw_text": ev.get("raw_text"),
                "severity": ev.get("severity"),
                "confidence": ev.get("confidence"),
                "source_reliability": ev.get("source_reliability"),
            }
            for ev in events
        ],
        "gdelt_context": (
            {
                "table": gdelt_ctx.get("table"),
                "gdelt_24h": gdelt_ctx.get("gdelt_24h"),
                "gdelt_7d": gdelt_ctx.get("gdelt_7d"),
                "top_titles": gdelt_ctx.get("top_titles", []),
            }
            if gdelt_ctx
            else None
        ),
    }


def build_cautions(gdelt_ctx: dict | None, events: list[dict]) -> list[dict]:
    cautions: list[dict] = []
    if gdelt_ctx and gdelt_ctx.get("source_limits"):
        cautions.append({"source": "gdelt", "text": gdelt_ctx["source_limits"]})
    seen: set[str] = set()
    for ev in events:
        text = ev.get("ko_raw_source_limits")
        if text and text not in seen:
            seen.add(text)
            cautions.append({"source": "telegram", "channel": ev.get("channel"), "text": text})
    return cautions


def pick_risk_summary(gdelt_ctx: dict | None, events: list[dict]) -> str | None:
    if gdelt_ctx:
        if gdelt_ctx.get("ko_brief"):
            return gdelt_ctx["ko_brief"]
        if gdelt_ctx.get("ko_summary"):
            return gdelt_ctx["ko_summary"]
    for ev in events:
        if ev.get("ko_summary"):
            return ev["ko_summary"]
    return None


def pick_llm_model(gdelt_ctx: dict | None, events: list[dict]) -> str | None:
    if gdelt_ctx and gdelt_ctx.get("summary_model"):
        return gdelt_ctx["summary_model"]
    for ev in events:
        if ev.get("ko_raw_summary_model"):
            return ev["ko_raw_summary_model"]
    return None


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))


def compute_input_hash(
    risk_row: dict,
    risk_summary: str | None,
    main_causes: list[dict],
    evidence_summary: dict,
    alert_evidence: dict,
    cautions: list[dict],
    prompt_version: str,
) -> str:
    payload = {
        "risk_row": {
            "risk_score_id": risk_row["risk_score_id"],
            "iso3": risk_row["iso3"],
            "horizon_days": risk_row["horizon_days"],
            "scored_at": str(risk_row["scored_at"]),
            "feature_date": str(risk_row.get("feature_date")),
            "model_version_id": risk_row["model_version_id"],
            "raw_score": risk_row["raw_score"],
            "calibrated_score": risk_row.get("calibrated_score"),
            "alert_level": risk_row["alert_level"],
            "scoring_run_id": str(risk_row.get("scoring_run_id")),
        },
        "risk_summary": risk_summary,
        "main_causes": main_causes,
        "evidence_summary": evidence_summary,
        "alert_evidence": alert_evidence,
        "cautions": cautions,
        "prompt_version": prompt_version,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def score_value(risk_row: dict) -> float:
    return (
        risk_row["calibrated_score"]
        if risk_row.get("calibrated_score") is not None
        else risk_row["raw_score"]
    )


def get_supabase_client():
    from supabase import Client, create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY 환경변수가 필요합니다.")
    return create_client(url, key)


def fetch_latest_risk_rows(client, horizon_days: int, alert_level: str) -> dict[str, dict]:
    """iso3별 scored_at DESC 최신 1건. Supabase REST에는 DISTINCT ON이 없어
    horizon_days(+alert_level) 조건으로 scored_at 내림차순 전체 조회 후
    파이썬에서 iso3별 첫 번째(=최신) 행만 취한다."""
    query = client.table("risk_score_history").select("*").eq("horizon_days", horizon_days)
    if alert_level and alert_level != "ALL":
        query = query.eq("alert_level", alert_level)
    resp = query.order("scored_at", desc=True).execute()
    rows = resp.data or []
    latest_by_iso3: dict[str, dict] = {}
    for row in rows:
        iso3 = row["iso3"]
        if iso3 not in latest_by_iso3:
            latest_by_iso3[iso3] = row
    return latest_by_iso3


def briefing_exists(client, risk_score_id: int, input_hash: str) -> bool:
    resp = (
        client.table("llm_briefings")
        .select("briefing_id")
        .eq("risk_score_id", risk_score_id)
        .eq("input_hash", input_hash)
        .limit(1)
        .execute()
    )
    return bool(resp.data)


def has_evidence(gdelt_ctx: dict | None, events: list[dict]) -> bool:
    """gdeltContextData에 해당 iso3 항목이 없고 eventData 이벤트도 0건이면
    evidence가 전혀 없는 것으로 간주한다. 이 경우 risk_summary/llm_model이
    fallback 끝까지 None이 되므로 애초에 briefing을 생성하지 않는다."""
    return gdelt_ctx is not None or len(events) > 0


def build_record(
    iso3: str,
    risk_row: dict,
    events: list[dict],
    gdelt_ctx: dict | None,
    gdelt_generated_at: str | None,
    prompt_version: str,
) -> dict:
    risk_summary = pick_risk_summary(gdelt_ctx, events)
    main_causes = build_main_causes(gdelt_ctx, events)
    top_risk_drivers = build_top_risk_drivers(gdelt_ctx, events)
    evidence_summary = build_evidence_summary(gdelt_ctx, events)
    alert_evidence = build_alert_evidence(gdelt_ctx, events, gdelt_generated_at)
    cautions = build_cautions(gdelt_ctx, events)
    llm_model = pick_llm_model(gdelt_ctx, events)

    input_hash = compute_input_hash(
        risk_row, risk_summary, main_causes, evidence_summary, alert_evidence, cautions,
        prompt_version,
    )

    return {
        "risk_score_id": risk_row["risk_score_id"],
        "iso3": iso3,
        "horizon_days": risk_row["horizon_days"],
        "scoring_run_id": risk_row.get("scoring_run_id"),
        "model_version_id": risk_row["model_version_id"],
        "scored_at": risk_row["scored_at"],
        "feature_date": risk_row.get("feature_date"),
        "risk_score": score_value(risk_row),
        "alert_level": risk_row["alert_level"],
        "top_risk_drivers": top_risk_drivers,
        "alert_evidence": alert_evidence,
        "risk_summary": risk_summary,
        "main_causes": main_causes,
        "evidence_summary": evidence_summary,
        "cautions": cautions,
        "llm_provider": "openai",
        "llm_model": llm_model,
        "prompt_version": prompt_version,
        "input_hash": input_hash,
        "generation_status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def run(
    html_path: Path,
    limit: int | None,
    alert_level: str,
    horizon_days: int,
    prompt_version: str,
    dry_run: bool,
    client=None,
) -> dict:
    """핵심 로직. client를 주입받으면(테스트용) get_supabase_client()를 호출하지 않는다."""
    blocks = extract_embedded_json(html_path)
    events_by_iso3 = group_events_by_iso3(blocks["eventData"])
    gdelt_by_iso3 = blocks["gdeltContextData"].get("countries", {})
    gdelt_generated_at = blocks["gdeltContextData"].get("generated_at")
    model_scores = blocks["modelScoreData"].get("countries", {})

    target_isos = sorted(model_scores.keys())
    if limit:
        target_isos = target_isos[:limit]

    if client is None:
        client = get_supabase_client()

    latest_rows = fetch_latest_risk_rows(client, horizon_days, alert_level)

    stats = {
        "created": 0,
        "skipped_duplicate": 0,
        "skipped_no_evidence": 0,
        "missing_risk_row": 0,
        "target": len(target_isos),
    }

    for iso3 in target_isos:
        risk_row = latest_rows.get(iso3)
        if risk_row is None:
            stats["missing_risk_row"] += 1
            print(
                f"[skip] {iso3}: risk_score_history에서 horizon_days={horizon_days}"
                f"{'' if alert_level == 'ALL' else f', alert_level={alert_level}'} 최신 row 없음"
            )
            continue

        events = events_by_iso3.get(iso3, [])
        gdelt_ctx = gdelt_by_iso3.get(iso3)

        if not has_evidence(gdelt_ctx, events):
            stats["skipped_no_evidence"] += 1
            print(f"[skip] {iso3}: Telegram/GDELT evidence 없음")
            continue

        record = build_record(iso3, risk_row, events, gdelt_ctx, gdelt_generated_at, prompt_version)

        if briefing_exists(client, record["risk_score_id"], record["input_hash"]):
            stats["skipped_duplicate"] += 1
            print(
                f"[skip] {iso3}: risk_score_id={record['risk_score_id']} "
                f"이미 동일 input_hash 존재"
            )
            continue

        if dry_run:
            print(
                f"[dry-run] {iso3}: insert 예정 (risk_score_id={record['risk_score_id']}, "
                f"input_hash={record['input_hash'][:12]}..., events={len(events)}, "
                f"alert_level={record['alert_level']}, risk_score={record['risk_score']})"
            )
        else:
            client.table("llm_briefings").insert(record).execute()
            print(f"[insert] {iso3}: risk_score_id={record['risk_score_id']} 적재 완료")
        stats["created"] += 1

    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Telegram/index.html의 baked LLM 요약을 Supabase llm_briefings에 적재"
    )
    ap.add_argument("--html", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="처리할 iso3 개수 상한")
    ap.add_argument(
        "--alert-level", default="ALL",
        help="risk_score_history.alert_level 필터 (예: HIGH). 'ALL'이면 필터 없음",
    )
    ap.add_argument("--horizon-days", type=int, default=7)
    ap.add_argument("--prompt-version", default="static-export-v1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    stats = run(
        html_path=args.html,
        limit=args.limit,
        alert_level=args.alert_level,
        horizon_days=args.horizon_days,
        prompt_version=args.prompt_version,
        dry_run=args.dry_run,
    )

    print(
        f"\n요약: 대상 {stats['target']}개국 | "
        f"{'생성 예정' if args.dry_run else '생성'} {stats['created']} | "
        f"스킵(중복) {stats['skipped_duplicate']} | "
        f"스킵(evidence 없음) {stats['skipped_no_evidence']} | "
        f"risk row 없음 {stats['missing_risk_row']}"
    )


if __name__ == "__main__":
    main()
