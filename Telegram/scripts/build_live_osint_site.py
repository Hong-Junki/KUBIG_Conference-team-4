import argparse
import html
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.live_osint.extraction import COUNTRY_CENTROIDS

DEFAULT_EVENTS = ROOT / "artifacts" / "live_osint" / "live_events.json"
DEFAULT_GDELT_CONTEXT = ROOT / "artifacts" / "live_osint" / "gdelt_context.json"
DEFAULT_MODEL_SCORES = ROOT / "artifacts" / "live_osint" / "model_scores.json"
DEFAULT_OUT = ROOT / "site" / "live_osint.html"

COUNTRY_KO_NAMES = {
    "AFG": "아프가니스탄",
    "ARM": "아르메니아",
    "AZE": "아제르바이잔",
    "BFA": "부르키나파소",
    "BGD": "방글라데시",
    "CAF": "중앙아프리카공화국",
    "CIV": "코트디부아르",
    "CMR": "카메룬",
    "COD": "콩고민주공화국",
    "COL": "콜롬비아",
    "DZA": "알제리",
    "ECU": "에콰도르",
    "EGY": "이집트",
    "ERI": "에리트레아",
    "ETH": "에티오피아",
    "GIN": "기니",
    "GNB": "기니비사우",
    "GTM": "과테말라",
    "HND": "온두라스",
    "HTI": "아이티",
    "IDN": "인도네시아",
    "IND": "인도",
    "IRN": "이란",
    "IRQ": "이라크",
    "ISR": "이스라엘",
    "KEN": "케냐",
    "KGZ": "키르기스스탄",
    "KWT": "쿠웨이트",
    "LBN": "레바논",
    "LBY": "리비아",
    "LTU": "리투아니아",
    "MDG": "마다가스카르",
    "MEX": "멕시코",
    "MLI": "말리",
    "MMR": "미얀마",
    "MOZ": "모잠비크",
    "NER": "니제르",
    "NGA": "나이지리아",
    "PAK": "파키스탄",
    "PHL": "필리핀",
    "PSE": "팔레스타인",
    "ROU": "루마니아",
    "RUS": "러시아",
    "SAU": "사우디아라비아",
    "SDN": "수단",
    "SEN": "세네갈",
    "SLE": "시에라리온",
    "SOM": "소말리아",
    "SSD": "남수단",
    "SYR": "시리아",
    "TCD": "차드",
    "TGO": "토고",
    "THA": "태국",
    "TJK": "타지키스탄",
    "TUN": "튀니지",
    "TUR": "튀르키예",
    "UGA": "우간다",
    "UKR": "우크라이나",
    "USA": "미국",
    "VEN": "베네수엘라",
    "YEM": "예멘",
    "ZWE": "짐바브웨",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a static live OSINT feed page.")
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--gdelt-context", type=Path, default=DEFAULT_GDELT_CONTEXT)
    parser.add_argument("--model-scores", type=Path, default=DEFAULT_MODEL_SCORES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def badge(confidence: float) -> str:
    if confidence >= 0.75:
        return "likely"
    if confidence >= 0.55:
        return "unverified"
    return "raw"


def option_tags(values: list[str], label: str, labels: dict[str, str] | None = None) -> str:
    opts = [f'<option value="">{html.escape(label)}</option>']
    for value in values:
        option_label = (labels or COUNTRY_KO_NAMES).get(value, value)
        opts.append(f'<option value="{html.escape(value)}">{html.escape(option_label)}</option>')
    return "\n".join(opts)


def stat(label: str, value: object) -> str:
    return f'<div class="stat"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>'


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {"generated_at": None, "source": "missing", "countries": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def type_label(event_type: str) -> str:
    labels = {
        "strike": "공습/미사일·드론 공격",
        "shelling_explosion": "포격/폭발",
        "armed_clash": "교전",
        "civil_unrest": "시위/소요",
        "military_movement": "군사 이동",
        "conflict_signal": "충돌 관련 신호",
    }
    return labels.get(event_type, "충돌 관련 신호")


def korean_summary(ev: dict) -> str:
    country = ev.get("country") or "국가 미상"
    location = ev.get("location_name") or country
    event_type = type_label(ev.get("event_type") or "")
    severity = float(ev.get("severity") or 0)
    confidence = float(ev.get("confidence") or 0)
    keywords = ", ".join(ev.get("matched_keywords") or [])
    confidence_label = "높은 신뢰도" if confidence >= 0.75 else "추가 확인 필요"
    severity_label = "높은 심각도" if severity >= 0.75 else "중간 이하 심각도"
    sentence = f"{location}({country})에서 {event_type}으로 분류된 텔레그램 신호입니다."
    detail = f"{severity_label}, {confidence_label}이며"
    if keywords:
        detail += f" 감지 키워드는 {keywords}입니다."
    else:
        detail += " 감지 키워드는 제한적입니다."
    return f"{sentence} {detail}"


def raw_korean_summary(ev: dict) -> str:
    text = str(ev.get("raw_text") or ev.get("summary") or "")
    lowered = text.lower()
    country = ev.get("country") or "국가 미상"
    location = ev.get("location_name") or country

    actions = []
    action_terms = [
        ("airstrike", "공습"),
        ("missile", "미사일 공격"),
        ("drone", "드론 관련 공격"),
        ("shelling", "포격"),
        ("explosion", "폭발"),
        ("attack", "공격"),
        ("strike", "타격"),
        ("clash", "교전"),
        ("battle", "전투"),
        ("troop", "병력 이동"),
        ("evacuation", "대피"),
        ("protest", "항의/시위"),
    ]
    for term, label in action_terms:
        if term in lowered and label not in actions:
            actions.append(label)

    casualty_bits = []
    casualty_patterns = [
        (r"\b(\d+)\s+(?:people\s+)?killed\b", "사망"),
        (r"\b(\d+)\s+(?:people\s+)?dead\b", "사망"),
        (r"\b(\d+)\s+(?:people\s+)?injured\b", "부상"),
        (r"\b(\d+)\s+(?:people\s+)?wounded\b", "부상"),
        (r"\bkill(?:ed|s)?\s+(?:over\s+|at\s+least\s+)?(\d+)\b", "사망"),
        (r"\bdead\s+(?:and\s+)?(?:over\s+|at\s+least\s+)?(\d+)\b", "사망"),
        (r"\binjur(?:ed|es|e)\s+(?:over\s+|at\s+least\s+)?(\d+)\b", "부상"),
        (r"\bwound(?:ed|s)?\s+(?:over\s+|at\s+least\s+)?(\d+)\b", "부상"),
    ]
    for pattern, label in casualty_patterns:
        for match in re.findall(pattern, lowered):
            bit = f"{label} {match}명"
            if bit not in casualty_bits:
                casualty_bits.append(bit)

    if actions:
        first = f"원문은 {location}({country})와 관련해 {', '.join(actions[:3])} 정황을 전하고 있습니다."
    else:
        first = f"원문은 {location}({country}) 관련 충돌 신호를 전하고 있습니다."

    if casualty_bits:
        second = f"본문에는 {', '.join(casualty_bits[:3])} 등 피해 규모 표현이 포함되어 있습니다."
    else:
        second = "구체적인 사상자 수는 원문에서 명확히 추출되지 않았습니다."
    return f"{first} {second}"


def time_window_count(events: list[dict], hours: int) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return sum(1 for ev in events if (parse_dt(ev.get("message_time")) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff)


def main():
    args = parse_args()
    payload = json.loads(args.events.read_text(encoding="utf-8"))
    gdelt_context = load_json_if_exists(args.gdelt_context)
    model_scores = load_json_if_exists(args.model_scores)
    events = payload.get("events", [])
    model_countries = set((model_scores.get("countries") or {}).keys())

    gdelt_countries = {
        country
        for country, item in (gdelt_context.get("countries") or {}).items()
        if int(item.get("gdelt_7d") or 0) > 0 or int(item.get("gdelt_24h") or 0) > 0
    }
    event_countries = {ev.get("country") for ev in events if ev.get("country") and ev.get("country") != "UNK"}
    countries = sorted((event_countries | gdelt_countries | model_countries) - {"UNK"})
    channels = sorted({ev.get("channel") or "unknown" for ev in events})
    event_types = sorted({ev.get("event_type") or "signal" for ev in events})
    precision_counts = Counter(ev.get("location_precision") or "missing" for ev in events)
    country_counts = Counter(ev.get("country") for ev in events if ev.get("country") and ev.get("country") != "UNK")
    keyword_counts = Counter()
    for ev in events:
        if not ev.get("ko_summary"):
            ev["ko_summary"] = korean_summary(ev)
        if not ev.get("ko_raw_summary"):
            ev["ko_raw_summary"] = raw_korean_summary(ev)
        keyword_counts.update(ev.get("matched_keywords") or [])
    top_country = country_counts.most_common(1)[0][0] if country_counts else "-"
    top_country = COUNTRY_KO_NAMES.get(top_country, top_country)
    top_keyword = keyword_counts.most_common(1)[0][0] if keyword_counts else "-"
    top_model_country = "-"
    top_model_score = "-"
    top_risk_items = []
    if model_scores.get("countries"):
        top_risk_items = sorted(
            (model_scores.get("countries") or {}).values(),
            key=lambda item: float(item.get("risk_score") or 0),
            reverse=True,
        )[:10]
        top_model = top_risk_items[0]
        top_model_country = COUNTRY_KO_NAMES.get(top_model.get("country") or "", top_model.get("country") or "-")
        top_model_score = f"{float(top_model.get('risk_score') or 0):.1f}"

    stats = "\n".join(
        [
            stat("모델 기준일", next((item.get("date") for item in (model_scores.get("countries") or {}).values() if item.get("date")), "-")),
            stat("최고 위험국", f"{top_model_country} {top_model_score}"),
            stat("주요 키워드", top_keyword),
        ]
    )
    event_data = json.dumps(events, ensure_ascii=False).replace("</", "<\\/")
    gdelt_context_data = json.dumps(gdelt_context, ensure_ascii=False).replace("</", "<\\/")
    model_scores_data = json.dumps(model_scores, ensure_ascii=False).replace("</", "<\\/")
    top_risk_data = json.dumps(top_risk_items, ensure_ascii=False).replace("</", "<\\/")
    country_ko_names_data = json.dumps(COUNTRY_KO_NAMES, ensure_ascii=False).replace("</", "<\\/")
    country_centroid_data = json.dumps(COUNTRY_CENTROIDS, ensure_ascii=False).replace("</", "<\\/")

    cards = []
    for ev in events:
        kw = ", ".join(ev.get("matched_keywords", []))
        url = ev.get("url")
        link = f'<a href="{html.escape(url)}" target="_blank" rel="noreferrer">source</a>' if url else ""
        confidence = float(ev.get("confidence") or 0)
        severity = float(ev.get("severity") or 0)
        country = ev.get("country") or ""
        country_label = COUNTRY_KO_NAMES.get(country, country) if country else "국가 미확정 신호"
        channel = ev.get("channel") or "unknown"
        event_type = ev.get("event_type") or "signal"
        precision = ev.get("location_precision") or "missing"
        location = ev.get("location_name") or "unknown"
        coords = ""
        if ev.get("latitude") is not None and ev.get("longitude") is not None:
            coords = f'{float(ev["latitude"]):.4f}, {float(ev["longitude"]):.4f}'
        event_id = ev.get("event_id") or ""
        cards.append(
            f"""
            <article class="event" id="event-{html.escape(event_id)}" tabindex="0"
              data-event-id="{html.escape(event_id)}" data-country="{html.escape(country)}" data-channel="{html.escape(channel)}"
              data-type="{html.escape(event_type)}" data-precision="{html.escape(precision)}"
              data-message-time="{html.escape(ev.get("message_time") or "")}"
              data-confidence="{confidence:.4f}" data-search="{html.escape((country + ' ' + channel + ' ' + event_type + ' ' + location + ' ' + kw + ' ' + (ev.get('summary') or '')).lower())}">
              <div class="topline">
                <strong>{html.escape(country_label)}</strong>
                <span>{html.escape(event_type)}</span>
                <em class="{badge(confidence)}">{badge(confidence)}</em>
              </div>
              <p>{html.escape(ev.get("summary") or "")}</p>
              <div class="meta">
                <span>{html.escape(ev.get("message_time") or "")}</span>
                <span>channel: {html.escape(channel)}</span>
                <span>confidence: {confidence:.2f}</span>
                <span>severity: {severity:.2f}</span>
                <span>location: {html.escape(precision)} / {html.escape(location)}</span>
                <span>{html.escape(coords)}</span>
                <span>{html.escape(kw)}</span>
                {link}
              </div>
            </article>
            """
        )

    doc = (
        TEMPLATE.replace("__GENERATED_AT__", html.escape(payload.get("generated_at", "")))
        .replace("__FILTERS__", html.escape(json.dumps(payload.get("filters", {}), ensure_ascii=False)))
        .replace("__STATS__", stats)
        .replace("__COUNTRY_OPTIONS__", option_tags(countries, "국가: 전체"))
        .replace("__CHANNEL_OPTIONS__", option_tags(channels, "채널: 전체"))
        .replace("__TYPE_OPTIONS__", option_tags(event_types, "유형: 전체"))
        .replace("__EVENT_DATA__", event_data)
        .replace("__GDELT_CONTEXT_DATA__", gdelt_context_data)
        .replace("__MODEL_SCORES_DATA__", model_scores_data)
        .replace("__TOP_RISK_DATA__", top_risk_data)
        .replace("__COUNTRY_CENTROID_DATA__", country_centroid_data)
        .replace("__CARDS__", "\n".join(cards))
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(doc, encoding="utf-8")
    print(f"saved: {args.out}")


TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>실시간 무력충돌 예측 대시보드</title>
  <style>
    :root {
      --bg: #f5f5f7;
      --ink: #1d1d1f;
      --muted: #7a7a7a;
      --line: #e0e0e0;
      --panel: #ffffff;
      --tile-dark: #272729;
      --blue: #0066cc;
      --blue-focus: #0071e3;
      --red: #d94f45;
      --orange: #f08a4b;
      --green: #5f9f75;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "SF Pro Text", system-ui, -apple-system, BlinkMacSystemFont, "Malgun Gothic", sans-serif; background: var(--bg); color: var(--ink); }
    main { max-width: 1440px; margin: 0 auto; padding: 0 18px 56px; }
    header { margin: 0 -18px 20px; padding: 54px 20px 42px; text-align: center; background: var(--tile-dark); color: #fff; }
    h1 { margin: 0 0 10px; font-family: "SF Pro Display", system-ui, -apple-system, BlinkMacSystemFont, "Malgun Gothic", sans-serif; font-size: clamp(34px, 5vw, 56px); font-weight: 600; line-height: 1.07; letter-spacing: -0.28px; }
    .desc { color: #cccccc; line-height: 1.47; max-width: 900px; margin: 0 auto; font-size: 17px; letter-spacing: -0.224px; }
    .ranking-strip { display: inline-flex; align-items: center; gap: 14px; margin-top: 24px; padding: 10px 18px; min-height: 44px; border: 1px solid rgba(255,255,255,.16); border-radius: 999px; color: #fff; background: rgba(255,255,255,.06); }
    .ranking-strip span { color: #cccccc; font-size: 13px; }
    .ranking-strip strong { min-width: 250px; font-size: 15px; font-weight: 600; text-align: left; }
    .toolbar { display: grid; grid-template-columns: repeat(7, minmax(120px, 1fr)); gap: 10px; margin: 18px 0; }
    select, input { width: 100%; min-height: 44px; border: 1px solid var(--line); border-radius: 999px; padding: 10px 16px; background: white; color: var(--ink); font-size: 14px; letter-spacing: -0.224px; }
    select:focus, input:focus, button:focus { outline: 2px solid var(--blue-focus); outline-offset: 2px; }
    .zoom-control { position: absolute; top: 18px; right: 18px; z-index: 5; display: inline-flex; align-items: center; gap: 6px; padding: 6px; border: 1px solid var(--line); border-radius: 999px; background: rgba(255,255,255,.86); backdrop-filter: saturate(180%) blur(20px); }
    .zoom-control button { width: 36px; height: 36px; border: 0; border-radius: 999px; background: var(--blue); color: #fff; font-size: 20px; line-height: 1; cursor: pointer; }
    .zoom-control button:active { transform: scale(.95); }
    .zoom-control output { min-width: 42px; color: var(--ink); font-size: 13px; text-align: center; }
    .dashboard { display: grid; grid-template-columns: minmax(0, 1.62fr) minmax(340px, .72fr); gap: 16px; align-items: stretch; margin: 14px 0 18px; }
    .map-wrap { position: relative; border: 1px solid var(--line); border-radius: 18px; overflow: hidden; background: #ffffff; min-height: 100%; padding: 0; }
    #map { width: 100%; height: 640px; margin: 0; background: transparent; }
    .js-plotly-plot .plotly .modebar { display: none; }
    #fallbackMap { display: none; position: relative; width: 100%; height: 430px; overflow: hidden; background:
      linear-gradient(90deg, rgba(50,60,54,.10) 1px, transparent 1px),
      linear-gradient(0deg, rgba(50,60,54,.09) 1px, transparent 1px),
      linear-gradient(180deg, #dbe1de 0%, #eef1ee 52%, #d5dbd8 100%);
      background-size: 10% 100%, 100% 16.666%, 100% 100%; }
    #fallbackMap::before { content: "Fallback coordinate view"; position: absolute; left: 12px; top: 10px; color: #59635d; font-size: 13px; z-index: 1; }
    .fallback-marker { position: absolute; width: 22px; height: 22px; transform: translate(-50%, -50%) rotate(45deg); border: 1px solid rgba(255,255,255,.88); box-shadow: 0 0 0 2px rgba(30,36,32,.28), 0 3px 16px rgba(0,0,0,.20); cursor: pointer; }
    .fallback-marker.city { background: rgba(215, 55, 49, .78); color: rgba(215, 55, 49, .72); }
    .fallback-marker.country { width: 34px; height: 34px; background: rgba(224, 136, 31, .55); color: rgba(224, 136, 31, .58); }
    .legend { display: flex; flex-wrap: wrap; gap: 12px; padding: 0 18px 16px; color: var(--muted); font-size: 13px; background: transparent; }
    .dot { display: inline-block; width: 12px; height: 12px; margin-right: 5px; border: 1px solid rgba(255,255,255,.75); border-radius: 999px; }
    .dot.city { background: rgba(207, 62, 54, .78); }
    .dot.country { background: rgba(226, 141, 45, .62); }
    .dot.gdelt { background: rgba(79, 128, 94, .72); }
    .dot.missing { background: #64706a; }
    .count { color: var(--muted); margin: 8px 2px 12px; font-size: 13px; }
    .stats { display: grid; grid-template-columns: repeat(3, minmax(160px, 1fr)); gap: 12px; margin: 16px 0 18px; }
    .stat { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 18px 20px; }
    .stat span { display: block; color: var(--muted); font-size: 13px; }
    .stat strong { display: block; font-size: 23px; margin-top: 4px; overflow-wrap: anywhere; }
    .detail { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 18px; min-height: 430px; max-height: 700px; overflow: auto; }
    .detail-empty { color: var(--muted); line-height: 1.55; }
    .detail-head { display: flex; align-items: flex-start; gap: 10px; border-bottom: 1px solid #f0f0f0; padding-bottom: 14px; margin-bottom: 14px; }
    .detail-title { min-width: 0; }
    .detail-title strong { display: block; font-size: 20px; overflow-wrap: anywhere; }
    .detail-title span { color: var(--muted); font-size: 13px; }
    .panel-group { border: 1px solid var(--line); border-radius: 18px; padding: 16px; margin: 14px 0; background: #fff; }
    .panel-group.telegram { background: #fff; }
    .panel-group.gdelt { background: #fafafc; }
    .panel-group-title { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; margin-bottom: 10px; }
    .panel-group-title h2 { margin: 0; font-size: 15px; letter-spacing: 0; }
    .panel-group-title span { color: var(--muted); font-size: 12px; }
    .detail-section { margin: 14px 0; }
    .panel-group .detail-section { margin: 12px 0; }
    .panel-group .detail-section:first-of-type { margin-top: 0; }
    .panel-group .detail-section:last-of-type { margin-bottom: 0; }
    .detail-section h2 { margin: 0 0 7px; font-size: 13px; color: var(--muted); font-weight: 700; letter-spacing: 0; }
    .ko-summary { line-height: 1.55; background: #fafafc; border: 1px solid #f0f0f0; border-radius: 12px; padding: 12px; }
    .gdelt-metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin: 8px 0 10px; }
    .gdelt-metric { border: 1px solid var(--line); border-radius: 12px; padding: 10px; background: #fff; }
    .gdelt-metric span { display: block; color: var(--muted); font-size: 12px; }
    .gdelt-metric strong { display: block; font-size: 20px; margin-top: 2px; }
    .gdelt-keywords { color: var(--muted); font-size: 13px; margin: 8px 0; }
    .gdelt-title { border-top: 1px solid var(--line); padding: 8px 0; }
    .gdelt-title:first-of-type { border-top: 0; }
    .gdelt-title a { display: block; line-height: 1.35; }
    .gdelt-title span { color: var(--muted); display: block; font-size: 12px; margin-top: 2px; }
    .risk-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 8px 0 12px; }
    .risk-metric { border: 1px solid var(--line); border-radius: 12px; padding: 10px; background: #fff; }
    .risk-metric span { display: block; color: var(--muted); font-size: 12px; }
    .risk-metric strong { display: block; font-size: 22px; margin-top: 2px; }
    .tier-critical { background: #7f1d1d; }
    .tier-high { background: #d94f45; }
    .tier-watch { background: #f08a4b; }
    .tier-low { background: #64706a; }
    .brief-list { margin: 8px 0 0; padding-left: 18px; color: #243029; line-height: 1.45; }
    .brief-list a { color: var(--ink); text-decoration: none; }
    .brief-list a:hover { color: var(--blue); text-decoration: underline; }
    details.raw-toggle { border: 1px solid var(--line); border-radius: 12px; padding: 10px 12px; background: #fafafc; }
    details.raw-toggle summary { cursor: pointer; color: var(--blue); font-size: 14px; font-weight: 600; }
    .raw-message { white-space: pre-wrap; line-height: 1.5; font-size: 13px; color: #1d1d1f; margin-top: 10px; }
    .detail .meta { margin-top: 8px; }
    .event { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 16px; margin: 12px 0; cursor: pointer; }
    .event:target, .event.selected { outline: 2px solid var(--blue); }
    .event:focus { outline: 2px solid var(--blue); outline-offset: 2px; }
    .event.hidden { display: none; }
    #events { display: none; }
    .topline { display: flex; align-items: center; gap: 10px; }
    .topline strong { font-size: 18px; }
    .topline span { color: var(--muted); }
    em { margin-left: auto; padding: 4px 8px; border-radius: 999px; color: white; font-style: normal; font-size: 12px; }
    em.likely { background: var(--red); }
    em.unverified { background: var(--orange); }
    em.raw { background: #64706a; }
    p { line-height: 1.55; }
    .meta { display: flex; flex-wrap: wrap; gap: 8px 14px; color: var(--muted); font-size: 13px; }
    a { color: var(--blue); font-weight: 700; }
    .source-link { display: inline-flex; align-items: center; margin-top: 8px; }
    @media (max-width: 820px) {
      main { padding: 0 12px 48px; }
      header { margin: 0 -12px 16px; padding: 42px 16px 34px; }
      .stats { grid-template-columns: 1fr; }
      .toolbar { grid-template-columns: 1fr; }
      .dashboard { grid-template-columns: 1fr; }
      .ranking-strip { width: 100%; justify-content: center; }
      .ranking-strip strong { min-width: 0; }
      .zoom-control { top: 12px; right: 12px; }
      #map { height: 430px; }
      #fallbackMap { height: 320px; }
      .detail { min-height: 0; max-height: none; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>실시간 무력충돌 예측 대시보드</h1>
      <div class="desc">
        모델 점수를 중심으로 국가별 무력충돌 위험도를 시각화하고, Telegram 및 GDELT 뉴스 신호를 보조 근거로 함께 확인합니다.
      </div>
      <div class="ranking-strip" aria-live="polite">
        <span>위험도 Top 10</span>
        <strong id="topRiskTicker">-</strong>
      </div>
    </header>
    <section class="toolbar" aria-label="feed filters">
      <select id="mapViewFilter">
        <option value="globe">지도 형태: 지구본</option>
        <option value="flat">지도 형태: 평면</option>
      </select>
      <select id="mapColorFilter">
        <option value="model">색상 기준: 모델 위험도</option>
        <option value="observed">색상 기준: 관측 신호</option>
      </select>
      <select id="timeFilter">
        <option value="">기간: 최신 모델일</option>
        <option value="720">기간: 최근 1개월</option>
        <option value="168">기간: 최근 1주</option>
        <option value="24">기간: 최근 24시간</option>
      </select>
      <select id="countryFilter">__COUNTRY_OPTIONS__</select>
      <select id="channelFilter">__CHANNEL_OPTIONS__</select>
      <select id="typeFilter">__TYPE_OPTIONS__</select>
      <select id="precisionFilter">
        <option value="">위치: 전체</option>
        <option value="city">위치: 도시 좌표</option>
        <option value="country">위치: 국가 추정</option>
        <option value="missing">위치: 미확정</option>
      </select>
      <input id="searchFilter" type="search" placeholder="검색어 또는 키워드" />
    </section>
    <section class="dashboard" aria-label="event dashboard">
      <div class="map-wrap" aria-label="event map">
        <div class="zoom-control" aria-label="지도 확대">
          <button id="zoomOutButton" type="button" aria-label="지도 축소">-</button>
          <output id="zoomValue">100%</output>
          <button id="zoomInButton" type="button" aria-label="지도 확대">+</button>
        </div>
        <div id="map"></div>
        <div id="fallbackMap"></div>
        <div class="legend">
          <span><i class="dot city"></i>Telegram 도시 신호</span>
          <span><i class="dot country"></i>Telegram 국가 신호</span>
          <span><i class="dot gdelt"></i>GDELT 뉴스 신호</span>
          <span><i class="dot missing"></i>지도 표시 불가</span>
        </div>
      </div>
      <aside class="detail" id="eventDetail" aria-live="polite">
        <div class="detail-empty">지도에서 국가 또는 이벤트를 선택하세요.</div>
      </aside>
    </section>
    <section class="stats">__STATS__</section>
    <div class="count" id="visibleCount"></div>
    <section id="events">__CARDS__</section>
  </main>
  <script id="eventData" type="application/json">__EVENT_DATA__</script>
  <script id="gdeltContextData" type="application/json">__GDELT_CONTEXT_DATA__</script>
  <script id="modelScoreData" type="application/json">__MODEL_SCORES_DATA__</script>
  <script id="topRiskData" type="application/json">__TOP_RISK_DATA__</script>
  <script id="countryCentroidData" type="application/json">__COUNTRY_CENTROID_DATA__</script>
  <script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
  <script>
    const events = JSON.parse(document.getElementById("eventData").textContent);
    const gdeltContext = JSON.parse(document.getElementById("gdeltContextData").textContent);
    const modelScores = JSON.parse(document.getElementById("modelScoreData").textContent);
    const topRiskData = JSON.parse(document.getElementById("topRiskData").textContent);
    const countryCentroids = JSON.parse(document.getElementById("countryCentroidData").textContent);
    const controls = {
      mapView: document.getElementById("mapViewFilter"),
      mapColor: document.getElementById("mapColorFilter"),
      zoomOut: document.getElementById("zoomOutButton"),
      zoomIn: document.getElementById("zoomInButton"),
      zoomValue: document.getElementById("zoomValue"),
      time: document.getElementById("timeFilter"),
      country: document.getElementById("countryFilter"),
      channel: document.getElementById("channelFilter"),
      type: document.getElementById("typeFilter"),
      precision: document.getElementById("precisionFilter"),
      search: document.getElementById("searchFilter"),
    };
    const cards = Array.from(document.querySelectorAll(".event"));
    const count = document.getElementById("visibleCount");
    const fallbackMap = document.getElementById("fallbackMap");
    const plotlyMap = document.getElementById("map");
    const detail = document.getElementById("eventDetail");
    const eventById = new Map(events.map((event) => [event.event_id, event]));
    const mappableEvents = events.filter((event) => event.latitude !== null && event.longitude !== null);
    const mapAvailable = typeof Plotly !== "undefined";
    let map = null;
    let countrySignalCounts = new Map();
    let currentVisibleIds = new Set();
    let selectedEventId = null;
    let selectedGdeltCountry = null;
    let mapZoom = 1;
    let tickerIndex = 0;
    const countryKoNames = {
      AFG: "아프가니스탄",
      ARM: "아르메니아",
      AZE: "아제르바이잔",
      BFA: "부르키나파소",
      CAF: "중앙아프리카공화국",
      COD: "콩고민주공화국",
      ETH: "에티오피아",
      HTI: "아이티",
      IND: "인도",
      IRN: "이란",
      IRQ: "이라크",
      ISR: "이스라엘",
      KEN: "케냐",
      KWT: "쿠웨이트",
      LBN: "레바논",
      LBY: "리비아",
      LTU: "리투아니아",
      MLI: "말리",
      MMR: "미얀마",
      NER: "니제르",
      NGA: "나이지리아",
      PAK: "파키스탄",
      PSE: "팔레스타인",
      ROU: "루마니아",
      RUS: "러시아",
      SAU: "사우디아라비아",
      SDN: "수단",
      SOM: "소말리아",
      SSD: "남수단",
      SYR: "시리아",
      TCD: "차드",
      TUR: "튀르키예",
      UKR: "우크라이나",
      USA: "미국",
      YEM: "예멘",
    };
    Object.assign(countryKoNames, {
      AFG: "아프가니스탄",
      ARM: "아르메니아",
      AZE: "아제르바이잔",
      BFA: "부르키나파소",
      BGD: "방글라데시",
      CAF: "중앙아프리카공화국",
      CIV: "코트디부아르",
      CMR: "카메룬",
      COD: "콩고민주공화국",
      COL: "콜롬비아",
      DZA: "알제리",
      ECU: "에콰도르",
      EGY: "이집트",
      ERI: "에리트레아",
      ETH: "에티오피아",
      GIN: "기니",
      GNB: "기니비사우",
      GTM: "과테말라",
      HND: "온두라스",
      HTI: "아이티",
      IDN: "인도네시아",
      IND: "인도",
      IRN: "이란",
      IRQ: "이라크",
      ISR: "이스라엘",
      KEN: "케냐",
      KGZ: "키르기스스탄",
      KWT: "쿠웨이트",
      LBN: "레바논",
      LBY: "리비아",
      LTU: "리투아니아",
      MDG: "마다가스카르",
      MEX: "멕시코",
      MLI: "말리",
      MMR: "미얀마",
      MOZ: "모잠비크",
      NER: "니제르",
      NGA: "나이지리아",
      PAK: "파키스탄",
      PHL: "필리핀",
      PSE: "팔레스타인",
      ROU: "루마니아",
      RUS: "러시아",
      SAU: "사우디아라비아",
      SDN: "수단",
      SEN: "세네갈",
      SLE: "시에라리온",
      SOM: "소말리아",
      SSD: "남수단",
      SYR: "시리아",
      TCD: "차드",
      TGO: "토고",
      THA: "태국",
      TJK: "타지키스탄",
      TUN: "튀니지",
      TUR: "튀르키예",
      UGA: "우간다",
      UKR: "우크라이나",
      USA: "미국",
      VEN: "베네수엘라",
      YEM: "예멘",
      ZWE: "짐바브웨",
    });
    function countryName(country) {
      return countryKoNames[country] || country || "국가 미확정 신호";
    }
    const locationKoNames = {
      beirut: "베이루트",
      bucharest: "부쿠레슈티",
      "gaza": "가자",
      "gaza city": "가자시티",
      kharkiv: "하르키우",
      kherson: "헤르손",
      kyiv: "키이우",
      kramatorsk: "크라마토르스크",
      kupiansk: "쿠피얀스크",
      lebanon: "레바논",
      lithuania: "리투아니아",
      "st. petersburg": "상트페테르부르크",
      sumy: "수미",
      syzran: "시즈란",
      "tambov oblast": "탐보프주",
      ukraine: "우크라이나",
    };

    function markerColor(precision) {
      if (precision === "city") return "#d73531";
      if (precision === "country") return "#dd8725";
      return "#64706a";
    }

    function updateZoom(delta) {
      mapZoom = Math.max(0.75, Math.min(2.4, Number((mapZoom + delta).toFixed(2))));
      if (controls.zoomValue) controls.zoomValue.textContent = `${Math.round(mapZoom * 100)}%`;
      if (currentVisibleIds.size || map) {
        applyFilters();
      }
    }

    function resizeMapForMode() {
      if (!plotlyMap) return;
      const flatMode = controls.mapView && controls.mapView.value === "flat";
      const width = Math.max(320, plotlyMap.parentElement ? plotlyMap.parentElement.clientWidth : plotlyMap.clientWidth);
      const height = flatMode
        ? Math.max(300, Math.min(430, Math.round(width * 0.32)))
        : Math.max(460, Math.min(620, Math.round(width * 0.56)));
      plotlyMap.style.height = `${height}px`;
      if (mapAvailable && map && map._fullLayout && window.Plotly && Plotly.Plots) {
        Plotly.Plots.resize(map);
      }
    }

    function renderTopRiskTicker() {
      const target = document.getElementById("topRiskTicker");
      if (!target || !topRiskData.length) return;
      const item = topRiskData[tickerIndex % topRiskData.length];
      const rank = (tickerIndex % topRiskData.length) + 1;
      const name = countryName(item.country);
      target.textContent = `${rank}위 ${name} · 위험도 ${Number(item.risk_score || 0).toFixed(1)}`;
      tickerIndex += 1;
    }

    function modelItem(country) {
      if (!country || country === "UNK") return null;
      return (modelScores.countries || {})[country];
    }

    function modelHistory(country) {
      if (!country || country === "UNK") return [];
      return (modelScores.history || {})[country] || [];
    }

    function latestModelDateMs() {
      const dates = Object.values(modelScores.countries || {})
        .map((item) => Date.parse(`${item.date || ""}T00:00:00Z`))
        .filter((value) => !Number.isNaN(value));
      return dates.length ? Math.max(...dates) : Date.now();
    }

    function periodModelSummary(country) {
      const hours = Number(controls.time.value || 0);
      if (!hours) return null;
      const latestMs = latestModelDateMs();
      const cutoffMs = latestMs - hours * 60 * 60 * 1000;
      const rows = modelHistory(country).filter((row) => {
        const dateMs = Date.parse(`${row.date || ""}T00:00:00Z`);
        return !Number.isNaN(dateMs) && dateMs >= cutoffMs && dateMs <= latestMs;
      });
      if (!rows.length) return null;
      const scores = rows.map((row) => Number(row.risk_score || 0));
      const onsets = rows.map((row) => Number(row.onset_prob || 0));
      const avg = scores.reduce((sum, value) => sum + value, 0) / scores.length;
      const max = Math.max(...scores);
      const avgOnset = onsets.reduce((sum, value) => sum + value, 0) / onsets.length;
      const peak = rows.reduce((best, row) => Number(row.risk_score || 0) > Number(best.risk_score || 0) ? row : best, rows[0]);
      return {
        count: rows.length,
        avg,
        max,
        avgOnset,
        peakDate: peak.date || "",
      };
    }

    function riskColor(score) {
      const value = Number(score || 0);
      if (value >= 98) return "#7f1d1d";
      if (value >= 95) return "#b4312c";
      if (value >= 85) return "#d94f45";
      if (value >= 65) return "#e9896a";
      if (value >= 40) return "#f0c7b2";
      if (value > 0) return "#f7e9df";
      return "#f3f4f0";
    }

    function tierLabel(tier) {
      const labels = {
        critical: "매우 높음",
        high: "높음",
        watch: "관찰",
        low: "낮음",
      };
      return labels[tier] || "낮음";
    }

    function signalFillColor(count) {
      const value = Number(count || 0);
      if (value >= 30) return "#b14c42";
      if (value >= 15) return "#cf714e";
      if (value >= 7) return "#e19a55";
      if (value >= 3) return "#e9c46a";
      if (value > 0) return "#9dbb89";
      return "#eef1ea";
    }

    function buildCountrySignalCounts(visibleIds, visibleGdeltCountries) {
      const counts = new Map();
      for (const eventId of visibleIds) {
        const event = eventById.get(eventId);
        if (!event || !event.country) continue;
        counts.set(event.country, (counts.get(event.country) || 0) + 1);
      }
      for (const country of visibleGdeltCountries || []) {
        const context = (gdeltContext.countries || {})[country] || {};
        const newsSignals = Math.min(10, Math.ceil(Number(context.gdelt_7d || context.gdelt_24h || 0) / 10));
        counts.set(country, (counts.get(country) || 0) + Math.max(1, newsSignals));
      }
      return counts;
    }

    function countryEvents(country, limit = 4) {
      return events
        .filter((event) => event.country === country)
        .sort((a, b) => Date.parse(b.message_time || 0) - Date.parse(a.message_time || 0))
        .slice(0, limit);
    }

    function hasGdeltContext(event) {
      const context = (gdeltContext.countries || {})[event.country || "UNK"];
      return Boolean(context && Number(context.gdelt_7d || 0) > 0);
    }

    function hasGdeltCountryContext(country) {
      const context = (gdeltContext.countries || {})[country || "UNK"];
      return Boolean(context && (Number(context.gdelt_7d || 0) > 0 || Number(context.gdelt_24h || 0) > 0));
    }

    function escapeHtml(value) {
      return String(value || "").replace(/[<>&"]/g, (char) => ({
        "<": "&lt;",
        ">": "&gt;",
        "&": "&amp;",
        '"': "&quot;",
      }[char]));
    }

    function localizeCountryCodes(value) {
      return String(value || "").replace(/\\(([A-Z]{3})\\)/g, (match, code) => {
        const name = countryName(code);
        return name === code ? match : `(${name})`;
      });
    }

    function formatNumber(value) {
      return Number(value || 0).toFixed(2);
    }

    function eventMeta(event) {
      const location = `${event.location_precision || "missing"} / ${event.location_name || "unknown"}`;
      const coords = event.latitude !== null && event.longitude !== null
        ? `${Number(event.latitude).toFixed(4)}, ${Number(event.longitude).toFixed(4)}`
        : "no coordinates";
      const keywords = (event.matched_keywords || []).join(", ") || "none";
      return `
        <div class="meta">
          <span>${escapeHtml(event.message_time || "")}</span>
          <span>channel: ${escapeHtml(event.channel || "unknown")}</span>
          <span>confidence: ${formatNumber(event.confidence)}</span>
          <span>severity: ${formatNumber(event.severity)}</span>
          <span>location: ${escapeHtml(location)}</span>
          <span>${escapeHtml(coords)}</span>
          <span>${escapeHtml(keywords)}</span>
        </div>
      `;
    }

    function gdeltSection(event) {
      const country = event.country || "";
      const countryLabel = country ? countryName(country) : "국가 미확정 신호";
      const context = (gdeltContext.countries || {})[country];
      if (!context) {
        return `
          <div class="panel-group gdelt">
            <div class="panel-group-title">
              <h2>GDELT News Context</h2>
              <span>news coverage layer</span>
            </div>
            <div class="detail-empty">No GDELT title context is available for ${escapeHtml(countryLabel)} yet.</div>
          </div>
        `;
      }
      const keywords = (context.top_keywords || []).join(", ") || "none";
      const titles = (context.top_titles || []).slice(0, 3).map((item) => {
        const title = escapeHtml(item.title || "Untitled");
        const meta = escapeHtml(`${item.date || ""} ${item.domain || ""}`.trim());
        const url = item.url
          ? `<a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">${title}</a>`
          : `<strong>${title}</strong>`;
        return `<div class="gdelt-title">${url}<span>${meta}</span></div>`;
      }).join("");
      return `
        <div class="panel-group gdelt">
          <div class="panel-group-title">
            <h2>GDELT News Context</h2>
            <span>news coverage layer</span>
          </div>
          <div class="gdelt-metrics">
            <div class="gdelt-metric"><span>titles latest day</span><strong>${escapeHtml(context.gdelt_24h || 0)}</strong></div>
            <div class="gdelt-metric"><span>titles latest 7d</span><strong>${escapeHtml(context.gdelt_7d || 0)}</strong></div>
          </div>
              ${context.ko_brief ? `<div class="gdelt-keywords">${escapeHtml(localizeCountryCodes(context.ko_brief))}</div>` : ""}
          <div class="ko-summary">${escapeHtml(localizeCountryCodes(context.ko_summary || "GDELT 기사 제목 기반 한국어 요약은 아직 생성되지 않았습니다."))}</div>
          ${context.source_limits ? `<div class="detail-empty">${escapeHtml(context.source_limits)}</div>` : ""}
          <div class="gdelt-keywords">Top GDELT terms: ${escapeHtml(keywords)}</div>
          ${titles || '<div class="detail-empty">No representative titles are available yet.</div>'}
        </div>
      `;
    }

    function modelRiskSection(country) {
      const item = modelItem(country);
      if (!item) {
        return `
          <div class="panel-group gdelt">
            <div class="panel-group-title">
              <h2>모델 위험도 점수</h2>
              <span>onset model layer</span>
            </div>
            <div class="detail-empty">No model score is available for ${escapeHtml(country)}.</div>
          </div>
        `;
      }
      const risk = Number(item.risk_score || 0);
      const onset = Number(item.onset_prob || 0);
      const base = Number(item.base_pred || 0);
      const calm = Number(item.calm_flag || 0);
      const mode = calm === 1 ? "신규 충돌 발생 감시 대상" : "현재 충돌/긴장 모니터링 대상";
      const period = periodModelSummary(country);
      const periodLabel = controls.time.value === "720"
        ? "최근 1개월"
        : controls.time.value === "168"
          ? "최근 1주"
          : controls.time.value === "24"
            ? "최근 24시간"
            : "";
      const periodBlock = period ? `
        <div class="detail-section">
          <h2>선택 기간 점수 흐름</h2>
          <div class="risk-metrics">
            <div class="risk-metric"><span>${escapeHtml(periodLabel)} 평균 위험도</span><strong>${period.avg.toFixed(1)}</strong></div>
            <div class="risk-metric"><span>${escapeHtml(periodLabel)} 최고 위험도</span><strong>${period.max.toFixed(1)}</strong></div>
            <div class="risk-metric"><span>평균 onset score</span><strong>${period.avgOnset.toFixed(3)}</strong></div>
          </div>
          <div class="detail-empty">기준: ${period.count}개 모델 일자, 최고점 날짜 ${escapeHtml(period.peakDate)}.</div>
        </div>
      ` : "";
      return `
        <div class="panel-group gdelt">
          <div class="panel-group-title">
            <h2>모델 위험도 점수</h2>
            <span>${escapeHtml(item.date || "")} · ${escapeHtml(item.run_ts || "")}</span>
          </div>
          <div class="risk-metrics">
            <div class="risk-metric"><span>위험도 점수</span><strong>${risk.toFixed(1)}</strong></div>
            <div class="risk-metric"><span>onset score</span><strong>${onset.toFixed(3)}</strong></div>
            <div class="risk-metric"><span>위험 단계</span><strong>${escapeHtml(tierLabel(item.tier))}</strong></div>
          </div>
          <div class="meta">
            <span>base_pred: ${base.toFixed(3)}</span>
            <span>calm_flag: ${calm}</span>
            <span>${escapeHtml(mode)}</span>
          </div>
          <div class="detail-empty">위험도 점수는 최신 모델 기준일의 onset score를 국가 간 순위로 변환한 값입니다. onset score는 신규 무력충돌 발생 가능성을 비교하기 위한 모델 신호이며, 보정된 절대 확률은 아닙니다.</div>
          ${periodBlock}
        </div>
      `;
    }

    function countryBriefingSection(country) {
      const recentEvents = countryEvents(country, 4);
      const context = (gdeltContext.countries || {})[country];
      const eventItems = recentEvents.map((event) => {
        const label = `${event.message_time || ""} · ${event.channel || "unknown"} · ${event.event_type || "signal"}`;
        const summary = event.ko_raw_brief || event.ko_summary || event.summary || "";
        const body = `<strong>${escapeHtml(label)}</strong><br>${escapeHtml(localizeCountryCodes(summary))}`;
        return event.url
          ? `<li><a href="${escapeHtml(event.url)}" target="_blank" rel="noreferrer">${body}</a></li>`
          : `<li>${body}</li>`;
      }).join("");
      const gdeltTitles = ((context || {}).top_titles || []).slice(0, 3).map((item) => {
        const label = `${item.date || ""} ${item.domain || ""}`.trim();
        return `<li><strong>${escapeHtml(label)}</strong><br>${escapeHtml(item.title || "Untitled")}</li>`;
      }).join("");
      return `
        <div class="panel-group telegram">
          <div class="panel-group-title">
            <h2>Recent Briefing</h2>
            <span>Telegram + GDELT context</span>
          </div>
          ${eventItems ? `<div class="detail-section"><h2>Telegram signals</h2><ul class="brief-list">${eventItems}</ul></div>` : '<div class="detail-empty">No recent Telegram signal is linked to this country in the current export.</div>'}
          ${context ? `
            <div class="detail-section">
              <h2>GDELT news context</h2>
              ${context.ko_brief ? `<div class="gdelt-keywords">${escapeHtml(localizeCountryCodes(context.ko_brief))}</div>` : ""}
              <div class="ko-summary">${escapeHtml(localizeCountryCodes(context.ko_summary || "GDELT title summary is not available yet."))}</div>
              ${gdeltTitles ? `<ul class="brief-list">${gdeltTitles}</ul>` : ""}
            </div>
          ` : '<div class="detail-empty">No GDELT title context is available for this country yet.</div>'}
        </div>
      `;
    }

    function selectCountry(country) {
      if (!country || country === "UNK") return;
      selectedEventId = null;
      selectedGdeltCountry = country;
      cards.forEach((card) => card.classList.remove("selected"));
      const item = modelItem(country);
      detail.innerHTML = `
        <div class="detail-head">
          <div class="detail-title">
            <strong>${escapeHtml(countryName(country))} - 모델 위험도</strong>
            <span>모델 점수 중심 국가 보기</span>
          </div>
          <em class="tier-${escapeHtml((item || {}).tier || "low")}">${escapeHtml(tierLabel((item || {}).tier))}</em>
        </div>
        ${modelRiskSection(country)}
        ${countryBriefingSection(country)}
      `;
    }

    function selectGdeltCountry(country) {
      const context = (gdeltContext.countries || {})[country];
      if (!context) return;
      selectedEventId = null;
      selectedGdeltCountry = country;
      cards.forEach((card) => card.classList.remove("selected"));
      const keywords = (context.top_keywords || []).join(", ") || "none";
      const titles = (context.top_titles || []).slice(0, 5).map((item) => {
        const title = escapeHtml(item.title || "Untitled");
        const meta = escapeHtml(`${item.date || ""} ${item.domain || ""}`.trim());
        const url = item.url
          ? `<a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">${title}</a>`
          : `<strong>${title}</strong>`;
        return `<div class="gdelt-title">${url}<span>${meta}</span></div>`;
      }).join("");
      detail.innerHTML = `
        <div class="detail-head">
          <div class="detail-title">
            <strong>${escapeHtml(countryName(country))} - GDELT 뉴스 맥락</strong>
            <span>Telegram 이벤트 없이 표시된 뉴스 레이어</span>
          </div>
          <em class="unverified">news</em>
        </div>
        <div class="panel-group gdelt">
          <div class="panel-group-title">
            <h2>GDELT News Context</h2>
            <span>country-level news layer</span>
          </div>
          <div class="gdelt-metrics">
            <div class="gdelt-metric"><span>titles latest day</span><strong>${escapeHtml(context.gdelt_24h || 0)}</strong></div>
            <div class="gdelt-metric"><span>titles latest 7d</span><strong>${escapeHtml(context.gdelt_7d || 0)}</strong></div>
          </div>
          ${context.ko_brief ? `<div class="gdelt-keywords">${escapeHtml(context.ko_brief)}</div>` : ""}
            <div class="ko-summary">${escapeHtml(localizeCountryCodes(context.ko_summary || "GDELT 기사 제목 기반 한국어 요약은 아직 생성되지 않았습니다."))}</div>
          ${context.source_limits ? `<div class="detail-empty">${escapeHtml(context.source_limits)}</div>` : ""}
          <div class="gdelt-keywords">Top GDELT terms: ${escapeHtml(keywords)}</div>
          ${titles || '<div class="detail-empty">No representative titles are available yet.</div>'}
        </div>
        <div class="panel-group telegram">
          <div class="panel-group-title">
            <h2>Telegram Signal</h2>
            <span>source message layer</span>
          </div>
          <div class="detail-empty">현재 export 범위에서 이 국가에 연결된 Telegram event는 선택되지 않았습니다.</div>
        </div>
      `;
    }

    function selectEvent(eventId) {
      const event = eventById.get(eventId);
      if (!event) return;
      selectedEventId = eventId;
      selectedGdeltCountry = null;
      cards.forEach((card) => card.classList.toggle("selected", card.dataset.eventId === eventId));
      const source = event.url
        ? `<a class="source-link" href="${escapeHtml(event.url)}" target="_blank" rel="noreferrer">Open Telegram source</a>`
        : "";
      const countryLabel = event.country ? countryName(event.country) : "국가 미확정 신호";
      const modelSection = event.country ? modelRiskSection(event.country) : "";
      detail.innerHTML = `
        <div class="detail-head">
          <div class="detail-title">
            <strong>${escapeHtml(countryLabel)} - ${escapeHtml(event.event_type || "signal")}</strong>
            <span>${escapeHtml(event.location_name || "unknown")} - ${escapeHtml(event.location_precision || "missing")}</span>
          </div>
          <em class="${badgeClass(event.confidence)}">${badgeClass(event.confidence)}</em>
        </div>
        ${eventMeta(event)}
        <div class="panel-group telegram">
          <div class="panel-group-title">
            <h2>Telegram Signal</h2>
            <span>source message layer</span>
          </div>
          <div class="detail-section">
            <h2>분류 요약</h2>
            <div class="ko-summary">${escapeHtml(localizeCountryCodes(event.ko_summary || event.summary || ""))}</div>
          </div>
          <div class="detail-section">
            <h2>원문 기반 한국어 요약</h2>
            ${event.ko_raw_brief ? `<div class="gdelt-keywords">${escapeHtml(localizeCountryCodes(event.ko_raw_brief))}</div>` : ""}
            <div class="ko-summary">${escapeHtml(localizeCountryCodes(event.ko_raw_summary || ""))}</div>
            ${event.ko_raw_source_limits ? `<div class="detail-empty">${escapeHtml(event.ko_raw_source_limits)}</div>` : ""}
          </div>
          <div class="detail-section">
            <details class="raw-toggle">
              <summary>원문 Telegram 메시지 보기</summary>
              <div class="raw-message">${escapeHtml(event.raw_text || event.summary || "")}</div>
              ${source}
            </details>
          </div>
        </div>
        ${modelSection}
        ${gdeltSection(event)}
      `;
    }

    function badgeClass(confidence) {
      const value = Number(confidence || 0);
      if (value >= 0.75) return "likely";
      if (value >= 0.55) return "unverified";
      return "raw";
    }

    function initMap() {
      if (!mapAvailable) {
        plotlyMap.style.display = "none";
        fallbackMap.style.display = "block";
        return;
      }
      map = plotlyMap;
      resizeMapForMode();
      Plotly.newPlot(map, [], globeLayout(), {
        responsive: true,
        displayModeBar: false,
        scrollZoom: false,
      });
      map.on("plotly_click", (ev) => {
        const point = ev.points && ev.points[0];
        if (!point) return;
        if (point.data && point.data.meta === "telegram") {
          selectEvent(point.customdata);
          return;
        }
        if (point.data && point.data.meta === "gdelt") {
          selectCountry(point.customdata);
          return;
        }
        if (point.data && point.data.meta === "country") {
          const country = point.customdata || point.location;
          selectCountry(country);
        }
      });
    }

    function globeLayout() {
      const flatMode = controls.mapView && controls.mapView.value === "flat";
      const zoom = Number(mapZoom || 1);
      const projectionScale = flatMode
        ? Math.max(1.08, zoom * 1.18)
        : 0.82 + ((zoom - 1) * 1.65);
      return {
        autosize: true,
        margin: { l: 0, r: 0, t: 0, b: 0 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        hovermode: "closest",
        showlegend: false,
        geo: {
          domain: flatMode ? { x: [0, 1], y: [0, 1] } : { x: [0.01, 0.99], y: [0.02, 0.98] },
          projection: flatMode
            ? { type: "natural earth", scale: projectionScale }
            : { type: "orthographic", scale: projectionScale, rotation: { lon: 35, lat: 18, roll: 0 } },
          bgcolor: "rgba(0,0,0,0)",
          showframe: false,
          showcoastlines: true,
          coastlinecolor: "#aeb8af",
          coastlinewidth: 0.8,
          showcountries: true,
          countrycolor: "#ffffff",
          countrywidth: 0.8,
          showland: true,
          landcolor: "#eef1ea",
          showocean: true,
          oceancolor: "#dce8ee",
          showlakes: true,
          lakecolor: "#dce8ee",
          showrivers: false,
          resolution: 50,
          lataxis: flatMode ? { range: [-55, 78] } : undefined,
          lonaxis: flatMode ? { range: [-180, 180] } : undefined,
        },
      };
    }

    function fallbackPosition(lat, lon) {
      const x = ((Number(lon) + 180) / 360) * 100;
      const y = ((90 - Number(lat)) / 180) * 100;
      return { x: Math.max(0, Math.min(100, x)), y: Math.max(0, Math.min(100, y)) };
    }

    function renderFallbackMarkers(visibleIds) {
      fallbackMap.innerHTML = "";
      for (const event of mappableEvents) {
        if (!visibleIds.has(event.event_id)) continue;
        const pos = fallbackPosition(event.latitude, event.longitude);
        const marker = document.createElement("a");
        marker.className = `fallback-marker ${event.location_precision === "city" ? "city" : "country"}`;
        marker.href = `#event-${event.event_id}`;
        marker.title = `${event.country ? countryName(event.country) : "국가 미확정 신호"} ${event.event_type || "signal"}`;
        marker.style.left = `${pos.x}%`;
        marker.style.top = `${pos.y}%`;
        marker.addEventListener("click", (clickEvent) => {
          clickEvent.preventDefault();
          selectEvent(event.event_id);
        });
        fallbackMap.appendChild(marker);
      }
    }

    function renderMarkers(visibleIds, visibleGdeltCountries) {
      if (!mapAvailable || !map) return;
      resizeMapForMode();
      currentVisibleIds = new Set(visibleIds);
      countrySignalCounts = buildCountrySignalCounts(visibleIds, visibleGdeltCountries);
      const eventCountries = new Set();
      const modelColorMode = !controls.mapColor || controls.mapColor.value === "model";
      const countryLocations = modelColorMode
        ? Object.keys(modelScores.countries || {}).filter((country) => country && country !== "UNK")
        : Array.from(countrySignalCounts.keys()).filter((country) => country && country !== "UNK");
      const countryValues = countryLocations.map((country) =>
        modelColorMode ? Number((modelItem(country) || {}).risk_score || 0) : Number(countrySignalCounts.get(country) || 0)
      );
      const countryTrace = {
        type: "choropleth",
        meta: "country",
        locationmode: "ISO-3",
        locations: countryLocations,
        z: countryValues,
        customdata: countryLocations,
        text: countryLocations.map((country) => {
          const item = modelItem(country);
          return item
            ? `${countryName(country)} · 위험도 ${Number(item.risk_score || 0).toFixed(1)} · onset ${Number(item.onset_prob || 0).toFixed(3)}`
            : `${countryName(country)}`;
        }),
        hovertemplate: modelColorMode
          ? "<b>%{text}</b><extra></extra>"
          : "<b>%{text}</b><br>observed signals: %{z}<extra></extra>",
        zmin: 0,
        zmax: modelColorMode ? 100 : Math.max(10, ...countryValues),
        showlegend: false,
        colorscale: modelColorMode
          ? [
              [0.00, "#f3f4f0"],
              [0.35, "#f7e9df"],
              [0.65, "#e9896a"],
              [0.85, "#d94f45"],
              [0.95, "#b4312c"],
              [1.00, "#7f1d1d"],
            ]
          : [
              [0.00, "#eef1ea"],
              [0.18, "#9dbb89"],
              [0.38, "#e9c46a"],
              [0.62, "#e19a55"],
              [0.82, "#cf714e"],
              [1.00, "#9c3f38"],
            ],
        showscale: false,
        marker: { line: { color: "rgba(255,255,255,0.82)", width: 0.8 } },
      };

      const visibleEvents = mappableEvents.filter((event) => visibleIds.has(event.event_id));
      for (const event of mappableEvents) {
        if (!visibleIds.has(event.event_id)) continue;
        if (event.country) eventCountries.add(event.country);
      }
      const telegramTrace = {
        type: "scattergeo",
        meta: "telegram",
        mode: "markers",
        lat: visibleEvents.map((event) => Number(event.latitude)),
        lon: visibleEvents.map((event) => Number(event.longitude)),
        customdata: visibleEvents.map((event) => event.event_id),
        text: visibleEvents.map((event) => `${event.location_name || (event.country ? countryName(event.country) : "국가 미확정 신호")} - ${event.event_type || "signal"}`),
        hovertemplate: "<b>%{text}</b><br>Telegram signal<extra></extra>",
        showlegend: false,
        marker: {
          size: visibleEvents.map((event) => event.location_precision === "city" ? 9 : 12),
          color: visibleEvents.map((event) => markerColor(event.location_precision)),
          opacity: visibleEvents.map((event) => event.location_precision === "city" ? 0.88 : 0.70),
          line: { color: "#ffffff", width: 1.4 },
        },
      };

      const gdeltOnlyCountries = Array.from(visibleGdeltCountries || []).filter((country) =>
        !eventCountries.has(country) && countryCentroids[country]
      );
      const gdeltTrace = {
        type: "scattergeo",
        meta: "gdelt",
        mode: "markers",
        lat: gdeltOnlyCountries.map((country) => Number(countryCentroids[country][0])),
        lon: gdeltOnlyCountries.map((country) => Number(countryCentroids[country][1])),
        customdata: gdeltOnlyCountries,
        text: gdeltOnlyCountries.map((country) => countryName(country)),
        hovertemplate: "<b>%{text}</b><br>GDELT news context<extra></extra>",
        showlegend: false,
        marker: {
          size: 11,
          color: "#477b5a",
          opacity: 0.78,
          line: { color: "#ffffff", width: 1.4 },
        },
      };

      for (const country of visibleGdeltCountries || []) {
        if (eventCountries.has(country)) continue;
        const coords = countryCentroids[country];
        if (!coords) continue;
      }
      Plotly.react(map, [countryTrace, telegramTrace, gdeltTrace], globeLayout(), {
        responsive: true,
        displayModeBar: false,
        showlegend: false,
        scrollZoom: false,
      });
    }

    function isWithinTimeWindow(value, hours) {
      if (!hours) return true;
      const timestamp = Date.parse(value);
      if (Number.isNaN(timestamp)) return false;
      return Date.now() - timestamp <= Number(hours) * 60 * 60 * 1000;
    }

    function applyFilters() {
      const search = controls.search.value.trim().toLowerCase();
      let visible = 0;
      const visibleIds = new Set();
      const gdeltLayerEnabled = !controls.channel.value && !controls.type.value && !controls.precision.value && !search;
      const visibleGdeltCountries = new Set();
      if (gdeltLayerEnabled) {
        for (const country of Object.keys(gdeltContext.countries || {})) {
          if (!countryCentroids[country] || !hasGdeltCountryContext(country)) continue;
          if (controls.country.value && controls.country.value !== country) continue;
          visibleGdeltCountries.add(country);
        }
      }
      for (const card of cards) {
        const ok =
          isWithinTimeWindow(card.dataset.messageTime, controls.time.value) &&
          (!controls.country.value || card.dataset.country === controls.country.value) &&
          (!controls.channel.value || card.dataset.channel === controls.channel.value) &&
          (!controls.type.value || card.dataset.type === controls.type.value) &&
          (!controls.precision.value || card.dataset.precision === controls.precision.value) &&
          (!search || card.dataset.search.includes(search));
        card.classList.toggle("hidden", !ok);
        if (ok) {
          visible += 1;
          visibleIds.add(card.dataset.eventId);
        }
      }
      const gdeltOnlyCount = Array.from(visibleGdeltCountries).filter((country) =>
        !Array.from(visibleIds).some((eventId) => (eventById.get(eventId) || {}).country === country)
      ).length;
      count.textContent = `표시 중인 Telegram 신호 ${visible}/${cards.length}개 · GDELT 전용 국가 ${gdeltOnlyCount}개`;
      renderMarkers(visibleIds, visibleGdeltCountries);
      renderFallbackMarkers(visibleIds);
      if (!visibleIds.has(selectedEventId)) {
        const firstVisibleId = visibleIds.values().next().value;
        if (selectedGdeltCountry && (modelItem(selectedGdeltCountry) || visibleGdeltCountries.has(selectedGdeltCountry))) {
          selectCountry(selectedGdeltCountry);
        } else if (controls.mapColor && controls.mapColor.value === "model") {
          const topCountry = Object.values(modelScores.countries || {})
            .sort((a, b) => Number(b.risk_score || 0) - Number(a.risk_score || 0))[0];
          if (topCountry) {
            selectCountry(topCountry.country);
          } else if (firstVisibleId) {
            selectEvent(firstVisibleId);
          }
        } else if (firstVisibleId) {
          selectEvent(firstVisibleId);
        } else if (visibleGdeltCountries.size) {
          selectCountry(visibleGdeltCountries.values().next().value);
        } else {
          selectedEventId = null;
          selectedGdeltCountry = null;
          detail.innerHTML = '<div class="detail-empty">No event matches the current filters.</div>';
        }
      }
    }

    initMap();
    cards.forEach((card) => {
      card.addEventListener("click", () => selectEvent(card.dataset.eventId));
      card.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectEvent(card.dataset.eventId);
        }
      });
    });
    [controls.mapView, controls.mapColor, controls.time, controls.country, controls.channel, controls.type, controls.precision, controls.search]
      .forEach((control) => control.addEventListener("input", applyFilters));
    controls.zoomOut.addEventListener("click", () => updateZoom(-0.15));
    controls.zoomIn.addEventListener("click", () => updateZoom(0.15));
    window.addEventListener("resize", () => {
      resizeMapForMode();
      if (currentVisibleIds.size) applyFilters();
    });
    renderTopRiskTicker();
    window.setInterval(renderTopRiskTicker, 2200);
    applyFilters();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
