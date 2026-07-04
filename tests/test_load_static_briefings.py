"""scripts/load_static_briefings.py 단위/통합 테스트.

Supabase 네트워크 호출은 하지 않는다 — FakeSupabaseClient로 risk_score_history /
llm_briefings 테이블 동작을 인메모리로 흉내낸다. 새 LLM 호출도 하지 않는다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import load_static_briefings as lsb  # noqa: E402


# ──────────────────────────────────────────────
# Fake Supabase client (네트워크 없이 table().select().eq()... 체인 흉내)
# ──────────────────────────────────────────────
class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, table: "_FakeTable", rows):
        self._table = table
        self._rows = rows

    def select(self, *_a, **_k):
        return self

    def eq(self, field, value):
        self._rows = [r for r in self._rows if r.get(field) == value]
        return self

    def order(self, field, desc=False):
        self._rows = sorted(self._rows, key=lambda r: r[field], reverse=desc)
        return self

    def limit(self, n):
        self._rows = self._rows[:n]
        return self

    def insert(self, record):
        self._table.rows.append(record)
        self._rows = [record]
        return self

    def execute(self):
        return _FakeResponse(self._rows)


class _FakeTable:
    def __init__(self, rows):
        self.rows = rows

    def select(self, *_a, **_k):
        return _FakeQuery(self, list(self.rows))

    def insert(self, record):
        return _FakeQuery(self, list(self.rows)).insert(record)


class FakeSupabaseClient:
    def __init__(self, risk_score_history, llm_briefings=None):
        self._tables = {
            "risk_score_history": _FakeTable(risk_score_history),
            "llm_briefings": _FakeTable(llm_briefings or []),
        }

    def table(self, name):
        return self._tables[name]


# ──────────────────────────────────────────────
# Fixture: 작은 합성 HTML (실제 index.html 구조를 최소 재현)
# ──────────────────────────────────────────────
def make_fixture_html(tmp_path: Path) -> Path:
    event_data = [
        {
            "event_id": "abc123",
            "is_conflict_related": True,
            "country": "UKR",
            "channel": "KyivIndependent_official",
            "message_id": "111",
            "message_time": "2026-06-24T14:31:05+00:00",
            "raw_text": "Missile strike reported",
            "url": "https://t.me/KyivIndependent_official/111",
            "severity": 0.9,
            "confidence": 0.95,
            "source_reliability": 0.85,
            "matched_keywords": ["missile", "strike"],
            "event_type": "strike",
            "ko_summary": "우크라이나 미사일 공격 관련 텔레그램 신호",
            "ko_raw_summary": "미사일 공격이 보고되었습니다.",
            "ko_raw_source_limits": "단일 채널 보도로 교차검증 필요",
            "ko_raw_summary_model": "gpt-4.1-mini",
        },
        {
            "event_id": "def456",
            "is_conflict_related": False,
            "country": "",
            "channel": "liveuamap",
            "message_id": "222",
            "message_time": "2026-06-24T10:00:00+00:00",
            "raw_text": "Unrelated update",
            "url": "https://t.me/liveuamap/222",
            "severity": 0.1,
            "confidence": 0.5,
            "matched_keywords": [],
            "ko_raw_summary_model": "gpt-4.1-mini",
        },
    ]
    gdelt_context_data = {
        "generated_at": "2026-06-24T15:27:15.552279+00:00",
        "source": "bigquery_gdelt_titles",
        "table": "conflict-early-warning.conflict_ew.gdelt_titles",
        "days": 7,
        "countries": {
            "UKR": {
                "gdelt_24h": 500,
                "gdelt_7d": 3000,
                "top_keywords": ["missile", "kyiv", "strike"],
                "top_titles": [
                    {"date": "2026-06-24", "title": "Missile strike hits Kyiv",
                     "domain": "example.com", "url": "https://example.com/a"}
                ],
                "anchor_date": "2026-06-24",
                "ko_summary": "우크라이나 전역에 미사일 공격이 이어지고 있습니다.",
                "ko_brief": "우크라이나: 미사일 공격 지속",
                "source_limits": "기사 제목만으로 요약",
                "summary_model": "gpt-4.1-mini",
            }
        },
    }
    model_score_data = {
        "generated_at": "2026-07-01T14:45:12.933327+00:00",
        "source": "bigquery_model_scores",
        "table": "conflict-ew-mvp-20260604.conflict_ew.model_scores",
        "countries": {
            "UKR": {"country": "UKR", "date": "2026-06-29", "run_ts": "2026-07-01T06:27:31Z",
                     "base_pred": 0.7, "onset_prob": 0.7, "calm_flag": 0,
                     "risk_score": 90.0, "onset_percentile": 90.0, "tier": "high"},
            "ZZZ": {"country": "ZZZ", "date": "2026-06-29", "run_ts": "2026-07-01T06:27:31Z",
                     "base_pred": 0.1, "onset_prob": 0.1, "calm_flag": 1,
                     "risk_score": 10.0, "onset_percentile": 10.0, "tier": "low"},
        },
    }

    html = f"""<!doctype html><html><body>
<script id="eventData" type="application/json">{json.dumps(event_data)}</script>
<script id="gdeltContextData" type="application/json">{json.dumps(gdelt_context_data)}</script>
<script id="modelScoreData" type="application/json">{json.dumps(model_score_data)}</script>
</body></html>"""
    path = tmp_path / "fixture.html"
    path.write_text(html, encoding="utf-8")
    return path


def make_risk_row(risk_score_id, iso3, horizon_days=7, alert_level="HIGH",
                   scored_at="2026-06-30T00:00:00+00:00", calibrated_score=None):
    return {
        "risk_score_id": risk_score_id,
        "iso3": iso3,
        "horizon_days": horizon_days,
        "scored_at": scored_at,
        "feature_date": "2026-06-29",
        "model_version_id": "onset_prod_v1",
        "raw_score": 0.75,
        "calibrated_score": calibrated_score,
        "alert_level": alert_level,
        "scoring_run_id": "11111111-1111-1111-1111-111111111111",
    }


# ──────────────────────────────────────────────
# 단위 테스트: 파싱
# ──────────────────────────────────────────────
def test_extract_embedded_json_from_real_html():
    real_html = ROOT / "Telegram" / "index.html"
    if not real_html.exists():
        pytest.skip("Telegram/index.html 없음")
    blocks = lsb.extract_embedded_json(real_html)
    assert set(blocks.keys()) == {"eventData", "gdeltContextData", "modelScoreData"}
    assert isinstance(blocks["eventData"], list)
    assert "countries" in blocks["gdeltContextData"]
    assert "countries" in blocks["modelScoreData"]


def test_extract_embedded_json_missing_block(tmp_path):
    path = tmp_path / "broken.html"
    path.write_text('<script id="eventData" type="application/json">[]</script>', encoding="utf-8")
    with pytest.raises(ValueError):
        lsb.extract_embedded_json(path)


def test_group_events_by_iso3_drops_blank_country():
    events = [{"country": "UKR"}, {"country": ""}, {"country": " lbn "}]
    grouped = lsb.group_events_by_iso3(events)
    assert set(grouped.keys()) == {"UKR", "LBN"}
    assert len(grouped["UKR"]) == 1


# ──────────────────────────────────────────────
# 단위 테스트: 매핑/해시
# ──────────────────────────────────────────────
def test_score_value_prefers_calibrated():
    row = make_risk_row(1, "UKR", calibrated_score=55.5)
    assert lsb.score_value(row) == 55.5
    row2 = make_risk_row(2, "UKR", calibrated_score=None)
    assert lsb.score_value(row2) == 0.75


def test_pick_risk_summary_prefers_gdelt_brief():
    gdelt_ctx = {"ko_brief": "brief", "ko_summary": "summary"}
    assert lsb.pick_risk_summary(gdelt_ctx, []) == "brief"
    assert lsb.pick_risk_summary({"ko_summary": "summary"}, []) == "summary"
    assert lsb.pick_risk_summary(None, [{"ko_summary": "tg summary"}]) == "tg summary"
    assert lsb.pick_risk_summary(None, []) is None


def test_pick_llm_model_prefers_gdelt_summary_model():
    assert lsb.pick_llm_model({"summary_model": "gpt-4.1-mini"}, []) == "gpt-4.1-mini"
    assert lsb.pick_llm_model(None, [{"ko_raw_summary_model": "gpt-4.1-mini"}]) == "gpt-4.1-mini"
    assert lsb.pick_llm_model(None, []) is None


def test_input_hash_deterministic_and_sensitive_to_change():
    row = make_risk_row(1, "UKR")
    h1 = lsb.compute_input_hash(row, "s", [], {}, {}, [], "v1")
    h2 = lsb.compute_input_hash(row, "s", [], {}, {}, [], "v1")
    assert h1 == h2
    h3 = lsb.compute_input_hash(row, "different", [], {}, {}, [], "v1")
    assert h1 != h3


# ──────────────────────────────────────────────
# 통합 테스트: run() 전체 흐름 (dry-run / insert / skip)
# ──────────────────────────────────────────────
def test_run_dry_run_reports_expected_countries(tmp_path, capsys):
    html_path = make_fixture_html(tmp_path)
    risk_rows = [make_risk_row(101, "UKR", alert_level="HIGH")]
    client = FakeSupabaseClient(risk_score_history=risk_rows)

    stats = lsb.run(
        html_path=html_path, limit=None, alert_level="ALL", horizon_days=7,
        prompt_version="static-export-v1", dry_run=True, client=client,
    )

    # modelScoreData에는 UKR, ZZZ 2개국이 있지만 risk_score_history에는 UKR만 존재
    assert stats["target"] == 2
    assert stats["created"] == 1
    assert stats["missing_risk_row"] == 1
    assert client.table("llm_briefings").rows == []  # dry-run은 실제 insert 없음


def test_run_insert_then_skip_duplicate(tmp_path):
    html_path = make_fixture_html(tmp_path)
    risk_rows = [make_risk_row(101, "UKR", alert_level="HIGH")]
    client = FakeSupabaseClient(risk_score_history=risk_rows)

    stats1 = lsb.run(
        html_path=html_path, limit=None, alert_level="ALL", horizon_days=7,
        prompt_version="static-export-v1", dry_run=False, client=client,
    )
    assert stats1["created"] == 1
    assert len(client.table("llm_briefings").rows) == 1
    inserted = client.table("llm_briefings").rows[0]
    assert inserted["risk_score_id"] == 101
    assert inserted["llm_provider"] == "openai"
    assert inserted["llm_model"] == "gpt-4.1-mini"
    assert inserted["generation_status"] == "completed"
    assert inserted["risk_summary"] == "우크라이나: 미사일 공격 지속"
    assert inserted["risk_score"] == 0.75  # calibrated_score 없음 → raw_score

    # 동일 HTML/동일 risk row로 재실행 → input_hash 동일 → skip
    stats2 = lsb.run(
        html_path=html_path, limit=None, alert_level="ALL", horizon_days=7,
        prompt_version="static-export-v1", dry_run=False, client=client,
    )
    assert stats2["created"] == 0
    assert stats2["skipped_duplicate"] == 1
    assert len(client.table("llm_briefings").rows) == 1  # 중복 적재 안 됨


def test_run_alert_level_filter_excludes_non_matching(tmp_path):
    html_path = make_fixture_html(tmp_path)
    risk_rows = [make_risk_row(101, "UKR", alert_level="LOW")]
    client = FakeSupabaseClient(risk_score_history=risk_rows)

    stats = lsb.run(
        html_path=html_path, limit=None, alert_level="HIGH", horizon_days=7,
        prompt_version="static-export-v1", dry_run=True, client=client,
    )
    # alert_level=HIGH 필터로 인해 UKR(LOW)이 조회되지 않아 risk row 없음 처리됨
    assert stats["created"] == 0
    assert stats["missing_risk_row"] == 2


# ──────────────────────────────────────────────
# evidence 없는 국가 skip (CIV briefing_id=5 null 사고 재현 방지)
# ──────────────────────────────────────────────
def test_has_evidence():
    assert lsb.has_evidence(gdelt_ctx={"ko_brief": "x"}, events=[]) is True
    assert lsb.has_evidence(gdelt_ctx=None, events=[{"country": "UKR"}]) is True
    assert lsb.has_evidence(gdelt_ctx=None, events=[]) is False


def test_run_skips_country_with_no_evidence_dry_run(tmp_path, capsys):
    html_path = make_fixture_html(tmp_path)
    # fixture 안 modelScoreData에는 ZZZ가 있지만 gdeltContextData/eventData에는 ZZZ가 없음
    # (CIV가 gdeltContextData/eventData 어디에도 없었던 실제 사고와 동일한 상황)
    risk_rows = [make_risk_row(101, "UKR", alert_level="HIGH"), make_risk_row(102, "ZZZ", alert_level="HIGH")]
    client = FakeSupabaseClient(risk_score_history=risk_rows)

    stats = lsb.run(
        html_path=html_path, limit=None, alert_level="ALL", horizon_days=7,
        prompt_version="static-export-v1", dry_run=True, client=client,
    )

    assert stats["created"] == 1          # UKR만 생성 예정
    assert stats["skipped_no_evidence"] == 1  # ZZZ는 evidence 없음으로 skip
    assert stats["missing_risk_row"] == 0
    assert client.table("llm_briefings").rows == []

    out = capsys.readouterr().out
    assert "[skip] ZZZ: Telegram/GDELT evidence 없음" in out


def test_run_skips_country_with_no_evidence_real_insert(tmp_path):
    """dry-run과 실제 insert가 동일한 skip 정책을 쓰는지 확인.
    evidence 없는 국가는 실제 insert 모드에서도 llm_briefings에 행이 생기면 안 된다."""
    html_path = make_fixture_html(tmp_path)
    risk_rows = [make_risk_row(102, "ZZZ", alert_level="HIGH")]
    client = FakeSupabaseClient(risk_score_history=risk_rows)

    stats = lsb.run(
        html_path=html_path, limit=None, alert_level="ALL", horizon_days=7,
        prompt_version="static-export-v1", dry_run=False, client=client,
    )

    assert stats["created"] == 0
    assert stats["skipped_no_evidence"] == 1
    assert client.table("llm_briefings").rows == []  # risk_summary/llm_model이 null인 행이 생성되지 않음
