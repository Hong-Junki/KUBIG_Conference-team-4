# Data Pipeline Audit

**작성일**: 2026-06-25  
**조사자**: Claude Code (데이터 엔지니어 역할)  
**목적**: 증분 수집 파이프라인 설계 전 현황 파악

---

## 1. 현재 데이터 파이프라인 요약

```
외부 소스                          로컬 parquet              BigQuery (conflict-ew-mvp-20260604)
───────────────────────────────    ─────────────────────    ─────────────────────────────────────
ACLED API             →  ACLED 수집기  →  input/raw/acled/     (BQ 적재 없음)
GDELT BQ public       →  GDELT 수집기  →  input/raw/gdelt/     →  gdelt_processed_events
GDELT GKG BQ public   →  GKG 수집기   →  (로컬 저장 없음)      →  gdelt_titles
yfinance + FRED       →  경제지표 수집기 →  input/raw/economic/  (BQ 적재 없음)
                                            ↓
                                      src/process/ (merge_raw → preprocess → feature_builder →
                                                    label_builder → build_dataset)
                                            ↓
                                      input/processed/
                                            ↓
                                      modeling_full_dataset (BQ, snapshot, 수동 업로드)
                                            ↓
                                      modeling_acled_free_view (BQ View, snapshot 기반)
```

---

## 2. 기존 수집 코드별 역할

### 2-1. GDELT 이벤트 수집기 (`src/collect/gdelt_collector.py`)

| 항목 | 내용 |
|------|------|
| 소스 | `gdelt-bq.gdeltv2.events_partitioned` (공개 BQ) |
| 수집 함수 | `collect_historical_bq()`, `collect_recent_doc()` |
| 진입점 | `run_historical.py --sources gdelt` |
| 기본 수집 기간 | `COLLECT_START`(2014-01-01) ~ `COLLECT_END`(2026-03-31) |
| 대상 국가 | 57개국 (주석), 실제 58개국 (코드 기준) |
| 수집 컬럼 | GLOBALEVENTID, SQLDATE, ActionGeo_CountryCode, EventCode, EventRootCode, QuadClass, GoldsteinScale, NumMentions, NumArticles, AvgTone |
| 로컬 저장 | `input/raw/gdelt/{iso3}.parquet` (국가별) |
| BQ 직재 적재 | **없음** (로컬 parquet만 저장) |
| 중복 제거 키 | GLOBALEVENTID |
| checkpoint | `input/raw/gdelt/.ckpt_historical.json` (월 단위) |
| idempotency | 체크포인트 기반 (월 단위 완료 추적) |
| 문제점 | 1) BQ에 직접 적재하지 않음<br>2) 로컬 checkpoint는 GA runner에서 유지 안 됨<br>3) COLLECT_END 하드코딩 |
| 운영 수정 사항 | BQ 직접 적재 + MAX(date) 기반 증분 기간 계산 |

### 2-2. GDELT GKG 기사 제목 수집기 (`src/collect/collect_gdelt_titles_gkg.py`)

| 항목 | 내용 |
|------|------|
| 소스 | `gdelt-bq.gdeltv2.gkg_partitioned` (공개 BQ) |
| 수집 함수 | `collect_titles_gkg()` |
| 진입점 | `python -m src.collect.collect_gdelt_titles_gkg --start ... --end ...` |
| 기본 수집 기간 | CLI 필수 인자 |
| 대상 국가 | 58개국 |
| BQ 직재 적재 | **있음** — DELETE+INSERT 방식 |
| 대상 BQ | ~~`conflict-early-warning`~~.conflict_ew.gdelt_titles → **버그** |
| 중복 제거 키 | iso3 + url (SQL 내 ROW_NUMBER) |
| checkpoint | `input/raw/gdelt_titles/.ckpt_titles_gkg.json` (월 단위) |
| **치명적 버그** | `TARGET_PROJECT = "conflict-early-warning"` 하드코딩 — 실제 프로젝트는 `conflict-ew-mvp-20260604` |
| DELETE+INSERT 위험 | DELETE 후 INSERT 실패 시 해당 월 데이터 공백 발생 가능 |
| 운영 수정 사항 | 프로젝트명 env 변수화 + MERGE 방식으로 전환 권장 |

### 2-3. ACLED 수집기 (`src/collect/acled_collector.py`)

| 항목 | 내용 |
|------|------|
| 소스 | ACLED REST API (OAuth 2.0) |
| 수집 함수 | `collect_historical()`, `collect_recent()` |
| 진입점 | `run_historical.py --sources acled` |
| BQ 직접 적재 | **없음** (로컬 parquet만) |
| 중복 제거 키 | event_id_cnty |
| 운영 역할 | **라벨 생성용** (y_escalation) — 운영 feature가 아님 |
| checkpoint | `input/raw/acled/.ckpt_historical.json` |
| 판단 | ACLED는 7일 lag + 미래 라벨용 → daily 운영 스케줄에서 제외, delayed-label workflow 별도 설계 권장 |

### 2-4. 경제지표 수집기 (`src/collect/economic_collector.py`)

| 항목 | 내용 |
|------|------|
| 소스 | yfinance (VIX/WTI/Gold/DXY) + fredapi (STLFSI4) |
| 수집 함수 | `collect_historical()`, `collect_recent()` |
| BQ 직접 적재 | **없음** (로컬 parquet만) |
| 저장 경로 | `input/raw/economic/indicators.parquet` |
| 중복 제거 키 | date |
| 운영 역할 | 운영 feature (econ_vix, econ_wti 등) |
| 주말·휴일 처리 | 수집 안 됨 (거래일만). ffill은 feature_builder에서 처리 |
| STLFSI4 지연 | 매주 목요일 발표, 주간 데이터 |
| 운영 수정 사항 | BQ `economic_daily` 직접 적재 + overlap 재수집 |

---

## 3. 실제 BigQuery 리소스 현황

**프로젝트**: `conflict-ew-mvp-20260604`  
**데이터셋**: `conflict_ew`

| 테이블 | 타입 | 파티션 | Clustering | 행 수 | 최초 적재일 | 최종 적재일 |
|--------|------|---------|------------|-------|------------|------------|
| `gdelt_processed_events` | TABLE | MONTH(event_date) | iso3, EventRootCode, QuadClass | 309,533,500 | 2014-01-01 | 2026-03-31 |
| `economic_daily` | TABLE | — | — | 3,101 | 2014-01-02 | 2026-03-30 |
| `gdelt_titles` | TABLE | MONTH(date) | iso3 | 859,303,212 | 2015-02-17 | 2026-05-29 |
| `modeling_full_dataset` | TABLE | MONTH(date) | country | 259,260 | 2014-01-01 | 2026-03-28 |
| `modeling_acled_free_view` | VIEW | — | — | (snapshot 기반) | 2014-01-01 | 2026-03-28 |
| `gkg_anchor_vectors` | TABLE | — | — | 미조회 | — | — |
| `gkg_embeddings` | TABLE | MONTH(date) | iso3 | 미조회 | — | — |

### 각 테이블 스키마 요약

**gdelt_processed_events**: GLOBALEVENTID, SQLDATE, ActionGeo_CountryCode, EventCode, EventRootCode, QuadClass, GoldsteinScale, NumMentions, NumArticles, AvgTone, event_date(TIMESTAMP), iso3

**economic_daily**: date, VIX, WTI, Gold, DXY, STLFSI4, VIX/WTI/Gold/DXY/STLFSI4_pct_change, econ_volatility_proxy

**gdelt_titles**: date(REQUIRED), iso3(REQUIRED), title, url(REQUIRED), domain, language, v2tone_avg, v2themes, v2persons

**modeling_full_dataset**: date, country, 22개 ACLED feature, 19개 GDELT feature, 15개 econ feature, y_escalation 등 64 columns

**modeling_acled_free_view**: SELECT from modeling_full_dataset (date, country, 19개 GDELT feature, 15개 econ feature, y_escalation) — **raw 데이터 업데이트 시 자동 반영 안 됨**

---

## 4. 문서와 실제 상태의 차이

| 항목 | 문서 기술 | 실제 확인값 |
|------|----------|------------|
| 대상 국가 수 | 57개 (코드 주석), 58개 (README) | **58개** (config.py COUNTRIES 목록, BQ 실제 데이터) |
| gdelt_processed_events | "약 3억 row" | 309,533,500행 (정확) |
| modeling_full_dataset | "259,260 rows, 64 columns" | 정확 |
| modeling_full_dataset 기간 | "2014-01-01 ~ 2026-03-28" | 정확 |
| modeling_acled_free_view | "raw 데이터 갱신 시 반영" (암묵적) | **미반영** — snapshot 기반 VIEW |
| collect_gdelt_titles_gkg.py TARGET_PROJECT | conflict-early-warning (코드) | conflict-ew-mvp-20260604 (실제 운영) |
| GCP_PROJECT_ID | .env에 없음 | conflict-ew-mvp-20260604 |
| ACLED BQ 적재 | 문서에 언급 없음 | BQ 적재 없음 (로컬 parquet만) |
| 경제지표 BQ 적재 | 문서에 언급 없음 | `economic_daily` 테이블 존재 (수동 업로드 추정) |

---

## 5. 각 소스의 최종 적재일

| 소스 | BQ 테이블 | MAX(date) | 증분 시작 권장일 |
|------|----------|-----------|----------------|
| GDELT 이벤트 | gdelt_processed_events | 2026-03-31 | 2026-03-29 (3일 overlap) |
| 경제지표 | economic_daily | 2026-03-30 | 2026-03-21 (10일 overlap) |
| GDELT GKG 제목 | gdelt_titles | 2026-05-29 | 2026-05-27 (3일 overlap) |
| ACLED | (BQ 없음) | N/A | delayed-label workflow 별도 |

---

## 6. 증분 수집 시작일 결정 방식

```
last_loaded_date = MAX(date_col) FROM target_table
collection_start = last_loaded_date - overlap_days
collection_end   = 데이터 소스에서 이용 가능한 최신 날짜 (보통 date.today() - 1)
```

권장 overlap:
- GDELT 이벤트: 3일 (15분 단위 업데이트, 이벤트 수정 거의 없음)
- 경제지표: 10일 (주간 STLFSI4 + 수정치 반영)
- GDELT GKG 제목: 3일

---

## 7. 원천 데이터 → feature → 모델 lineage

```
[1] gdelt-bq.gdeltv2.events_partitioned (공개 BQ)
     ↓ gdelt_collector.collect_historical_bq()
[2] input/raw/gdelt/{iso3}.parquet (로컬)
     ↓ (gdelt_processed_events는 별도 BQ 업로드로 추정)
[2'] conflict-ew-mvp-20260604.conflict_ew.gdelt_processed_events (BQ 원천)
     ↓ feature_builder._build_gdelt_features()
[3] input/processed/gdelt/{iso3}.parquet  →  rolling (7d/14d/30d) 집계
     ↓
[4] input/processed/features/features.parquet
     ↓ build_dataset.py
[5] train/val/test parquet
     ↓ (수동 BQ 업로드)
[6] modeling_full_dataset (BQ snapshot)
     ↓ (View)
[7] modeling_acled_free_view (SELECT from snapshot)
     ↓
[8] 모델 입력 (GDELT 19 feature + econ 15 feature)
```

**주의**: 단계 [6]은 snapshot — raw 업데이트가 자동으로 [6]~[7]에 반영되지 않음.

### GDELT feature 생성 규칙 (feature_builder 기반)
- GoldsteinScale rolling mean/std (7d, 14d, 30d)
- AvgTone rolling mean (7d, 14d, 30d)
- NumMentions rolling sum (7d, 14d, 30d)
- event_count rolling (7d, 14d, 30d)
- QuadClass 비율 (1~4)
- lag 없음 (GDELT는 당일 데이터 사용 가능)

### economic feature 생성 규칙
- VIX/WTI/Gold/DXY 일별 값 + 1일/7일 pct change
- STLFSI4 (주간 ffill + 1일/7일 pct change)
- `econ_volatility_proxy` = VIX_pct_change * WTI_pct_change 부호 등

---

## 8. 재사용 가능한 기존 코드

| 코드 | 재사용 가능 부분 |
|------|----------------|
| `gdelt_collector._build_query()` | GDELT BQ 쿼리 생성 (재사용) |
| `gdelt_collector._run_bq_query()` | BQ 실행 + retry (재사용) |
| `gdelt_collector.dry_run_query()` | 스캔량 확인 (재사용) |
| `gdelt_collector._parse_sqldate()` | SQLDATE 파싱 (재사용) |
| `gdelt_collector._map_fips_to_iso3()` | FIPS→ISO3 매핑 (재사용) |
| `economic_collector.collect_yfinance()` | yfinance 수집 (재사용) |
| `economic_collector.collect_fred()` | FRED 수집 (재사용) |
| `utils.retry` | 재시도 데코레이터 (재사용) |
| `utils.get_logger()` | 로거 (재사용) |
| `config.COUNTRIES` | 국가 목록 (재사용) |
| `config.GDELT_BQ_TABLE` | GDELT 소스 테이블명 (재사용) |

---

## 9. 수정이 필요한 코드

| 파일 | 문제 | 수정 |
|------|------|------|
| `collect_gdelt_titles_gkg.py` | TARGET_PROJECT 하드코딩 (`conflict-early-warning`) | `GCP_PROJECT_ID` env 변수화 |
| `config.py` | COLLECT_END 하드코딩 (2026-03-31) | 증분 수집에서는 BQ MAX(date) 사용 |
| `gdelt_collector.py` | BQ 적재 없음 | 증분 수집기에서 BQ MERGE 추가 |
| `economic_collector.py` | BQ 적재 없음 | 증분 수집기에서 BQ MERGE 추가 |
| `.gitignore` | `docs/` 제외됨 | 제외 해제 (수정 완료) |
| `.gitignore` | 새 SA JSON 패턴 누락 | `conflict-ew-mvp-*.json` 추가 (수정 완료) |

---

## 10. ACLED 운영 사용 여부 판단

**판단: ACLED는 daily 운영 스케줄에서 제외**

근거:
1. `modeling_acled_free_view`에 ACLED feature가 없음 (GDELT + 경제지표만)
2. feature_builder.py에서 ACLED feature는 7일 lag 적용 + `acled_missing_mask` 컬럼 존재
3. ACLED는 주 1회 갱신 (실시간성 없음)
4. y_escalation 라벨 생성에 ACLED가 사용되므로 미래 정보 → 운영 시점 feature로 부적합
5. ACLED API 접근 권한 확인 필요 (학술용 계정 여부)

**권장**: ACLED는 별도 delayed-label workflow로 분리. 새 라벨 데이터가 available해지면 주 1회 또는 월 1회 별도 실행.

---

## 11. GitHub Actions 적용 가능성

**가능** — 단, 다음 조건 충족 필요:

- GCP 서비스 계정 JSON 또는 Workload Identity Federation 설정
- 필요한 GitHub Secrets 등록
- BQ에 WRITE 권한 부여된 서비스 계정 사용
- workflow는 `workflow_dispatch` 로만 시작 (schedule은 검증 후 주석 해제)

**비용 추정** (증분 수집):
- GDELT events (3일분): gdelt-bq public table, ~50GB/month 스캔 → $0.25/회
- GDELT GKG (3일분): ~150GB/회 스캔 → $0.75/회  
- 경제지표: API 기반, BQ 스캔 없음

---

## 12. 비용 및 보안 위험

### 보안 위험

1. **`.env`에 credential 경로 기록**: `conflict-early-warning-4672e791d960.json` 파일명 노출
   - 현재 `.env`는 gitignore되어 있으나, 서비스 계정 JSON 자체도 반드시 gitignore 확인
   - **조치 필요**: `conflict-early-warning` 프로젝트의 credential rotation 권장
   
2. **구 프로젝트 (`conflict-early-warning`) 서비스 계정**:
   - 새 프로젝트 `conflict-ew-mvp-20260604`에 접근 가능한지 확인 필요
   - 새 프로젝트 전용 서비스 계정 생성 권장

3. **GitHub Secrets에 SA JSON 저장 시**:
   - JSON 내용 절대 로그 출력 금지
   - OIDC + Workload Identity Federation 방식 권장

### 비용 위험
- `gdelt_processed_events` 전체 스캔 시: 309M rows × 12 bytes × 12 col ≈ 수십 TB
- 반드시 partition filter (`_PARTITIONTIME`) + `WHERE event_date BETWEEN` 사용
- 월 무료 한도: 1TB/월. 일별 증분은 약 50GB 이하 예상

---

## 13. 추천 구현 구조

```
src/collect/incremental/
  __init__.py
  state.py              ← BQ MAX(date) 기반 state 조회
  bigquery_io.py        ← staging + MERGE 유틸
  collect_gdelt_incremental.py   ← GDELT 증분 수집 + BQ 적재
  collect_economic_incremental.py ← 경제지표 증분 수집 + BQ 적재
  run_incremental.py    ← CLI 오케스트레이터
```

**적재 방식**: staging table → validation → MERGE → staging drop

**상태 관리**: BigQuery `MAX(date)` (로컬 checkpoint 사용 안 함)

---

## 14. 아직 확인하지 못한 사항

| 항목 | 이유 |
|------|------|
| `gkg_anchor_vectors`, `gkg_embeddings` 행 수/기간 | 운영 pipeline과 무관, 별도 실험 테이블로 판단 |
| ACLED API 실제 접근 가능 여부 | 학술 계정 제한 확인 필요 |
| `conflict-early-warning` 프로젝트 BQ 테이블 현황 | 구 프로젝트 조회 권한 불명 |
| `modeling_full_dataset`의 econ 컬럼 계산 로직 | feature_builder 전체 로직 심층 분석 필요 |
| GitHub Actions runner에서 BQ 접근 권한 | 실제 SA 권한 설정 확인 필요 |
