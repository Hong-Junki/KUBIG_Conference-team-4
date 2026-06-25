# 증분 데이터 수집 파이프라인

**작성일**: 2026-06-25  
**현재 schedule 상태**: **비활성** (workflow_dispatch 수동 실행만 가능)

---

## 1. 목적

기존에 수집된 과거 데이터를 재수집하지 않고, 최신 데이터만 증분으로 BigQuery에 적재한다.  
GitHub Actions의 수동 실행으로 1회 검증하고, 향후 cron 스케줄 활성화 가능한 구조로 구성한다.

---

## 2. 전체 아키텍처

```
GitHub Actions (workflow_dispatch)
  │
  ├─ 1. unit test (pytest)
  ├─ 2. GCP 인증 (SA key 또는 WIF)
  ├─ 3. 권한 검증 (임시 테이블 create/write/merge/delete)
  ├─ 4. BQ 현황 확인 (읽기 전용)
  ├─ 5. dry-run (스캔량 확인)
  └─ 6. 증분 수집 실행 (단일 CLI: run_incremental --sources gdelt economic gdelt_titles)
       │
       ├─ GDELT 이벤트 수집기
       │    gdelt-bq.gdeltv2.events_partitioned (공개 BQ)
       │    → local DataFrame
       │    → staging table
       │    → MERGE INTO gdelt_processed_events (key: GLOBALEVENTID)
       │
       ├─ 경제지표 수집기
       │    yfinance (VIX/WTI/Gold/DXY) + FRED (STLFSI4)
       │    → local DataFrame
       │    → staging table
       │    → MERGE INTO economic_daily (key: date, upsert)
       │
       └─ GDELT GKG 기사 수집기
            gdelt-bq.gdeltv2.gkg_partitioned (공개 BQ)
            → staging table (월 단위 INSERT INTO staging)
            → MERGE INTO gdelt_titles (key: iso3+url, upsert)
```

---

## 3. 데이터 소스별 수집 주기

| 소스 | 원천 | 수집 주기 (권장) | BQ target table |
|------|------|-----------------|----------------|
| GDELT 이벤트 | gdelt-bq.gdeltv2.events_partitioned | 일 1회 (cron 활성화 시) | gdelt_processed_events |
| 경제지표 | yfinance + FRED | 일 1회 | economic_daily |
| GDELT GKG 기사 | gdelt-bq.gdeltv2.gkg_partitioned | 일 1회 | gdelt_titles |
| ACLED | ACLED API | **별도 delayed-label workflow** | (이 파이프라인 미담당) |

> 세 소스(`gdelt`, `economic`, `gdelt_titles`)는 **단일 CLI, 단일 workflow**로 실행한다.  
> `gdelt_titles`는 더 이상 별도 실행이 아니다.

> 이 파이프라인은 "주기적 배치 수집"이다. GitHub Actions cron의 최소 간격은 5분이며, 진정한 실시간 수집이 아니다.

---

## 4. 데이터 지연 특성

| 소스 | 지연 특성 |
|------|----------|
| GDELT 이벤트 | 15분 단위 업데이트. 하루 치 데이터는 익일 00:00 UTC 이후 완전. |
| VIX/WTI/Gold/DXY | 당일 거래 종료 후. 주말·미국 공휴일은 데이터 없음. |
| STLFSI4 | 주간 발표 (매주 목요일). 2~3주 지연으로 수정치 발표. |

---

## 5. raw target table (담당 범위)

| 테이블 | MERGE 키 | 파티션 | 수정치 처리 |
|--------|---------|---------|-----------|
| gdelt_processed_events | GLOBALEVENTID | MONTH(event_date) | 없음 (삽입 전용) |
| economic_daily | date | 없음 | UPDATE (수정치 반영) |
| gdelt_titles | iso3 + url | 없음 | UPDATE (v2tone 등 수정치 반영) |

**이 파이프라인이 담당하지 않는 테이블** (절대 수정하지 않음):  
`modeling_full_dataset`, `modeling_acled_free_view`, feature 테이블, ACLED 관련 테이블, 모델 추론 결과

---

## 6. 증분 기간 계산 방식

```
last_loaded_date = MAX(date_col) FROM target_table  ← BQ에서 실제 조회
collection_start = last_loaded_date - overlap_days
collection_end   = date.today() - 1                 ← 어제 날짜 (오늘 데이터 미완성)
```

**날짜를 코드 상수로 하드코딩하지 않는다.** 항상 BQ의 실제 MAX(date)에서 계산.

---

## 7. overlap 정책

| 소스 | 기본 overlap | 이유 |
|------|------------|------|
| GDELT 이벤트 | 3일 | 늦게 인덱싱된 이벤트 재수집 |
| 경제지표 | 10일 | STLFSI4 수정치 + 공휴일 보완 |
| GDELT GKG | 3일 | 늦게 입수된 기사 재수집 |

---

## 8. dedup / MERGE 방식

```sql
-- GDELT 이벤트 (삽입 전용, 수정 없음)
MERGE `proj.conflict_ew.gdelt_processed_events` T
USING staging S
ON T.GLOBALEVENTID = S.GLOBALEVENTID
WHEN NOT MATCHED THEN INSERT (...)

-- 경제지표 (수정치 반영, UPSERT)
MERGE `proj.conflict_ew.economic_daily` T
USING staging S
ON T.date = S.date
WHEN MATCHED THEN UPDATE SET T.VIX = S.VIX, ...
WHEN NOT MATCHED THEN INSERT (...)

-- GDELT GKG 기사 (수정치 반영, UPSERT)
-- MERGE key = (iso3, url)
-- iso3+url 조합은 실데이터 기준 중복 0건 (2026-05-01 이후 검증 완료)
-- url 단독은 1,404,287건 중복 (동일 기사가 복수 국가에 매핑) → url 단독 key 불가
MERGE `proj.conflict_ew.gdelt_titles` T
USING staging S
ON T.iso3 = S.iso3 AND T.url = S.url
WHEN MATCHED THEN UPDATE SET T.v2tone_avg = S.v2tone_avg, ...
WHEN NOT MATCHED THEN INSERT (...)
```

staging table 이름: `_staging_{table}_{run_id}` (실행마다 고유)

---

## 9. 로컬 실행법

```bash
# 환경 설정
cp .env.example .env  # GCP_PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS 등 입력
pip install -r requirements.txt

# BQ 현황 확인 (읽기 전용)
python -m src.collect.incremental.run_incremental --validate-only

# 권한 검증 (임시 테이블 create/write/merge/delete, raw table 변경 없음)
python -m src.collect.incremental.run_incremental --validate-permissions

# dry-run (스캔량만 확인, 실제 BQ 쓰기 없음)
python -m src.collect.incremental.run_incremental \
  --sources gdelt economic gdelt_titles --dry-run

# 실제 수집 (세 소스 단일 CLI로 실행)
python -m src.collect.incremental.run_incremental \
  --sources gdelt economic gdelt_titles

# 날짜 범위 강제 지정 (단일 소스)
python -m src.collect.incremental.run_incremental \
  --sources gdelt \
  --start 2026-04-01 \
  --end 2026-04-10
```

---

## 10. dry-run 실행법

```bash
python -m src.collect.incremental.run_incremental --dry-run
```

dry-run은 다음을 수행하고 멈춘다:
- BQ MAX(date) 조회
- GDELT 쿼리 스캔량 예측 (actual 실행 없음)
- 경제지표: API 수집 생략 안내
- BQ 쓰기 없음

---

## 11. GitHub Actions 수동 실행법

1. GitHub 저장소 → **Actions** 탭
2. **Incremental Data Collection** workflow 선택
3. **Run workflow** 클릭
4. 옵션 입력:
   - `sources`: `gdelt economic gdelt_titles` (기본값 — 세 소스 모두)
   - `dry_run`: `false` (실제 수집) 또는 `true` (스캔량 확인만)
   - `overlap_days`: 빈칸 (소스별 기본값 사용)
5. **Run workflow** 확인

---

## 12. GitHub Secrets 설정법

| Secret 이름 | 내용 | 설정 위치 |
|------------|------|----------|
| `GCP_PROJECT_ID` | `conflict-ew-mvp-20260604` | GitHub repo → Settings → Secrets |
| `GCP_SA_KEY` | 서비스 계정 JSON 전체 내용 | GitHub repo → Settings → Secrets |
| `FRED_API_KEY` | FRED API 키 | GitHub repo → Settings → Secrets |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | WIF provider 경로 (WIF 사용 시) | — |
| `GCP_SERVICE_ACCOUNT` | 서비스 계정 이메일 (WIF 사용 시) | — |

> **SA JSON(GCP_SA_KEY)은 절대 코드나 로그에 출력하지 않는다.**

**설정 절차**:
```
GitHub repo → Settings → Secrets and variables → Actions → New repository secret
```

---

## 13. cron 활성화 방법

수동 검증이 완료된 후:

1. `.github/workflows/incremental-data-collection.yml` 파일 열기
2. 다음 주석을 해제:
   ```yaml
   # schedule:
   #   - cron: "20 1 * * *"
   ```
   → 아래로 변경:
   ```yaml
   schedule:
     - cron: "20 1 * * *"
   ```
3. UTC 01:20 = KST 10:20 (매일 아침 한국 시간 10시 20분 실행)
4. 커밋 후 push

---

## 14. cron 비활성화 방법

1. `.github/workflows/incremental-data-collection.yml` 파일 열기
2. `schedule:` 블록을 다시 주석 처리
3. 커밋 후 push
4. 또는 GitHub Actions UI → workflow → Disable workflow

---

## 15. validation 항목

적재 후 자동으로 다음을 검증한다:

| 항목 | 실패 기준 |
|------|----------|
| 총 row 수 | 0행 |
| null date | 존재하면 실패 |
| future date | MAX(date) > CURRENT_DATE |
| 국가 수 | GDELT: 0개국 |

검증 실패 시 workflow는 non-zero exit code로 종료된다.

---

## 16. 오류 복구 방법

### staging table이 남아있는 경우
```sql
-- BQ 콘솔에서 확인
SELECT table_name FROM `conflict-ew-mvp-20260604.conflict_ew.INFORMATION_SCHEMA.TABLES`
WHERE table_name LIKE '_staging_%';

-- 삭제
DROP TABLE `conflict-ew-mvp-20260604.conflict_ew._staging_...`;
```

### 중복 데이터가 들어간 경우
```sql
-- GDELT: GLOBALEVENTID 기준 중복 확인
SELECT GLOBALEVENTID, COUNT(*) as cnt
FROM `conflict-ew-mvp-20260604.conflict_ew.gdelt_processed_events`
WHERE event_date >= TIMESTAMP("2026-04-01")
GROUP BY 1 HAVING cnt > 1;
```

### 잘못된 날짜 범위를 수집한 경우
```sql
-- 특정 날짜 범위 삭제 후 재수집
-- 주의: 반드시 partition filter 사용
DELETE FROM `conflict-ew-mvp-20260604.conflict_ew.gdelt_processed_events`
WHERE event_date BETWEEN TIMESTAMP("2026-04-01") AND TIMESTAMP("2026-04-10");
```

---

## 17. 비용 관리

- GDELT 공개 테이블 조회는 **BigQuery 무료 한도 1TB/월** 소진
- 증분 수집 (3일분): 예상 ~50GB/회 → $0.25/회 (초과분 $5/TB)
- 비용 안전장치:
  - `--max-gb-per-query 100` (월 단위 쿼리 한도)
  - `--max-bytes-billed` MERGE 쿼리 한도
  - dry-run으로 사전 확인 필수

---

## 18. credential rotation 안내

### 서비스 계정 JSON 교체 (GCP_SA_KEY)
1. GCP Console → IAM & Admin → Service Accounts
2. 새 JSON 키 생성 (기존 키 삭제 전 새 키 먼저 적용)
3. GitHub Secrets에서 GCP_SA_KEY 값 교체
4. 구 JSON 파일 로컬에서 삭제 (`conflict-early-warning-4672e791d960.json` 포함)
5. GCP Console에서 구 키 비활성화 및 삭제

> **주의**: `conflict-early-warning` 프로젝트의 기존 서비스 계정(`conflict-early-warning-4672e791d960.json`)은  
> 로컬 `.env`에 경로가 기록되어 있다. 해당 파일이 분실되지 않았는지 확인하고,  
> 더 이상 필요 없다면 GCP에서 키를 비활성화한다. **Credential rotation을 강력히 권장한다.**

---

## 19. feature/modeling table 갱신과의 관계

이 파이프라인은 **raw 원천 데이터 적재만 담당**한다.

```
[이 파이프라인]
  gdelt_processed_events  ← GDELT 이벤트
  economic_daily          ← yfinance + FRED
  gdelt_titles            ← GDELT GKG 기사
      ↓  (별도 단계, 현재 수동)
[feature 갱신]  feature_builder.py 실행 → 새 feature parquet 생성
      ↓  (별도 단계, 현재 수동)
[modeling 갱신]  modeling_full_dataset BQ 업로드
      ↓
[view 갱신]      modeling_acled_free_view 자동 반영 (View이므로)
```

`modeling_full_dataset`은 raw 데이터 갱신 시 자동으로 업데이트되지 않는다.  
향후 feature 갱신 파이프라인은 별도 workflow로 구현 예정.

---

## 20. ACLED을 별도 운영하는 이유

ACLED는 다음 이유로 daily 운영 스케줄에 포함하지 않는다:

1. **운영 feature가 아님**: `modeling_acled_free_view`에 ACLED feature 없음
2. **미래 정보 위험**: y_escalation 라벨 생성에 사용 → 운영 시점 feature로 쓰면 data leakage
3. **주 1회 갱신**: 일별 수집 실익 없음
4. **API 접근 권한**: 학술용 계정 제한 가능성
5. **delayed-label 특성**: 7일 lag + 라벨 추가가 충분히 쌓인 후 재학습에 사용

→ ACLED는 별도 `delayed-label-collection` workflow (주 1회 또는 월 1회)로 관리 권장.
