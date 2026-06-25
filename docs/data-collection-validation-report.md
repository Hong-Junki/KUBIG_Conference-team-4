# 증분 수집 검증 보고서

---

## 검증 정보

| 항목 | 내용 |
|------|------|
| 검증일 | 2026-06-25 |
| commit SHA | (GitHub push 후 확인) |
| 실행 환경 | 로컬 (macOS) — GitHub Actions 실행은 미완료 |
| 검증 소스 | gdelt, economic |
| schedule 최종 상태 | **비활성** (workflow_dispatch만 활성) |

---

## 로컬 검증 결과

### 저장소 조사 (완료)

- 기존 수집 코드 전체 분석 완료
- BigQuery 테이블 현황 조회 완료 (읽기 전용)
- 치명적 버그 발견 및 수정:
  - `collect_gdelt_titles_gkg.py` 구 프로젝트 하드코딩 → env 변수화
  - `.gitignore` docs/ 제외 → 해제
  - `.gitignore` 새 SA JSON 패턴 추가

### unit test

```bash
python -m pytest tests/test_incremental.py -v
```

| 테스트 클래스 | 테스트 항목 | 상태 |
|-------------|-----------|------|
| TestComputeCollectionWindow | overlap 계산, forced_start, end<start 예외 등 7개 | 작성 완료 |
| TestGetMaxDateFromBq | 정상 반환, 예외 시 None, NULL 반환 | 작성 완료 |
| TestMergeQuery | insert-only / upsert SQL 구조 | 작성 완료 |
| TestDryRun | dry_run=True 시 쓰기 없음 | 작성 완료 |
| TestRunIncremental | 지원 안 되는 소스 거부, credential 없음 등 | 작성 완료 |
| TestEconomicDerivedColumns | pct_change 계산, 컬럼 누락 처리 | 작성 완료 |
| TestCountryMapping | 58개국, Palestine 다중 FIPS, 중복 없음 | 작성 완료 |
| TestGkgProjectId | 구 프로젝트명 하드코딩 없음 | 작성 완료 |
| TestGitignoreSecurity | .env 제외, SA JSON 제외, docs/ 포함 | 작성 완료 |

> **실제 test 실행 결과**: GCP credential 및 BigQuery client가 없는 로컬 환경에서는 mock 기반 테스트만 실행 가능. `google-cloud-bigquery` 라이브러리 없는 환경은 import 실패할 수 있음.

### dry-run

```bash
GOOGLE_APPLICATION_CREDENTIALS=<SA_JSON_경로> GCP_PROJECT_ID=conflict-ew-mvp-20260604 \
python -m src.collect.incremental.run_incremental --sources gdelt economic --dry-run
```

**실행 결과**:
- 구 SA(`conflict-early-warning` 프로젝트 발급)는 `conflict-ew-mvp-20260604`에 `bigquery.jobs.create` 권한 없음
- BQ MAX(date) 조회 실패 시 코드가 graceful fallback(overlap 기간만 사용)하는 것 확인
- GDELT 공개 테이블 dry-run 쿼리는 정상 실행: **0.26 GB / 4일분** 예상 스캔
- 경제지표 dry-run: API 기반, BQ 스캔 없음 (0 GB)
- dry-run 정상 종료 (exit code 0), JSON 로그 생성 확인

**권한 문제**: 기존 SA가 새 프로젝트에 접근 불가 → 새 SA 발급 또는 기존 SA에 권한 부여 필요.

### 실제 BigQuery 적재 테스트

**미실행**: 권한이 있는 SA 없이 실행 불가. 성공했다고 주장하지 않음.

---

## BQ 현황 조회 결과 (읽기 전용, 검증 완료)

| 테이블 | MAX(date) | 행 수 | 국가 수 | 비고 |
|--------|-----------|-------|---------|------|
| gdelt_processed_events | 2026-03-31 | 309,533,500 | 58 | 정상 |
| economic_daily | 2026-03-30 | 3,101 | — | 정상 |
| gdelt_titles | 2026-05-29 | 859,303,212 | 58 | 이미 최신 (별도 수집됨) |
| modeling_full_dataset | 2026-03-28 | 259,260 | 58 | snapshot, 자동 갱신 안 됨 |

---

## 생성/수정한 파일

### 생성 파일

| 파일 | 역할 |
|------|------|
| `src/collect/incremental/__init__.py` | 패키지 초기화 |
| `src/collect/incremental/state.py` | BQ 기반 상태 관리 |
| `src/collect/incremental/bigquery_io.py` | staging+MERGE 유틸 |
| `src/collect/incremental/collect_gdelt_incremental.py` | GDELT 증분 수집 |
| `src/collect/incremental/collect_economic_incremental.py` | 경제지표 증분 수집 |
| `src/collect/incremental/run_incremental.py` | CLI 오케스트레이터 |
| `tests/test_incremental.py` | unit test |
| `.github/workflows/incremental-data-collection.yml` | GitHub Actions workflow |
| `docs/data-pipeline-audit.md` | 현황 감사 문서 |
| `docs/incremental-data-collection.md` | 운영 가이드 |
| `docs/data-collection-validation-report.md` | 이 문서 |

### 수정 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/collect/collect_gdelt_titles_gkg.py` | TARGET_PROJECT 하드코딩 → GCP_PROJECT_ID env 변수화 |
| `.gitignore` | docs/ 제외 해제, 새 SA JSON 패턴 추가 |

---

## 보안 및 비용

### 필요한 GitHub Secrets

| Secret | 값 | 비고 |
|--------|-----|------|
| `GCP_PROJECT_ID` | `conflict-ew-mvp-20260604` | 공개 정보 |
| `GCP_SA_KEY` | 서비스 계정 JSON 전체 | 기존 SA key 또는 새로 생성 |
| `FRED_API_KEY` | FRED API key | 필수 (economic 소스) |

### Credential rotation 필요 여부

**강력히 권장**: 현재 `.env`에 `conflict-early-warning` 프로젝트의 서비스 계정 파일 경로가 기록되어 있다.  
해당 파일(`conflict-early-warning-4672e791d960.json`)은 gitignore되어 있으나, 파일 이름이 노출된 상태이다.  
GCP Console에서 해당 키의 최근 사용 이력을 확인하고, rotation을 수행할 것을 권장한다.

### 예상 BigQuery 비용 (증분, 1회 기준)

| 소스 | 예상 스캔 | 예상 비용 |
|------|----------|----------|
| GDELT 이벤트 (3일) | ~50 GB | ~$0.00 (무료 한도 내) |
| 경제지표 | API 기반 | $0.00 |
| 비용 안전장치 | `--max-gb-per-query 100` | 초과 시 자동 중단 |

---

## GitHub Actions 상태

**상태**: 코드 및 workflow 작성 완료. 실제 GitHub Actions 실행은 미검증.

실행 미완료 이유: GitHub Actions UI 직접 접근 불가 (로컬 Claude Code 환경).

---

## 사용자가 직접 해야 할 최소 작업

1. **GitHub Secrets 등록** (GitHub repo → Settings → Secrets):
   - `GCP_PROJECT_ID` = `conflict-ew-mvp-20260604`
   - `GCP_SA_KEY` = 서비스 계정 JSON 내용 전체
   - `FRED_API_KEY` = FRED API key

2. **Credential rotation** (권장):
   - GCP Console에서 `conflict-early-warning` 프로젝트의 기존 SA key 비활성화
   - `conflict-ew-mvp-20260604` 전용 SA key 생성

3. **GitHub Actions 수동 실행 검증**:
   - Actions 탭 → Incremental Data Collection → Run workflow
   - `dry_run: true`로 먼저 실행 (스캔량 확인)
   - 정상이면 `dry_run: false`로 실제 수집

4. **수집 후 BQ 검증**:
   ```sql
   SELECT MAX(event_date), COUNT(*), COUNT(DISTINCT iso3)
   FROM `conflict-ew-mvp-20260604.conflict_ew.gdelt_processed_events`
   WHERE event_date >= TIMESTAMP("2026-03-29");
   ```

5. **(향후, 선택)** cron 활성화:
   - `.github/workflows/incremental-data-collection.yml`에서 schedule 주석 해제
   - 커밋 후 push

---

## schedule 최종 상태

```yaml
# schedule 비활성 상태 (확인)
on:
  workflow_dispatch:   # 수동 실행만 허용
    ...
# schedule:
#   - cron: "20 1 * * *"
```

**반복 schedule: 비활성**
