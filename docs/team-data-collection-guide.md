# 팀원 공유 가이드 — 과거 데이터 추가 수집

> 무력충돌 예측 조기경보 시스템 — 데이터 기간 확장 (2014 ~ 2021)
> 4명 분담 수집 후 통합. 본 가이드 한 파일로 환경 세팅·수집·검증·회신까지 끝낼 수 있도록 self-contained하게 작성됨.

---

## 0. 작업 개요

현재 프로젝트는 2022-01 ~ 2025-03 (약 3년) 기간만 수집되어 있어 모델 학습에 데이터가 부족합니다. 분쟁 사이클 1회 이상을 포함하려면 8년치가 필요하고, 4명이 2년씩 분담합니다.

| 담당자 | 수집 기간 | 비고 |
|-------|----------|------|
| **팀원 A** | 2014-01-01 ~ 2015-12-31 | GDELT 2.0 호환 시작점(2015 안전) |
| **팀원 B** | 2016-01-01 ~ 2017-12-31 | |
| **팀원 C** | 2018-01-01 ~ 2019-12-31 | |
| **팀원 D** | 2020-01-01 ~ 2021-12-31 | 코로나 + Karabakh + Myanmar |

수집 대상: ACLED · GDELT · 경제지표 (총 3종)

---

## 1. 받을 파일

운영자가 zip으로 묶어 공유합니다 (Drive 또는 직접 전달):

```
collect-package/
├── src/
│   ├── __init__.py
│   └── collect/
│       ├── __init__.py
│       ├── config.py            ← 58개국 + API 설정
│       ├── utils.py             ← 재시도 / 로거 / 체크포인트
│       ├── acled_collector.py
│       ├── gdelt_collector.py
│       ├── economic_collector.py
│       └── run_historical.py    ← 메인 실행 스크립트
├── requirements.txt
├── .env.example                 ← 키 입력 위치 안내
└── team-data-collection-guide.md  ← 본 문서
```

**별도 DM으로 받는 것**: `.env` (실제 API 키 입력된 파일) — `.env.example` 자리에 그대로 두면 됨

---

## 2. 사전 준비

### 2-1. Python 3.10 환경 + 가상환경

```bash
# macOS / Linux
python3.10 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2-2. 패키지 설치

```bash
pip install -r requirements.txt
```

(설치 시간 약 3~5분. 주요 패키지: `acled`, `google-cloud-bigquery`, `yfinance`, `fredapi`, `pandas`, `pyarrow`)

### 2-3. `.env` 배치

DM으로 받은 `.env` 파일을 프로젝트 루트(= `requirements.txt`와 같은 위치)에 두세요. `.env.example`은 그대로 두거나 삭제 OK.

```bash
# 확인
cat .env  # macOS / Linux
type .env # Windows
```

`ACLED_USERNAME` / `ACLED_PASSWORD` / `GOOGLE_APPLICATION_CREDENTIALS` / `FRED_API_KEY` 4개 값이 채워져 있어야 합니다.

`GOOGLE_APPLICATION_CREDENTIALS`는 JSON 파일 경로입니다. JSON 파일도 같이 DM으로 받았으면 그 경로를 절대경로로 적어주세요.

### 2-4. dry-run으로 사전 점검 (선택, 권장)

GDELT BigQuery 비용 사전 확인:
```bash
python -m src.collect.run_historical \
  --sources gdelt --dry-run \
  --start 2014-01-01 --end 2015-12-31
```
출력 예시: `GDELT 전체 예상 스캔량: 12.34 GB / 예상 비용: $0.00`

(BigQuery 무료 한도가 월 1TB라 4명 합쳐도 무료 한도 내. 비용 발생하면 운영자에 즉시 알림)

---

## 3. 데이터 수집 실행

### 3-1. 본인 담당 기간으로 명령 실행

**팀원 A (2014-2015)**:
```bash
python -m src.collect.run_historical \
  --start 2014-01-01 --end 2015-12-31
```

**팀원 B (2016-2017)**:
```bash
python -m src.collect.run_historical \
  --start 2016-01-01 --end 2017-12-31
```

**팀원 C (2018-2019)**:
```bash
python -m src.collect.run_historical \
  --start 2018-01-01 --end 2019-12-31
```

**팀원 D (2020-2021)**:
```bash
python -m src.collect.run_historical \
  --start 2020-01-01 --end 2021-12-31
```

→ **연도 숫자만 본인 담당으로 바꾸면 됩니다.**

### 3-2. 예상 소요 시간

| 소스 | 시간 (2년치 기준) |
|------|------------------|
| ACLED | 30분 ~ 1시간 (페이지네이션 + OAuth) |
| GDELT BigQuery | 10 ~ 20분 |
| 경제지표 | 1 ~ 2분 |
| **총합** | **약 1시간** |

**중간에 끊겨도 OK** — 모든 수집기는 체크포인트 기반 idempotent. 같은 명령 다시 실행하면 이어서 받습니다.

**macOS는 절전 방지 권장**:
```bash
caffeinate -i python -m src.collect.run_historical --start 2014-01-01 --end 2015-12-31
```

### 3-3. 일부 소스만 수집할 때

```bash
# 경제지표만
python -m src.collect.run_historical --sources economic --start ... --end ...

# ACLED + GDELT만
python -m src.collect.run_historical --sources acled gdelt --start ... --end ...
```

---

## 4. 결과 검증

수집 후 자동으로 검증이 실행됩니다. 추가로 수동 검증:

```bash
python -m src.collect.run_historical --validate-only
```

**기대 출력**:
```
ACLED:  58/58개국 파일 존재
GDELT BQ: 58/58개국 파일 존재
경제지표: NNN일 × 5개 컬럼
검증 완료 — 이슈 없음
```

**수집 결과물 위치**:
```
input/raw/
├── acled/
│   ├── AFG.parquet
│   ├── ARM.parquet
│   └── ... (58개)
├── gdelt/
│   ├── AFG.parquet
│   ├── ARM.parquet
│   └── ... (58개)
└── economic/
    └── indicators.parquet
```

---

## 5. 결과 회신 방법

### 5-1. 본인 결과만 압축

```bash
# 프로젝트 루트에서 (macOS / Linux)
zip -r raw_{본인이름}_{기간}.zip input/raw/

# 예시
zip -r raw_홍길동_2014-2015.zip input/raw/
```

Windows는 `input/raw/` 폴더를 우클릭 → 압축으로 만들기 → 파일명 변경.

### 5-2. 업로드

공유 Drive에 업로드 (또는 운영자에게 직접 전달).

파일명 예시:
- `raw_A_2014-2015.zip`
- `raw_B_2016-2017.zip`
- `raw_C_2018-2019.zip`
- `raw_D_2020-2021.zip`

### 5-3. 다음 미팅(2026-05-07) 전까지 회신

운영자가 4개 zip을 ISO3별로 concat → 통합 데이터셋 구축.

---

## 6. 주의사항 / 트러블슈팅

### ACLED 관련
- **Research 라이선스는 최근 12개월 제한이 있을 수 있음** — `flood_user_blocked` 또는 비어있는 응답이 오면 즉시 운영자에 알림. Academic 라이선스로 전환 필요할 수 있음
- OAuth 토큰은 24시간 유효 — 1일 이상 끊겼다가 재시작하면 자동 갱신됨
- ACLED `country` 이름이 ISO3과 다른 경우 있음 (예: PSE → "Palestine", CIV → "Ivory Coast") — `config.py`의 `acled_names` 필드에 매핑 처리 완료, 신경 쓸 필요 없음

### GDELT BigQuery 관련
- 4명이 같은 GCP 서비스 계정을 공유하므로 **각자 동시 실행 시 비용이 합산**됨. 가능하면 시간 분산해서 실행
- 비용은 무료 한도(월 1TB) 안에 들어오지만 모니터링 필요. `--dry-run`으로 사전 확인 권장
- `events_partitioned` 테이블 사용 (이미 `config.py`에 설정됨) — 일반 `events` 대비 28배 절감

### 경제지표 관련
- yfinance 거래일 기준이라 주말/공휴일 제외됨. 정상
- FRED `STLFSI4`는 주간 갱신 — 일간 ffill 필요 (전처리 단계에서 자동 처리)
- macOS Python SSL 오류 발생 시 `economic_collector.py`에 패치 이미 적용됨

### 일반
- **타임존은 모두 UTC로 통일** — 코드에서 자동 처리, 신경 X
- 디스크 여유 공간: ACLED ~50MB / GDELT ~700MB (2년치 기준) / 경제지표 ~10KB → **총 1GB 이상 비워두기**
- Python 3.11 / 3.12에서도 동작은 하지만 **3.10 권장** (일부 패키지 호환성)

### 막혔을 때
1. 에러 메시지 전체 복사
2. 어느 명령에서 발생했는지
3. 운영체제 / Python 버전 (`python --version`)
→ DM으로 운영자에게 공유

---

## 7. 체크리스트

작업 시작 전:
- [ ] zip 압축 풀기 + 가상환경 생성
- [ ] `pip install -r requirements.txt` 완료
- [ ] DM으로 받은 `.env` 프로젝트 루트에 배치
- [ ] `python -m src.collect.run_historical --validate-only` 실행 → 에러 없음 확인 (파일이 없다는 경고는 정상)

수집 중:
- [ ] 본인 담당 기간으로 명령 실행
- [ ] (선택) `--dry-run`으로 BigQuery 비용 사전 확인
- [ ] `caffeinate -i ...` (macOS) 또는 절전 모드 OFF

수집 후:
- [ ] `--validate-only`로 58/58개국 확인
- [ ] zip 압축
- [ ] Drive 업로드 + 운영자에게 알림

---

끝. 감사합니다.
