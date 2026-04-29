# 팀원 공유 가이드 — 데이터·모델 결과 제출

> 무력충돌 예측 조기경보 시스템 (24-72h 위험도 0-100점)
> 본 문서: 공유받은 raw 데이터로 어떻게 모델을 만들고 어떤 형태로 결과를 제출하면 비교가 가능한지 정리

---

## 0. 한눈에 보기

```
공유 폴더(Drive) ─┬─ input/raw/         ← 가공 전 원본 (ACLED · GDELT · 경제)
                 ├─ input/processed/    ← 이미 만든 피처 + 라벨 + 데이터셋
                 └─ team-collaboration-guide.md  ← 본 문서

여러분이 만들어 보낼 것 ─┬─ predictions.csv    ← 최소 제출물 (필수)
                       └─ (선택) report.md   ← 모델 설명 + 결과 요약
```

**가장 빠른 시작**: `input/processed/dataset/full.parquet` 하나로 학습 가능. raw는 직접 피처를 새로 만들 때만 필요.

**본 문서로 충분**: 본 가이드 한 파일로 데이터 사용·모델 학습·결과 제출까지 끝낼 수 있도록 self-contained하게 작성됨.

---

## 1. 공유 데이터 명세

### 1-1. `input/raw/` (원본 데이터)

| 폴더 | 파일 형식 | 행수/크기 | 기간 | 비고 |
|------|----------|----------|------|------|
| `acled/{ISO3}.parquet` | 58개국 × 1파일 | ~492k 행 / 8.2 MB | 2022-01-01 ~ 2025-03-31 | 정치적 폭력 이벤트 |
| `gdelt/{ISO3}.parquet` | 58개국 × 1파일 | ~102M 행 / 1.1 GB | 2022-01-01 ~ 2025-03-31 | 글로벌 이벤트 (15분 갱신) |
| `economic/indicators.parquet` | 1파일 | 1070일 × 5지표 / 40 KB | 2022-01-03 ~ 2026-03-30 | VIX/WTI/Gold/DXY/STLFSI4 |

**ACLED 스키마** (이벤트 단위):
```
event_id_cnty, event_date(UTC), year, disorder_type, event_type, sub_event_type,
actor1, inter1, actor2, inter2, iso, country, admin1, latitude, longitude,
fatalities, timestamp
```

**GDELT 스키마** (이벤트 단위, BigQuery `events_partitioned` 추출):
```
GLOBALEVENTID, SQLDATE, ActionGeo_CountryCode(FIPS), EventCode, EventRootCode,
QuadClass, GoldsteinScale, NumMentions, NumArticles, AvgTone, event_date(UTC), iso3
```

**Economic 스키마** (일간 시계열):
```
index=date, columns=[VIX, WTI, Gold, DXY, STLFSI4]
```
- yfinance(VIX/WTI/Gold/DXY) → EST → UTC 변환됨
- FRED(STLFSI4) → 주간 갱신 (ffill 필요)

### 1-2. `input/processed/` (이미 가공된 데이터, 권장 시작점)

| 경로 | 내용 | 용도 |
|------|------|------|
| `dataset/full.parquet` | 68,614행 × 63컬럼 (54피처+3라벨+6메타) | **곧바로 학습** |
| `dataset/full_se.parquet` | full + Macis SE score(55번째 피처) | SE 피처 사용 시 |
| `dataset/train.parquet` | 42,340행 (~2023-12-31) | 학습 셋 |
| `dataset/val.parquet` | 10,556행 (2024-01 ~ 2024-06) | 검증 셋 |
| `dataset/test.parquet` | 15,718행 (2024-07 ~ 2025-03) | **테스트 셋 (학습/튜닝 금지)** |
| `features/baseline_scores.parquet` | 국가별 5y 사상자 prior B(0-100) | 위험도 점수 산출용 |
| `features/se_scores.parquet` | 66,890행 (Macis SE score) | SE 피처 단독 |
| `acled/`, `gdelt/`, `economic/`, `labels/` | 중간 산출물(원본 → 피처 변환 단계) | 피처를 새로 만들 때 참고 |

### 1-3. 메타·라벨 컬럼 (피처에서 반드시 제외)

```python
# 미래 정보 누수 방지 — 학습 시 X에서 반드시 빼야 함
LABEL_META_COLS = [
    "y", "y_onset", "y_escalation",
    "fatalities_next3d", "event_count_next3d",
    "past14d_event_count", "past14d_fatalities_mean",
]
```

### 1-4. 라벨 3종 — 어떤 것을 학습 타겟으로?

| 컬럼 | 양성 비율 | 정의 | 용도 |
|------|----------|------|------|
| `y` | 69.99% | 향후 3일 내 ACLED 이벤트 ≥1건 | persistence 비교용 (단독 학습 금지) |
| `y_onset` | 1.14% | 평시→분쟁 전환만 (희소) | onset 보조 평가 |
| **`y_escalation`** | **4.49%** | onset + 급격 악화 | **★ 주 학습 타겟** |

> `y` 단독 학습은 persistence baseline(0.984)에 지배되어 학술적 가치 없음. 반드시 `y_escalation` 사용.

---

## 2. 동일한 학습/검증/테스트 분할 사용 (필수)

비교가 가능하려면 **모두 같은 split**을 써야 합니다.

```
train: ~2023-12-31  (42,340행, 양성 4.75%)
val:   2024-01-01 ~ 2024-06-30  (10,556행, 양성 4.43%)
test:  2024-07-01 ~ 2025-03-31  (15,718행, 양성 4.06%)
```

**규칙**:
- ✋ test set은 학습/하이퍼파라미터 튜닝에 절대 사용 금지
- ✋ test set으로 모델 선택 금지 (val로만)
- ✋ 미래 정보 누수 금지 — `LABEL_META_COLS` 컬럼 모두 제외
- ✅ Optuna/grid search는 val PR-AUC 기준
- ✅ random seed는 42 권장 (결과 재현용)

---

## 3. 결과 제출 포맷

### 옵션 A: predictions CSV만 제출 (권장 — R/Julia/외부 환경 OK)

**파일명**: `predictions__{모델명}__{이름}.csv`
**컬럼**:
```csv
date,country,y_prob
2024-07-01,UKR,0.7234
2024-07-01,RUS,0.3120
...
```

**규칙**:
- `date`: ISO 형식 (UTC) — test set과 일치
- `country`: ISO3 코드 — test set과 일치
- `y_prob`: 0~1 사이 확률값 (캘리브레이션된 값 권장)
- 행 수: **15,718** (test set 전체) — 누락 행 있으면 평가 불가
- 인코딩: UTF-8

> 테스트 셋(`input/processed/dataset/test.parquet`)의 `date` + `country` 조합 그대로 따라 만드세요.

### 옵션 B: 코드 + 모델 가중치 함께 제출 (Python 환경)

본인 환경에서 학습한 코드 일체 + 모델 파일을 압축해서 보내기:

```
{모델명}__{이름}/
├── model.pkl (또는 .pt, .joblib 등)
├── train.py     ← 학습에 사용한 스크립트
├── predict.py   ← test.parquet → predictions.csv 생성 스크립트
├── requirements.txt
└── README.md    ← 실행 방법
```

`predict.py`는 다음과 같이 동작해야 함:
```bash
python predict.py --test input/processed/dataset/test.parquet --out predictions.csv
```

→ 결과 CSV 형식은 옵션 A와 동일.

### (선택) `report.md` — 모델 설명서

```markdown
# 모델: {이름}
## 접근법
- 어떤 알고리즘? 왜?
- 사용한 피처: 54개 기본 / +SE / 새로 만든 것
- 하이퍼파라미터 튜닝 방식
## 결과 (test set)
- PR-AUC, Precision@top5%, Recall@top5%, persistence_gain
## 한계 / 개선 아이디어
```

---

## 4. 평가 기준 (6개 직교 지표군)

자체 평가 + 우리 쪽에서 동일 스크립트로 재평가. **굵은 지표가 합격선**:

| 지표군 | 항목 | 합격선 |
|-------|------|--------|
| A. 순위 | PR-AUC, Macro PR-AUC, Onset PR-AUC | 높을수록 ↑ |
| B. Triviality | **persistence_gain** = PR-AUC − persistence_baseline | **> 0** (필수) |
| C. Alert budget | P/R@top1%·top5%·top10% | P@top5% ≥ 0.10 권장 |
| D. Calibration | ECE (Expected Calibration Error, 10 bins) | < 0.05 권장 |
| E. Lead Time | val 상위 5% 분위수 τ로 산출, 이벤트 전 알림 도달일 평균 | 1일 이상 |
| F. Precision-target | R@P≥0.10·0.20·0.30 (precision 임계값에서의 recall) | R@P≥0.10 ≥ 0.10 권장 |

**용어 설명**:
- **persistence_gain**: 어제와 오늘의 라벨이 같다고 가정하는 단순 baseline(`y_t = y_{t-1}`) 대비 PR-AUC가 얼마나 더 높은지. 양수가 아니면 모델이 trivial baseline보다 못한다는 뜻
- **P@top-k%**: 점수 상위 k%를 알림으로 골랐을 때의 precision (k=5면 상위 786건이 진짜 양성일 비율)
- **R@P≥t**: precision이 t 이상으로 유지되는 임계값에서의 recall (운영 지표)

**현재 베이스라인** (참고용, test set 기준 `y_escalation`):

| 모델 | PR-AUC | persistence_gain | P@top5% | R@P≥0.10 |
|------|--------|-----------------|---------|----------|
| Persistence (지속 가정) | 0.0354 | 0 (기준) | — | — |
| Logistic Regression | 0.0534 | +0.0180 | 0.062 | N/A |
| LightGBM | 0.0779 | +0.0424 | 0.120 | 0.194 |
| **LightGBM + SE (현 1위)** | **0.1307** | **+0.0952** | **0.190** | **0.558** |
| XGBoost | 0.0764 | +0.0410 | 0.125 | 0.204 |
| LSTM | 0.0414 | +0.0060 | 0.033 | N/A |
| Stacking Ensemble | 0.1263 | +0.0909 | 0.171 | N/A |

**현실적 기대치**:
- 양성 4.49% 희소 타겟이라 PR-AUC가 높지 않음. 0.10 넘으면 우수.
- 우리 1위 모델은 random 대비 **2.9배** 더 정확.

---

## 5. 진행 방식

1. **데이터 받기**: 공유 Drive에서 `input/raw/` 또는 `input/processed/` 다운로드
2. **모델 만들기**: 옵션 A(CSV) 또는 옵션 B(코드+가중치) — 접근법은 자유
3. **결과 자체 검증**:
   - 행 수 = 15,718
   - 컬럼 = `date, country, y_prob`
   - `y_prob` 범위 0 ~ 1
   - `(date, country)` 조합이 `test.parquet`과 정확히 일치
4. **공유**: 같은 Drive 폴더에 본인 이름 폴더 만들어 업로드
5. **다음 주 목요일(2026-05-07) 미팅**: 각자 결과 가지고 와서 함께 논의

---

## 6. 추가 설명

**피처를 새로 만들고 싶을 때**
`input/raw/` 사용 권장. 미래 정보 누수 주의:
- ACLED는 주간 갱신 → 피처로 쓸 때 7일 lag 적용 필수
- GDELT는 당일 데이터를 다음날 사용 (shift 1)
- 모든 timestamp는 UTC

**다른 데이터 추가**
가능. 외부 소스 사용 시 `report.md`에 출처·스키마·라이선스·기간만 적어주면 됨.

**Python 외 환경**
옵션 A(CSV 제출)면 R/Julia/Spark/직접 만든 LLM 앙상블 모두 가능.

**test set 사용 규칙**
자기 결과 확인용 OK. 단, **test set으로 모델을 고르는 행위 금지** — val로만 선택.

**데이터 기간(3년) 한계**
ACLED Research 라이선스 12개월 제한 때문. Academic 라이선스 있으면 과거 확장 가능.

**위험도 0-100점 변환**
`y_prob`만 있으면 다음 식으로 변환됨 — 직접 100점 점수를 만들 필요 없음.
`risk_score = 0.2·B + 0.4·C_state + 0.4·F + hotspot`
- B: ACLED 5년 사상자 prior (`features/baseline_scores.parquet`)
- C_state: U(시민피해) / C(분쟁) / S(사회) / I(정보) 4개 서브스코어 평균 (90일 rolling z-score → 0-100)
- F: ML 예측값 × spread_factor (val 99분위수 → 100 매핑)
- hotspot: 인접국 C≥50 카운트 × 1.5 (최대 +5)

**실행 환경**
옵션 B 제출 시 `requirements.txt` 또는 `environment.yml` 동봉. 참고 환경: Python 3.10, macOS / Colab T4.
