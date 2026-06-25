# PRD: 무력충돌 조기경보 모델링 및 대시보드 MVP

## 0. 프로젝트 개요

### 제품명

Conflict Early Warning Dashboard  
한국어명: 무력충돌 조기경보 대시보드 / 세계 충돌 온도 지도

### 한 줄 설명

ACLED, GDELT, 경제지표 기반으로 국가별 무력충돌 escalation 위험을 예측하고, 예측 결과를 지도·카드·시계열 그래프·브리핑으로 보여주는 웹 대시보드.

### 이번 개발 목표

이번 프로젝트의 1차 목표는 두 가지다.

1. `y_escalation`을 예측하는 LightGBM baseline 모델을 만들고, test set 전체에 대한 예측 파일을 생성한다.
2. 생성된 예측 결과 CSV를 웹 대시보드에서 시각화할 수 있는 기초 틀을 만든다.

이번 MVP에서는 모델링을 우선하고, 웹앱은 모델 결과 CSV를 읽어 시각화하는 구조로 설계한다.

---

## 1. 배경

지정학적 충돌과 무력 분쟁은 더 이상 일부 지역만의 문제가 아니라, 글로벌 공급망, 에너지 가격, 환율, 투자심리, 기업 운영 리스크에 직접적인 영향을 미치는 핵심 변수다.

기존 방식은 주로 뉴스 모니터링, 정성적 리포트, 사후 분석에 의존한다. 하지만 사용자는 “어느 국가의 위험 신호가 높아지고 있는지”, “왜 위험한지”, “얼마나 일찍 감지할 수 있는지”를 빠르게 파악하기 어렵다.

본 프로젝트는 ACLED의 정치적 폭력 이벤트, GDELT의 글로벌 뉴스 이벤트, 경제지표를 활용하여 국가별 무력충돌 escalation 위험을 예측하고 이를 시각적으로 설명하는 조기경보 시스템을 만드는 것을 목표로 한다.

---

## 2. 문제 정의

### 2.1 기존 위험도 파악 방식의 한계

현재 무력충돌 리스크를 확인하려면 다음과 같은 문제가 있다.

- ACLED, GDELT, 경제지표 등 데이터가 분산되어 있다.
- 뉴스량이 많다고 반드시 실제 위험이 높은 것은 아니다.
- 반대로 뉴스량이 적은 국가는 실제 위험이 있어도 낮게 보일 수 있다.
- 단순 사건 발생 여부인 `y`는 양성 비율이 너무 높아 persistence baseline에 지배될 수 있다.
- 사용자는 단순 확률값보다 “왜 위험한지”와 “최근 어떻게 변했는지”를 함께 알고 싶어 한다.

따라서 본 프로젝트는 단순히 “향후 3일 내 사건이 하나라도 발생했는가”를 예측하는 것이 아니라, **onset 또는 급격 악화에 해당하는 escalation 위험**을 예측하는 방향으로 설계한다.

### 2.2 모델링 관점의 핵심 문제

본 프로젝트는 일반적인 이진분류 문제가 아니라 희귀 이벤트 조기경보 문제다.

따라서 단순 accuracy보다 다음 지표가 중요하다.

- PR-AUC
- persistence_gain
- P@top5%
- Recall@Precision threshold
- ECE
- Lead Time

### 2.3 대시보드 관점의 핵심 문제

모델이 `0.13` 또는 `0.82` 같은 확률값을 출력하더라도, 사용자는 그 수치만 보고 의미를 이해하기 어렵다.

따라서 대시보드는 다음 질문에 답해야 한다.

1. 지금 어떤 국가가 위험한가?
2. 얼마나 위험한가?
3. 왜 위험한가?
4. 최근 위험도가 어떻게 변했는가?
5. 모델 결과가 단순 baseline보다 의미 있는가?

---

## 3. 핵심 사용자

### 3.1 1차 사용자

- 프로젝트 팀원
- 발표 청중
- 모델 결과를 비교해야 하는 KUBIG 스터디 구성원

### 3.2 2차 사용자

- 글로벌 정세 변화가 공급망, 투자, 유가, 환율, 해외 운영 리스크에 미치는 영향을 빠르게 파악하고 싶은 사용자
- 국제 뉴스의 맥락을 빠르게 파악하고 싶은 일반 사용자
- 특정 국가의 위험도 변화와 그 원인을 보고 싶은 사용자

### 3.3 사용자 니즈

사용자는 다음 질문에 답하고 싶어 한다.

1. 어떤 국가의 escalation 위험이 높은가?
2. 모델이 어느 국가를 위험하다고 판단했는가?
3. 특정 국가의 위험 점수는 시간에 따라 어떻게 변했는가?
4. 모델이 단순 persistence baseline보다 나은가?
5. 위험도가 높은 이유를 feature와 브리핑으로 설명할 수 있는가?
6. 팀원별 모델 결과를 같은 형식으로 비교할 수 있는가?

---

## 4. 제품 목표

이번 MVP의 목표는 완전한 상용 서비스가 아니라, 팀 프로젝트 발표 및 모델 비교를 위한 **모델링 baseline + 시각화 가능한 대시보드 기초 틀**을 만드는 것이다.

### 4.1 반드시 달성할 목표

1. `train.parquet`, `val.parquet`, `test.parquet`를 읽어 모델링할 수 있다.
2. `y_escalation`을 타겟으로 LightGBM baseline을 학습한다.
3. validation set 기준으로 PR-AUC 등 기본 지표를 출력한다.
4. test set 전체에 대해 `y_prob`를 생성한다.
5. 최종 제출 형식인 `predictions__lightgbm__byeonghyeon.csv`를 생성한다.
6. 생성된 prediction 파일을 대시보드에서 읽어 국가별 위험도를 표시할 수 있는 구조를 만든다.
7. 국가별 위험도, 시계열, 상세 패널, 브리핑으로 이어지는 대시보드 구조를 설계한다.

### 4.2 이번 버전에서 무리하지 않을 것

1. 실시간 API 연동
2. 사용자 로그인
3. Supabase 데이터베이스 연결
4. 고급 지도 인터랙션 완성
5. raw 데이터 기반 피처 재생성
6. Telegram/Reddit 소셜 데이터 통합
7. 고급 딥러닝 모델 구현
8. test set 기반 모델 선택
9. 완전한 예측 확률 calibration
10. 운영 수준의 실시간 알림 시스템

---

## 5. 데이터 구조

### 5.1 프로젝트 폴더 구조

현재 프로젝트 루트는 `conflict-warning-dashboard`로 둔다.

권장 구조는 다음과 같다.

- `PRD.md`
- `team-collaboration-guide.md`
- `input/processed/dataset/train.parquet`
- `input/processed/dataset/val.parquet`
- `input/processed/dataset/test.parquet`
- `input/processed/dataset/full.parquet`
- `input/processed/dataset/full_se.parquet`
- `input/processed/features/baseline_scores.parquet`
- `input/processed/features/features.parquet`
- `input/processed/features/se_scores.parquet`
- `modeling/`
- `outputs/`
- `dashboard/`

### 5.2 우선 사용할 데이터

이번 프로젝트에서는 raw 데이터가 아니라 이미 가공된 processed dataset을 우선 사용한다.

필수 파일:

| 파일 | 용도 |
|---|---|
| `input/processed/dataset/train.parquet` | 모델 학습 |
| `input/processed/dataset/val.parquet` | 모델 검증 및 튜닝 |
| `input/processed/dataset/test.parquet` | 최종 예측 생성 |

추가 실험용 파일:

| 파일 | 용도 |
|---|---|
| `input/processed/dataset/full.parquet` | 전체 데이터 확인 |
| `input/processed/dataset/full_se.parquet` | SE score 포함 실험 |
| `input/processed/features/baseline_scores.parquet` | 국가별 baseline prior 및 risk score 확장 |
| `input/processed/features/se_scores.parquet` | SE score 단독 확인 |
| `input/processed/features/features.parquet` | 중간 feature 산출물 확인 |

### 5.3 split 구조

모든 팀원이 동일한 split을 사용한다.

| split | 용도 | 주의사항 |
|---|---|---|
| train | 모델 학습 | 학습에 사용 |
| val | 모델 선택 및 하이퍼파라미터 튜닝 | PR-AUC 기준 튜닝 |
| test | 최종 예측 제출 | 학습/튜닝 금지 |

test set은 최종 예측 파일 생성에만 사용한다.

### 5.4 데이터 사용 원칙

1. test set은 학습 또는 하이퍼파라미터 튜닝에 절대 사용하지 않는다.
2. test set으로 모델을 선택하지 않는다.
3. validation set 성능을 기준으로 모델을 선택한다.
4. 미래 정보가 포함된 컬럼은 feature에서 제외한다.
5. 모든 팀원은 동일한 train/val/test split을 사용한다.

---

## 6. 학습 타겟

### 6.1 주 타겟

주 학습 타겟은 `y_escalation`이다.

의미:

- onset + 급격 악화에 해당하는 무력충돌 escalation 발생 여부
- 희귀 이벤트 조기경보에 더 적합한 라벨
- 본 프로젝트의 주 모델링 대상

### 6.2 사용하지 말아야 할 타겟

`y`는 주 타겟으로 사용하지 않는다.

이유:

- 향후 3일 내 ACLED 이벤트 1건 이상 발생 여부
- 양성 비율이 너무 높음
- persistence baseline에 지배될 가능성이 큼
- 모델이 “이미 분쟁이 지속되는 국가”만 학습할 수 있음
- 학술적/모델링 가치가 낮음

`y_onset`은 보조 평가용으로만 사용한다.

---

## 7. feature 제외 컬럼

모델 학습 시 아래 컬럼은 feature에서 반드시 제외한다.

- `y`
- `y_onset`
- `y_escalation`
- `fatalities_next3d`
- `event_count_next3d`
- `past14d_event_count`
- `past14d_fatalities_mean`

추가로 `date`는 직접 feature로 사용하지 않는다.

`country`는 categorical feature로 사용할 수 있다.

### 7.1 기본 제외 목록

모델 입력에서 제외할 기본 컬럼은 다음과 같다.

| 컬럼 | 제외 이유 |
|---|---|
| `date` | 시간 index로만 사용 |
| `y` | 라벨 |
| `y_onset` | 보조 라벨 |
| `y_escalation` | 주 타겟 |
| `fatalities_next3d` | 미래 정보 |
| `event_count_next3d` | 미래 정보 |
| `past14d_event_count` | 라벨/메타 보조 컬럼 |
| `past14d_fatalities_mean` | 라벨/메타 보조 컬럼 |

### 7.2 country 처리

`country`는 다음 방식 중 하나로 처리한다.

1. LightGBM categorical feature로 사용
2. label encoding 후 사용
3. 비교 실험에서는 country를 제외하여 국가 memorization 영향을 확인

1차 baseline에서는 `country`를 categorical feature로 사용하는 것을 기본으로 한다.

---

## 8. 모델링 목표

### 8.1 1차 baseline

LightGBM binary classifier를 사용한다.

이유:

- tabular 데이터에 강함
- 희귀 이벤트 분류에 적용하기 쉬움
- feature importance 및 SHAP 해석 가능
- 빠르게 baseline을 만들 수 있음
- 현재 팀 baseline에서도 성능이 좋은 편
- LSTM 등 딥러닝 모델보다 우선적으로 안정적인 기준선을 만들 수 있음

### 8.2 학습 방식

- train set으로 학습한다.
- val set으로 평가 및 모델 선택을 수행한다.
- test set은 최종 예측에만 사용한다.
- 클래스 불균형 대응을 위해 `scale_pos_weight` 또는 class weight를 사용한다.
- random seed는 42를 사용한다.
- validation PR-AUC를 우선 지표로 사용한다.

### 8.3 1차 평가 지표

우선 다음 지표를 계산한다.

- PR-AUC
- P@top5%
- Recall@Precision≥0.10
- ECE

추후 다음 지표를 추가한다.

- Macro PR-AUC
- Onset PR-AUC
- Lead Time
- persistence_gain
- Recall@Precision≥0.20
- Recall@Precision≥0.30

### 8.4 클래스 불균형 대응

`y_escalation`은 희귀 이벤트이므로 클래스 불균형 대응이 필요하다.

사용 가능한 방식:

- `scale_pos_weight`
- class weight
- threshold tuning
- top-k alert 관점 평가
- calibration 보정

1차 baseline에서는 `scale_pos_weight = negative_count / positive_count` 방식으로 시작한다.

---

## 9. 최종 제출 파일

### 9.1 파일명

`predictions__{모델명}__{이름}.csv`

예시:

`predictions__lightgbm__byeonghyeon.csv`

### 9.2 컬럼

| 컬럼 | 설명 |
|---|---|
| date | test set의 날짜 |
| country | ISO3 국가 코드 |
| y_prob | `y_escalation=1`일 예측 확률 |

### 9.3 제출 규칙

- test set 전체 행을 빠짐없이 포함해야 한다.
- 행 수는 test set과 동일해야 한다.
- `date, country` 조합은 test set과 정확히 일치해야 한다.
- `y_prob`는 0~1 사이 실수여야 한다.
- 인코딩은 UTF-8을 사용한다.
- 컬럼명은 반드시 `date,country,y_prob`로 맞춘다.

### 9.4 제출 전 검증

제출 전 다음을 확인한다.

1. 행 수가 test set과 같은가?
2. `date,country` 조합이 test set과 정확히 같은가?
3. `y_prob`에 결측치가 없는가?
4. `y_prob`가 모두 0~1 범위 안에 있는가?
5. 컬럼 순서가 `date,country,y_prob`인가?

---

## 10. 위험도 점수 변환

모델 출력인 `y_prob`는 웹앱에서 위험도 점수로 변환할 수 있다.

MVP 기준:

- `risk_score = y_prob × 100`

표시용 등급:

| y_prob 구간 | Risk Level | 한국어 표시 |
|---|---|---|
| 0.00 이상 0.30 미만 | Low | 낮음 |
| 0.30 이상 0.60 미만 | Moderate | 주의 |
| 0.60 이상 0.80 미만 | High | 높음 |
| 0.80 이상 1.00 이하 | Critical | 매우 높음 |

주의:

- `y_prob`는 완전히 calibration된 실제 확률이라고 단정하지 않는다.
- UI에서는 “모델 기반 위험 점수” 또는 “escalation 위험 신호”로 표현한다.
- 최종 서비스에서는 `baseline_scores.parquet`의 국가 prior, ML 예측값, hotspot 보정 등을 결합한 0~100점 risk score로 확장할 수 있다.

---

## 11. 모델링 코드 구조

권장 파일 구조:

- `modeling/train_lightgbm.py`
- `modeling/predict_lightgbm.py`
- `modeling/evaluate.py`
- `modeling/utils.py`
- `modeling/requirements.txt`
- `modeling/README.md`

출력 폴더:

- `outputs/models/`
- `outputs/predictions/`
- `outputs/reports/`

### 11.1 utils.py

역할:

- parquet 파일 로드
- 제외 컬럼 정의
- feature/target 분리
- country categorical 처리
- 제출 CSV 검증
- 출력 폴더 생성

필수 함수:

- `load_datasets()`
- `get_feature_columns()`
- `make_xy()`
- `validate_prediction_file()`
- `ensure_output_dirs()`

### 11.2 train_lightgbm.py

역할:

- train/val 데이터 로드
- `y_escalation` 타겟 설정
- 제외 컬럼 제거
- `country` categorical 처리
- LightGBM 학습
- val PR-AUC 평가
- 추가 지표 출력
- 모델 저장

출력:

- `outputs/models/lightgbm_baseline.pkl`
- `outputs/reports/lightgbm_val_metrics.json`

### 11.3 predict_lightgbm.py

역할:

- 저장된 모델 로드
- test 데이터 로드
- test 예측 수행
- `date,country,y_prob` 형식 CSV 저장
- 행 수, 확률 범위, date-country 조합 검증

출력:

- `outputs/predictions/predictions__lightgbm__byeonghyeon.csv`

### 11.4 evaluate.py

역할:

- PR-AUC 계산
- P@top5% 계산
- Recall@Precision threshold 계산
- ECE 계산

### 11.5 requirements.txt

포함 후보:

- pandas
- pyarrow
- numpy
- scikit-learn
- lightgbm
- joblib
- matplotlib

---

## 12. 개발 우선순위

## Phase 1. 데이터 구조 확인

할 일:

1. `train.parquet`, `val.parquet`, `test.parquet` 로드
2. shape 확인
3. 컬럼 목록 확인
4. `y_escalation` 양성 비율 확인
5. 결측치 확인
6. `date`, `country` 분포 확인
7. feature 후보 컬럼 확인
8. 제외 컬럼이 제대로 제외되는지 확인

완료 기준:

- 데이터가 LightGBM 학습 가능한 상태인지 확인된다.

## Phase 2. LightGBM baseline

할 일:

1. feature/target 분리
2. `country` categorical 처리
3. LightGBM baseline 학습
4. val PR-AUC 출력
5. P@top5%, ECE 등 기본 지표 출력
6. 모델 저장

완료 기준:

- validation set에 대한 baseline 성능이 출력된다.
- 모델 파일이 저장된다.

## Phase 3. test prediction 생성

할 일:

1. 저장된 LightGBM 모델 로드
2. test set 예측
3. `date,country,y_prob` 형식으로 저장
4. 제출 CSV 검증

완료 기준:

- `outputs/predictions/predictions__lightgbm__byeonghyeon.csv` 생성
- test set 전체 행 포함
- y_prob 범위 0~1

## Phase 4. LightGBM + SE 실험

할 일:

1. `full_se.parquet` 또는 `se_scores.parquet`의 SE score를 train/val/test에 병합
2. `macis_se_score` 포함 모델 학습
3. val 기준 성능 비교
4. 성능이 좋으면 최종 제출 후보로 사용

완료 기준:

- 기본 LightGBM과 SE 포함 LightGBM의 성능 비교가 가능하다.

## Phase 5. 웹 대시보드 MVP

할 일:

1. Next.js + TypeScript + Tailwind CSS 프로젝트 생성
2. predictions CSV 로드
3. 위험도 상위 국가 카드 표시
4. 국가별 테이블 표시
5. 국가 클릭 시 시계열 그래프 표시
6. 모델 선택 구조 추가

완료 기준:

- 생성된 predictions CSV를 웹에서 시각화할 수 있다.

## Phase 6. 발표용 고도화

할 일:

1. 브리핑 카드 추가
2. 모델 성능 지표 표시
3. feature importance 또는 SHAP 시각화
4. 지도 시각화 추가
5. 발표용 디자인 개선

---

## 13. 웹 대시보드 목표

모델링 결과가 생성된 이후 웹앱은 `predictions__*.csv`를 읽어 시각화한다.

### 13.1 메인 대시보드

화면 요소:

1. 상단 헤더
   - 서비스명: Conflict Temperature Map
   - 부제: ACLED + GDELT + Economic Indicators 기반 무력충돌 조기경보 대시보드
   - 기준 날짜 선택
   - 모델 선택

2. 요약 KPI 카드
   - 평균 위험 점수
   - 위험도 상위 국가 수
   - Critical 국가 수
   - 전일 대비 가장 크게 상승한 국가
   - 사용 중인 모델명

3. 위험도 상위 국가 카드
   - TOP 5 국가 표시
   - 국가명
   - `y_prob`
   - `risk_score`
   - `risk_level`
   - 전일 대비 변화량
   - 짧은 설명 문장

4. 지도 또는 지도 대체 UI
   - 가능하면 세계 지도 choropleth 구현
   - 구현이 오래 걸리면 국가별 카드/테이블/랭킹 UI로 대체
   - 추후 지도 컴포넌트를 붙일 수 있도록 구조 분리

### 13.2 국가 클릭 상세 패널

국가를 클릭하면 다음 정보를 보여준다.

1. 국가명
2. 현재 `y_prob`
3. `risk_score`
4. `risk_level`
5. 최근 7일 또는 30일 `y_prob` 시계열 그래프
6. 위험도 상승 이유
7. 주요 feature 요약
8. 간단한 브리핑

### 13.3 해석 가능성 및 관련 근거 제공

본 대시보드는 단순히 모델 예측 확률만 표시하지 않는다. 사용자가 위험도 점수를 이해할 수 있도록 모델이 위험하다고 판단한 근거와 관련 데이터 신호를 함께 제공한다.

#### 13.3.1 Feature 기반 설명

1차 MVP에서는 SHAP이나 외부 뉴스 API를 바로 붙이기보다, 이미 processed dataset에 포함된 feature를 활용해 위험도 상승 이유를 설명한다.

예를 들어 특정 국가의 `y_prob` 또는 `risk_score`가 높을 경우 다음과 같은 feature 신호를 확인한다.

- ACLED event count 증가
- ACLED fatalities 증가
- GDELT tone 악화
- GDELT mentions 증가
- GDELT event count 증가
- GDELT QuadClass 4 또는 conflict-related signal 증가
- 경제지표 변동성 증가

대시보드에서는 이를 사용자가 이해하기 쉬운 문장으로 변환한다.

예시 문장:

- 최근 7일간 ACLED 이벤트 수가 증가하면서 실제 충돌 관련 신호가 강화되었습니다.
- GDELT tone이 악화되고 mentions가 증가하면서 국제 뉴스상 긴장 신호가 커졌습니다.
- 최근 경제지표 변동성이 확대되며 시장 불안 신호가 함께 관측되었습니다.

#### 13.3.2 모델 해석 기반 설명

LightGBM 모델은 feature importance와 SHAP을 활용해 설명 가능성을 확보할 수 있다.

1차 MVP에서는 전체 feature importance를 우선 제공한다.  
추후 고도화 단계에서는 국가-날짜별 local SHAP explanation을 추가하여 특정 예측값을 높인 요인을 보여준다.

설명 예시:

- 이 예측값을 높인 주요 요인 TOP 5
  1. acled_event_count_7d
  2. acled_fatalities_14d
  3. gdelt_tone_mean_7d
  4. gdelt_mentions_sum_7d
  5. econ_vix_pct_7d

이를 대시보드에서는 다음과 같이 표현한다.

- 최근 실제 분쟁 이벤트와 사상자 관련 신호가 증가했고, 뉴스 tone이 악화되면서 escalation 위험 점수가 상승했습니다.

#### 13.3.3 관련 뉴스 및 이벤트 연결

관련 기사 또는 이벤트 근거는 단계적으로 구현한다.

1차 MVP에서는 별도 뉴스 API를 붙이지 않고, 현재 데이터에 포함된 ACLED/GDELT feature 신호를 근거로 표시한다.

2차 버전에서는 GDELT DOC API 또는 별도 뉴스 API를 활용하여 선택 국가와 선택 날짜 주변의 관련 기사 제목, 출처, 날짜, URL을 보여준다.

관련 뉴스 UI 예시:

- 기사 제목
- 출처
- 날짜
- URL
- 한국어 요약 한 줄

다만 기사 원문 링크가 없는 경우에는 feature 기반 근거와 ACLED/GDELT 이벤트 신호 요약으로 대체한다.

#### 13.3.4 국가 상세 패널에서의 표시 방식

국가를 클릭하면 다음 정보를 함께 보여준다.

1. 현재 `y_prob`
2. `risk_score`
3. `risk_level`
4. 위험도를 높인 주요 feature TOP 5
5. 최근 7일 또는 30일 feature 변화 요약
6. 관련 ACLED/GDELT 신호 요약
7. 관련 뉴스 또는 이벤트 링크
8. 한국 기준 영향 태그

#### 13.3.5 구현 우선순위

1차 MVP:

- feature 기반 위험 설명
- 전체 LightGBM feature importance
- 국가별 위험도 상승 요인 텍스트 생성
- 관련 뉴스 링크 없이 데이터 신호 중심 설명

2차 MVP:

- SHAP local explanation
- 관련 뉴스 API 연결
- 국가별 관련 기사 TOP 3 표시
- 기사 요약 및 한국어 브리핑 생성

3차 고도화:

- 실제 ACLED 이벤트 마커
- GDELT 기사 링크
- 한국 기준 영향 태그 자동 생성
- 모델 설명과 뉴스 근거를 결합한 자동 브리핑

### 13.4 국가별 시계열 그래프

그래프 요구사항:

1. x축: date
2. y축: `y_prob` 또는 `risk_score`
3. 선 그래프 형태
4. 위험 기준선 표시 가능
5. 실제 이벤트 마커는 추후 추가

### 13.5 오늘의 브리핑

브리핑 카드는 다음 내용을 포함한다.

1. 위험도 급등 국가 TOP 3
2. 국가별 2~3줄 요약
3. 한국 기준 영향 태그
   - 유가
   - 환율
   - 공급망
   - 교민/여행
   - 투자심리
4. 데이터 기반 근거
   - `y_prob` 상승
   - ACLED event count 증가
   - GDELT tone 악화
   - 경제지표 변동

### 13.6 모델 비교 기능

복수 모델의 prediction 파일을 비교할 수 있도록 설계한다.

예시 파일명:

- `predictions__lightgbm__byeonghyeon.csv`
- `predictions__xgboost__teammate.csv`
- `predictions__lstm__teammate.csv`

화면 요소:

1. 모델 선택 드롭다운
2. 선택된 모델의 예측 결과 표시
3. 모델별 위험도 상위 국가 비교
4. 가능하면 모델별 성능 지표 카드 표시

---

## 14. UI/UX 방향

### 14.1 전체 디자인 방향

1. 발표용으로 직관적이고 깔끔해야 한다.
2. 너무 복잡한 분석 툴처럼 보이지 않게 한다.
3. 지도, 카드, 그래프, 브리핑이 한 화면에서 자연스럽게 연결되어야 한다.
4. “왜 이 점수인지”를 반드시 보여준다.
5. 지도 구현보다 데이터 흐름, 카드, 그래프, 브리핑을 먼저 완성한다.

### 14.2 추천 레이아웃

상단 헤더:

- 서비스명
- 날짜 선택
- 모델 선택

상단 KPI 영역:

- 평균 위험도
- Critical 국가 수
- 위험도 상승 TOP 국가
- 데이터 기준일

중앙 영역:

- 왼쪽: 지도 또는 국가 리스트
- 오른쪽: 선택 국가 상세 패널

하단 영역:

- 시계열 그래프
- 오늘의 브리핑 카드

### 14.3 색상 방향

| Risk Level | 색상 방향 |
|---|---|
| Low | 차가운 색 |
| Moderate | 노란색 계열 |
| High | 주황색 계열 |
| Critical | 빨간색 계열 |
| No Data | 회색 |

구현 시 Tailwind CSS 기본 색상 체계를 사용한다.

---

## 15. 이번 MVP에서 하지 않을 것

- 실시간 API 연동
- Supabase 연결
- 사용자 로그인
- 복잡한 지도 인터랙션
- raw 데이터 기반 피처 재생성
- Telegram/Reddit 소셜 데이터 통합
- 고급 딥러닝 모델 구현
- test set 기반 모델 선택
- 완전한 production 배포
- 운영 수준의 alert system

---

## 16. Claude Code 작업 원칙

1. 먼저 모델링 baseline을 만든다.
2. 웹앱은 predictions CSV가 생성된 이후 연결한다.
3. 모든 코드는 재실행 가능해야 한다.
4. test set은 예측에만 사용한다.
5. feature leakage를 방지한다.
6. 제출 CSV 검증 함수를 반드시 포함한다.
7. 한 번에 전체를 만들지 말고 Phase별로 구현한다.
8. Claude Code는 프로젝트 루트인 `conflict-warning-dashboard`에서 실행한다.
9. Claude Code는 `PRD.md`와 `team-collaboration-guide.md`를 먼저 읽고 작업한다.
10. 큰 수정 전에는 먼저 구현 계획을 제안한다.

---

## 17. Claude Code 첫 작업 프롬프트

Claude Code 실행 후 다음 프롬프트를 사용한다.

PRD.md와 team-collaboration-guide.md를 읽고 현재 프로젝트를 이해해줘.

현재 목표는 먼저 웹앱이 아니라 모델링 baseline을 만드는 거야.  
input/processed/dataset/train.parquet, val.parquet, test.parquet를 사용해서 y_escalation을 예측하는 LightGBM baseline을 만들고 싶어.

아직 코드를 작성하지 말고 먼저 다음을 확인해줘.

1. 현재 폴더 구조가 적절한지
2. 어떤 파일을 입력으로 사용할지
3. 어떤 컬럼을 feature에서 제외해야 하는지
4. train/val/test를 어떻게 사용할지
5. 최종 제출 CSV 형식은 무엇인지
6. 만들 Python 파일 구조는 어떻게 할지
7. 실행 순서는 어떻게 될지

그다음 내가 승인하면 modeling 폴더와 Python 파일들을 생성해줘.

---

## 18. Claude Code 구현 승인 프롬프트

Claude Code가 계획을 제안한 뒤, 계획이 적절하면 다음 프롬프트를 사용한다.

좋아. 그 계획대로 구현해줘.

modeling 폴더를 만들고 아래 파일을 생성해줘.

- train_lightgbm.py
- predict_lightgbm.py
- evaluate.py
- utils.py
- requirements.txt
- README.md

구현 조건:

- target은 y_escalation
- LABEL_META_COLS는 feature에서 반드시 제외
- date는 feature에서 제외
- country는 categorical feature로 사용
- train으로 학습하고 val로 평가
- test는 predict_lightgbm.py에서 예측만 수행
- 최종 출력은 outputs/predictions/predictions__lightgbm__byeonghyeon.csv
- 컬럼은 date,country,y_prob
- test set 전체 행이 빠짐없이 들어가도록 검증 함수 포함
- 먼저 기본 LightGBM baseline만 구현하고 Optuna, SE score, SHAP은 이후 단계로 미뤄줘.

---

## 19. MVP 완료 기준

다음 조건을 만족하면 1차 MVP 완료로 본다.

1. parquet 데이터 로드 가능
2. `y_escalation` 기준 LightGBM 학습 가능
3. validation PR-AUC 출력
4. P@top5%, ECE 등 기본 지표 출력
5. test set 예측 생성
6. 제출 CSV 형식 검증
7. `outputs/predictions/predictions__lightgbm__byeonghyeon.csv` 저장
8. 이후 대시보드에서 해당 CSV를 읽을 수 있는 구조 확보
9. 웹앱에서 국가별 위험도 테이블 또는 카드 표시 가능
10. 국가별 시계열 그래프를 연결할 수 있는 구조 확보

---

## 20. 향후 추가 기능

1. LightGBM + SE score 실험
2. Optuna 기반 하이퍼파라미터 튜닝
3. SHAP feature importance 시각화
4. 모델별 prediction 비교
5. 세계 지도 choropleth 고도화
6. 실제 ACLED 이벤트 마커 표시
7. GDELT 기사 링크 연결
8. 자동 브리핑 생성
9. Supabase 연결
10. Vercel 배포
11. 모바일 반응형 개선
12. 팀원별 모델 결과 통합 대시보드
13. Feature 기반 위험도 설명 자동 생성
14. 국가-날짜별 SHAP local explanation
15. 관련 뉴스 제목·출처·URL 표시