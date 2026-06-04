# GDELT Title C2 Feature Builder 실행 가이드

## 목적

기존 `gdelt_title_features.parquet`를 재사용해 **BigQuery 추가 비용 없이**  
3일 rolling / spike / country-normalized 파생 피처를 로컬에서 생성한다.

실험 C2 = C feature set + C2 파생 피처를 추가한 실험.

---

## 입력 / 출력

| 항목 | 경로 |
|------|------|
| 입력 | `members/byeonghyeon/input/processed/gdelt_titles/gdelt_title_features.parquet` |
| 출력 | `members/byeonghyeon/input/processed/gdelt_titles/gdelt_title_c2_features.parquet` |

입력 파일은 `build_gdelt_title_features.py`로 생성한 C 피처 parquet (214,257행 × 23컬럼).

---

## 생성 피처 (19개)

### 3일 rolling (9개)
1d 컬럼을 country별 date 정렬 후 과거 방향 3행 rolling으로 집계.

| 피처명 | 계산 기준 |
|--------|----------|
| `gdelt_title_count_3d` | sum(count_1d, 3) |
| `gdelt_title_nonnull_count_3d` | sum(nonnull_count_1d, 3) |
| `gdelt_title_negative_count_3d` | sum(negative_count_1d, 3) |
| `gdelt_title_positive_count_3d` | sum(positive_count_1d, 3) |
| `gdelt_title_eng_count_3d` | sum(eng_count_1d, 3) |
| `gdelt_title_tone_mean_3d` | mean(tone_mean_1d, 3) |
| `gdelt_title_tone_min_3d` | min(tone_min_1d, 3) |
| `gdelt_title_domain_diversity_3d` | mean(domain_diversity_1d, 3) |
| `gdelt_title_lang_diversity_3d` | mean(lang_diversity_1d, 3) |

### Spike feature (5개)
3일 일평균과 7일/14일 일평균의 비율. **1보다 크면 최근 급증**.

| 피처명 | 계산식 | 해석 |
|--------|--------|------|
| `gdelt_title_count_3d_vs_7d` | (count_3d/3) / (count_7d/7 + ε) | 보도량 3d 급증 vs 7d |
| `gdelt_title_count_3d_vs_14d` | (count_3d/3) / (count_14d/14 + ε) | 보도량 3d 급증 vs 14d |
| `gdelt_title_negative_3d_vs_7d` | (neg_3d/3) / (neg_7d/7 + ε) | 부정 보도 3d 급증 vs 7d |
| `gdelt_title_negative_3d_vs_14d` | (neg_3d/3) / (neg_14d/14 + ε) | 부정 보도 3d 급증 vs 14d |
| `gdelt_title_tone_drop_3d_vs_7d` | tone_mean_7d − tone_mean_3d | 최근 3일 톤 하락 (양수=악화) |

> `count_14d`와 `negative_count_14d`는 기존 parquet에 없어 1d 컬럼으로 재계산.

### Country-normalized feature (5개)
각 country 내에서 과거 90일 rolling mean/std를 기준으로 정규화.  
**미래 정보 없음**: 90d rolling에 current row 포함, PRECEDING 방향만 사용.

| 피처명 | 계산식 | 해석 |
|--------|--------|------|
| `gdelt_title_count_7d_country_z_90d` | (count_7d - mean_90d) / (std_90d + ε) | 보도량 z-score |
| `gdelt_title_negative_count_7d_country_z_90d` | 동일 | 부정 보도량 z-score |
| `gdelt_title_tone_mean_7d_country_z_90d` | 동일 | 톤 z-score |
| `gdelt_title_count_7d_country_ratio_90d` | count_7d / (mean_90d + ε) | 보도량 ratio |
| `gdelt_title_negative_count_7d_country_ratio_90d` | 동일 | 부정 보도량 ratio |

> z-score > 0: 최근 보도가 해당 국가 90일 평균보다 높음  
> ratio > 1: 최근 보도가 해당 국가 90일 평균보다 많음

---

## Leakage 방지

| 규칙 | 구현 |
|------|------|
| 과거 방향 rolling | `x.rolling(N, min_periods=1)` — 기본이 과거 방향 ✅ |
| 당일 포함, 미래 없음 | rolling window에 FOLLOWING 없음 ✅ |
| country별 독립 계산 | `groupby("country").transform(...)` ✅ |
| Sparse grid 주의 | BQ parquet는 보도 있는 날만 포함 — rolling이 달력일 아닌 행 기준 |

> **Sparse grid 한계**: 보도 없는 날이 parquet에 없어 3d/14d rolling이  
> "직전 3/14 달력일"이 아닌 "직전 3/14 보도일" 기준이 됨.  
> 2015년 이후 대부분의 국가-일에서는 매일 보도가 있으므로 영향 제한적.

---

## 전제 조건

```bash
pip install pandas numpy pyarrow
```

입력 파일이 없으면 먼저 C 피처 parquet를 생성해야 한다:
```bash
python members/byeonghyeon/modeling/build_gdelt_title_features.py
```

---

## 실행 방법

### 기본 실행

```bash
cd /path/to/KUBIG_Conference-team-4

python members/byeonghyeon/modeling/build_gdelt_title_c2_features.py
```

예상 소요 시간: **5~10분** (groupby rolling 연산, 메모리 ~2GB)

### 입력/출력 경로 명시

```bash
python members/byeonghyeon/modeling/build_gdelt_title_c2_features.py \
  --input  members/byeonghyeon/input/processed/gdelt_titles/gdelt_title_features.parquet \
  --output members/byeonghyeon/input/processed/gdelt_titles/gdelt_title_c2_features.parquet
```

### 결과 검증만 (이미 파일이 있는 경우)

```bash
python members/byeonghyeon/modeling/build_gdelt_title_c2_features.py --verify
```

shape, null, 컬럼별 mean/std를 출력하고 종료.

---

## 출력 파일

```
members/byeonghyeon/input/processed/gdelt_titles/
└── gdelt_title_c2_features.parquet
      컬럼: date, country, [19개 C2 피처]
      행수: 214,257행 (C 피처 parquet와 동일)
      크기: 약 20~40 MB 예상
```

> **이 파일은 git에 커밋하지 않는다.** `members/byeonghyeon/input/` 경로는 gitignored.

---

## 학습 스크립트에서의 사용 예시

```python
import pandas as pd
from build_gdelt_title_features import GDELT_TITLE_FEATURE_COLS, BQ_MIN_DATE
from build_gdelt_title_c2_features import GDELT_TITLE_C2_FEATURE_COLS

TITLE_PARQUET = ".../gdelt_title_features.parquet"
C2_PARQUET    = ".../gdelt_title_c2_features.parquet"

gdelt_c_df  = pd.read_parquet(TITLE_PARQUET)
gdelt_c2_df = pd.read_parquet(C2_PARQUET)

# left-join: C 피처 먼저, C2 피처 추가
for split in [train, val, test]:
    split = split.merge(gdelt_c_df,  on=["date", "country"], how="left")
    split = split.merge(gdelt_c2_df, on=["date", "country"], how="left")

# 결측 0 채움 (보도 없는 날 또는 coverage gap)
all_gdelt = GDELT_TITLE_FEATURE_COLS + GDELT_TITLE_C2_FEATURE_COLS
for split in [train, val, test]:
    split[all_gdelt] = split[all_gdelt].fillna(0)

# coverage_mask (C와 공유)
for split in [train, val, test]:
    split["gdelt_title_coverage_mask"] = (
        split["date"] < pd.Timestamp(BQ_MIN_DATE, tz="UTC")
    ).astype(int)

# feature_cols = B 35 + C title 21 + coverage_mask 1 + C2 파생 19 = 76
feature_cols_C2 = FEATURE_COLS_B + GDELT_TITLE_FEATURE_COLS + ["gdelt_title_coverage_mask"] + GDELT_TITLE_C2_FEATURE_COLS
```

---

## 주의사항

1. **메모리**: groupby rolling 연산은 약 1~2 GB RAM 필요.
2. **시간**: 214,257행 × 국가별 rolling은 5~10분 소요.
3. **git 제외**: 출력 parquet는 `members/byeonghyeon/input/` 아래 있어 gitignored.
4. **C 피처와 별도 파일**: C2 피처는 C 피처 parquet를 수정하지 않고 별도 parquet로 저장.
   학습 시 두 parquet를 모두 로드해 merge한다.
