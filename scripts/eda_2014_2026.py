# %% [markdown]
# # 2014~2026 합본 데이터 EDA (12년치)
#
# > 5/9 raw_merged 통합 (2014-01-01 ~ 2026-03-28, 12.24년, 58국, 259,260행) 기준
# > 모델 재학습 전 데이터 구조·이상치·신호 강도 사전 점검
#
# **목적**:
# 1. 어떤 국가/시기에 어떤 소스가 비어있는지 (커버리지)
# 2. 양성 라벨이 시간/국가별로 어떻게 분포하는지 (drift, 클러스터)
# 3. 피처별 신호 강도 (단변량 lift)
# 4. 피처간 중복도 (correlation)
# 5. 백테스트 케이스에서 모델이 잡을 수 있는 신호가 보이는지 (raw 시계열)
#
# **사용법**: `python scripts/eda_2014_2026.py` — `output/evaluation/eda-plots/`에 7장 PNG 생성. VSCode interactive로 셀 단위 실행 가능 (`# %%` 마커).

# %% Imports & paths
from __future__ import annotations
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")  # non-interactive backend — script-safe (plt.show is no-op)
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "input/processed/dataset"
PROCESSED = PROJECT_ROOT / "input/processed"
RAW_MERGED = PROJECT_ROOT / "input/raw_merged"
PLOT_DIR = PROJECT_ROOT / "output/evaluation/eda-plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Korean font (macOS)
for cand in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"]:
    if any(cand in f.name for f in mpl.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 120


def save_fig(name: str, fig=None):
    fig = fig or plt.gcf()
    out = PLOT_DIR / f"{name}.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


# %% Load datasets
full = pd.read_parquet(DATASET / "full.parquet")
train = pd.read_parquet(DATASET / "train.parquet")
val = pd.read_parquet(DATASET / "val.parquet")
test = pd.read_parquet(DATASET / "test.parquet")
baseline = pd.read_parquet(PROCESSED / "features/baseline_scores.parquet")

LABEL_COLS = [
    "y", "y_onset", "y_escalation",
    "fatalities_next3d", "event_count_next3d",
    "past14d_event_count", "past14d_fatalities_mean",
]
META_COLS = ["date", "country"]
FEATURE_COLS = [c for c in full.columns if c not in META_COLS + LABEL_COLS]

ACLED_COLS = [c for c in FEATURE_COLS if c.startswith("acled_")]
GDELT_COLS = [c for c in FEATURE_COLS if c.startswith("gdelt_")]
ECON_COLS = [c for c in FEATURE_COLS if c.startswith("econ_")]

print(f"full  : {len(full):>7,}행 ({full.date.min().date()} ~ {full.date.max().date()}), 58국")
print(f"train : {len(train):>7,}행 (~{train.date.max().date()}), y_escalation={train.y_escalation.mean():.4f}")
print(f"val   : {len(val):>7,}행 ({val.date.min().date()}~{val.date.max().date()}), y_escalation={val.y_escalation.mean():.4f}")
print(f"test  : {len(test):>7,}행 ({test.date.min().date()}~{test.date.max().date()}), y_escalation={test.y_escalation.mean():.4f}")
print(f"\nfeatures : {len(FEATURE_COLS)}개 (ACLED {len(ACLED_COLS)} + GDELT {len(GDELT_COLS)} + econ {len(ECON_COLS)})")


# %% [markdown]
# ## ① 국가별 커버리지 매트릭스 (ACLED·GDELT·경제)
#
# raw_merged의 실제 데이터 존재 여부를 국가 × 연도 매트릭스로 시각화.
# **확인 포인트**: 2014-2017 ACLED 결측 패턴이 어느 국가에 집중되는지.

# %%
def acled_coverage():
    cov = {}
    for p in sorted((RAW_MERGED / "acled").glob("*.parquet")):
        df = pd.read_parquet(p, columns=["event_date"])
        df["year"] = pd.to_datetime(df["event_date"]).dt.year
        cov[p.stem] = df.groupby("year").size()
    return pd.DataFrame(cov).T.fillna(0).astype(int)


def gdelt_coverage():
    cov = {}
    for p in sorted((RAW_MERGED / "gdelt").glob("*.parquet")):
        df = pd.read_parquet(p, columns=["SQLDATE"])
        df["year"] = (df["SQLDATE"] // 10000).astype(int)
        cov[p.stem] = df.groupby("year").size()
    return pd.DataFrame(cov).T.fillna(0).astype(int)


print("ACLED 커버리지 산출 중...")
acled_cov = acled_coverage()
print("GDELT 커버리지 산출 중...")
gdelt_cov = gdelt_coverage()

# Annual coverage summary (몇 개국이 데이터 보유)
annual_summary = pd.DataFrame({
    "ACLED": (acled_cov > 0).sum(),
    "GDELT": (gdelt_cov > 0).sum(),
})
print("\n연도별 데이터 보유 국가 수 (전체 58국 중):")
print(annual_summary)

# %%
fig, axes = plt.subplots(1, 2, figsize=(20, 14))

for ax, cov, name in [(axes[0], acled_cov, "ACLED"), (axes[1], gdelt_cov, "GDELT")]:
    cov_sorted = cov.reindex(sorted(cov.index))
    im = ax.imshow(np.log1p(cov_sorted.values), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(cov_sorted.columns)))
    ax.set_xticklabels(cov_sorted.columns, rotation=45)
    ax.set_yticks(range(len(cov_sorted.index)))
    ax.set_yticklabels(cov_sorted.index, fontsize=7)
    ax.set_title(f"{name} 이벤트 수 log1p (행 = 국가, 열 = 연도)")
    ax.set_xlabel("연도")
    plt.colorbar(im, ax=ax, fraction=0.04)

plt.suptitle("① 국가별 커버리지 매트릭스 — 어두울수록 결측", y=1.00)
plt.tight_layout()
save_fig("01_coverage")
plt.show()

# Save annual summary as csv
annual_summary.to_csv(PLOT_DIR / "01_coverage_annual.csv")


# %% [markdown]
# ## ② 타겟 분포 시계열
#
# 일별 `y_escalation = 1` 합산. train/val/test 경계 + 백테스트 3케이스 마커.

# %%
daily = full.groupby("date").agg(
    pos=("y_escalation", "sum"),
    n=("y_escalation", "size"),
)
daily["rate"] = daily["pos"] / daily["n"]

split_train_end = pd.Timestamp("2023-12-31", tz="UTC")
split_val_end = pd.Timestamp("2024-06-30", tz="UTC")
events = [
    (pd.Timestamp("2022-02-24", tz="UTC"), "Ukraine"),
    (pd.Timestamp("2023-04-15", tz="UTC"), "Sudan"),
    (pd.Timestamp("2023-10-07", tz="UTC"), "Gaza"),
]

fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

ax = axes[0]
ax.plot(daily.index, daily["pos"], lw=0.6, color="steelblue")
ax.axvline(split_train_end, color="red", ls="--", lw=1, label="train→val (2023-12-31)")
ax.axvline(split_val_end, color="darkred", ls="--", lw=1, label="val→test (2024-06-30)")
for d, name in events:
    ax.axvline(d, color="orange", alpha=0.6, lw=0.8)
    ax.text(d, daily["pos"].max() * 0.92, name, rotation=90, fontsize=8, ha="right", color="darkorange")
ax.set_ylabel("일별 양성국 수\n(y_escalation=1)")
ax.set_title("② 타겟 분포 시계열 (2014-01 ~ 2026-03)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(daily.index, daily["rate"].rolling(30, min_periods=1).mean(), lw=0.8, color="darkgreen", label="30일 rolling 양성비")
ax.axvline(split_train_end, color="red", ls="--", lw=1)
ax.axvline(split_val_end, color="darkred", ls="--", lw=1)
ax.set_ylabel("양성비 (rolling 30d)")
ax.set_xlabel("날짜")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
save_fig("02_target_timeseries")
plt.show()

print(f"\n전체 양성일: {daily['pos'].sum():,}회")
print(f"양성국 평균(일별): {daily['pos'].mean():.2f}국")
print(f"양성률 30일 rolling 최대: {daily['rate'].rolling(30, min_periods=1).mean().max():.4f} @ {daily['rate'].rolling(30, min_periods=1).mean().idxmax().date()}")


# %% [markdown]
# ## ③ 국가별 양성비 (train vs test)
#
# 58국별로 train/test의 `y_escalation` 양성비. **drift 식별**: train과 test에서 양성 비율이 크게 다른 국가는 distribution shift 의심.

# %%
train_rate = train.groupby("country")["y_escalation"].mean()
test_rate = test.groupby("country")["y_escalation"].mean()
df_rate = pd.DataFrame({"train": train_rate, "test": test_rate}).fillna(0)
df_rate["diff"] = df_rate["test"] - df_rate["train"]
df_rate = df_rate.sort_values("train", ascending=False)

fig, ax = plt.subplots(figsize=(16, 7))
x = np.arange(len(df_rate))
ax.bar(x - 0.2, df_rate["train"], width=0.4, label="train (2014~2023)", color="steelblue", alpha=0.85)
ax.bar(x + 0.2, df_rate["test"], width=0.4, label="test (2024.07~2025.03)", color="firebrick", alpha=0.85)
ax.axhline(train.y_escalation.mean(), color="steelblue", ls=":", lw=1, label=f"train 평균 {train.y_escalation.mean():.3f}")
ax.axhline(test.y_escalation.mean(), color="firebrick", ls=":", lw=1, label=f"test 평균 {test.y_escalation.mean():.3f}")
ax.set_xticks(x)
ax.set_xticklabels(df_rate.index, rotation=90, fontsize=8)
ax.set_ylabel("y_escalation 양성비")
ax.set_title("③ 국가별 y_escalation 양성비 — train(파랑) vs test(빨강)")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save_fig("03_country_positive_rate")
plt.show()

# Top 5 drift 국가
print("\nTrain→Test 양성비 변화 Top 5 증가:")
print(df_rate.nlargest(5, "diff")[["train", "test", "diff"]].round(4))
print("\nTrain→Test 양성비 변화 Top 5 감소:")
print(df_rate.nsmallest(5, "diff")[["train", "test", "diff"]].round(4))


# %% [markdown]
# ## ④ 피처 분포 + skew (54피처)
#
# 히스토그램 grid + skewness 표. **log 변환 후보**: skew > 5 (heavy right-tailed).

# %%
fig, axes = plt.subplots(9, 6, figsize=(20, 24))
for i, col in enumerate(FEATURE_COLS):
    ax = axes.flat[i]
    vals = train[col].replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(vals, bins=50, color="steelblue", edgecolor="black", linewidth=0.3)
    ax.set_title(col, fontsize=8)
    ax.tick_params(axis="both", labelsize=6)
# Hide unused axes
for j in range(len(FEATURE_COLS), len(axes.flat)):
    axes.flat[j].axis("off")
plt.suptitle("④ 피처 분포 — train (54피처)", y=1.001, fontsize=14)
plt.tight_layout()
save_fig("04_feature_distributions")
plt.show()

# Skewness table
skew = train[FEATURE_COLS].skew().sort_values(ascending=False)
print(f"\n상위 skew 15개 (log1p 변환 후보):")
print(skew.head(15).round(2))
skew.to_csv(PLOT_DIR / "04_feature_skew.csv", header=["skew"])


# %% [markdown]
# ## ⑤ 단변량 lift (피처별 quantile bin → 양성률)
#
# 각 피처를 10분위로 나눠 분위별 `y_escalation` 양성률. 전체 평균 대비 lift = bin 양성률 / overall.
# **강한 신호 피처**: top decile lift ≥ 5×.

# %%
overall = train.y_escalation.mean()
lift_table = []

fig, axes = plt.subplots(9, 6, figsize=(20, 24))
for i, col in enumerate(FEATURE_COLS):
    ax = axes.flat[i]
    try:
        bins = pd.qcut(train[col], q=10, duplicates="drop")
        lift = train.groupby(bins, observed=True)["y_escalation"].mean()
        ax.bar(range(len(lift)), lift.values, color="darkgreen", alpha=0.7)
        ax.axhline(overall, color="red", ls="--", lw=0.8)
        top = lift.iloc[-1] / overall if overall > 0 else 0
        ax.set_title(f"{col}\n top-decile lift={top:.1f}x", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        lift_table.append((col, lift.iloc[-1], top))
    except Exception as e:
        ax.set_title(f"{col} (skip)", fontsize=7)
        ax.axis("off")
for j in range(len(FEATURE_COLS), len(axes.flat)):
    axes.flat[j].axis("off")
plt.suptitle(f"⑤ 단변량 lift — overall = {overall:.4f}, 빨간 선 = 평균", y=1.001, fontsize=14)
plt.tight_layout()
save_fig("05_univariate_lift")
plt.show()

# Top lift table
df_lift = pd.DataFrame(lift_table, columns=["feature", "top_decile_rate", "lift"]).sort_values("lift", ascending=False)
print(f"\n상위 lift 15개 (top decile 양성률 / overall):")
print(df_lift.head(15).round(3))
df_lift.to_csv(PLOT_DIR / "05_lift.csv", index=False)


# %% [markdown]
# ## ⑥ 피처 correlation matrix
#
# 54피처 상관행렬. **rolling window (7d/14d/30d) 중복도**, **economic 피처군 내부 상관도** 시각 확인.

# %%
corr = train[FEATURE_COLS].corr()
fig, ax = plt.subplots(figsize=(18, 16))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(FEATURE_COLS)))
ax.set_xticklabels(FEATURE_COLS, rotation=90, fontsize=6)
ax.set_yticks(range(len(FEATURE_COLS)))
ax.set_yticklabels(FEATURE_COLS, fontsize=6)
plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
ax.set_title("⑥ 피처 correlation (Pearson) — 54x54")

# Group separators
boundaries = [len(ACLED_COLS), len(ACLED_COLS) + len(GDELT_COLS)]
for b in boundaries:
    ax.axvline(b - 0.5, color="black", lw=0.8)
    ax.axhline(b - 0.5, color="black", lw=0.8)

plt.tight_layout()
save_fig("06_correlation")
plt.show()

# High correlation pairs
abs_corr = corr.abs()
mask = np.triu(np.ones_like(abs_corr, dtype=bool), k=1)
pairs = abs_corr.where(mask).stack().sort_values(ascending=False)
print(f"\n|상관| > 0.95 쌍 ({(pairs > 0.95).sum()}개):")
print(pairs[pairs > 0.95].head(20).round(3))


# %% [markdown]
# ## ⑦ 백테스트 3케이스 D-30 ~ D+7 raw 시계열
#
# 우크라이나 2022-02-24 / 수단 2023-04-15 / 가자 2023-10-07.
# 핵심 피처(ACLED 7d 사상자, GDELT goldstein/mentions)가 D-day 전후로 어떻게 변하는지 사전 확인.
# **이 그래프에서 D-day 전 신호가 보이지 않으면 모델도 못 잡음** → 피처 엔지니어링 보강 필요 신호.

# %%
backtest = [
    ("UKR", pd.Timestamp("2022-02-24", tz="UTC"), "Ukraine"),
    ("SDN", pd.Timestamp("2023-04-15", tz="UTC"), "Sudan"),
    ("PSE", pd.Timestamp("2023-10-07", tz="UTC"), "Gaza"),
]

# country code → name in our country col (data uses ISO3 in country col? check)
country_to_iso = {"Ukraine": "UKR", "Sudan": "SDN", "Palestine": "PSE"}
sample_iso = full["country"].iloc[0]
country_col_is_iso = len(sample_iso) == 3 and sample_iso.isupper()
print(f"country col format: {'ISO3' if country_col_is_iso else 'name'} (sample: {sample_iso})")

key_features = [
    "acled_event_count_7d",
    "acled_fatalities_7d",
    "gdelt_goldstein_mean_7d",
    "gdelt_tone_mean_7d",
    "gdelt_mentions_sum_7d",
    "gdelt_quadclass_4_ratio",
]

fig, axes = plt.subplots(len(backtest), len(key_features), figsize=(22, 9), sharey="col")

for r, (iso, event_date, name) in enumerate(backtest):
    if country_col_is_iso:
        sub = full[full["country"] == iso].copy()
    else:
        # map ISO3 to country name
        iso2name = {"UKR": "Ukraine", "SDN": "Sudan", "PSE": "Palestine"}
        sub = full[full["country"] == iso2name[iso]].copy()
    sub = sub.sort_values("date")
    window = sub[(sub.date >= event_date - pd.Timedelta(days=30)) & (sub.date <= event_date + pd.Timedelta(days=7))]

    for c, feat in enumerate(key_features):
        ax = axes[r, c]
        ax.plot(window["date"], window[feat], lw=1.2, color="steelblue", marker="o", ms=2)
        ax.axvline(event_date, color="red", ls="--", lw=1)
        ax.set_title(f"{name} — {feat}", fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.3)

plt.suptitle("⑦ 백테스트 3케이스 D-30 ~ D+7 raw 피처 시계열 (빨간선 = 사건일)", y=1.00, fontsize=12)
plt.tight_layout()
save_fig("07_backtest_raw")
plt.show()


# %% [markdown]
# ## 요약 — 다음 행동 후보
#
# 결과 PNG·CSV는 모두 `output/evaluation/eda-plots/`에 저장됨. 이 셀들 실행 후 출력 보고 다음을 결정:
#
# 1. **2014-2017 ACLED 결측 처리**: ① 커버리지 매트릭스 + train/test 양성비 변화 ③ 보고 country별 ACLED 시작일 이전 행 제외할지 결정
# 2. **log 변환 피처**: ④ skew Top 15 → log1p 변환할 피처 목록 확정 (특히 fatalities_max_*, mentions_sum_*)
# 3. **drop 피처 후보**: ⑥ |상관| > 0.95 쌍 → rolling 7d/14d/30d 중 하나만 살릴지 검토 (LightGBM은 견디지만 LogReg/LSTM 위해)
# 4. **신호 우선순위**: ⑤ top-decile lift Top 15 → SHAP 결과와 대조용 baseline
# 5. **백테스트 사전 진단**: ⑦ Ukraine D-day 전 신호 강도 vs Sudan/Gaza 비교 — 사전 신호가 거의 없는 케이스는 모델 한계 명시
