"""서빙 피처 조립 — build_dataset.py 의 join 순서를 복제해 model_input(419) 생성.

서빙 흐름:
  raw(BQ) → [임베딩 적재] → 각 피처 그룹 df → assemble(left-join + acled drop) → model_input
각 그룹 출처:
  - 임베딩 파생 214: src/serve/embedding_features.embedding_derived (저장 아티팩트 transform) ✅로컬검증
  - gkg_pool_ 87   : BQ 제목 pooling (10_title_pooling SQL, 윈도우) — 서빙시 BQ
  - Track1 17      : gkg_feature_builder SQL over gdelt_titles — 서빙시 BQ
  - 비임베딩 ~90   : feature_builder + gdelt enriched (events/econ) — 서빙시 BQ
  - 라벨           : 서빙엔 불필요(추론), 학습/검증에만

verify_local(): 로컬 중간 parquet + 임베딩 모듈로 조립 → full_pca16_aclfree 와 419 parity 확인.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

FEAT = "input/processed/features"
LABEL_META = ["y", "y_onset", "y_escalation", "fatalities_next3d", "event_count_next3d",
              "past14d_event_count", "past14d_fatalities_mean", "past14d_fatalities_std"]

# build_dataset.py join 순서 (features 베이스에 차례로 left-join, 각 fillna 0)
JOIN_ORDER = [
    "gkg_features.parquet",           # Track1 17
    "gkg_embeddings_pca16.parquet",   # pca16 16   (임베딩 그룹 — 서빙은 embedding_derived가 대체)
    "gkg_emb_aux.parquet",            # aux 40
    "gkg_emb_anchors.parquet",        # anchor cos 16
    "gkg_emb_title_pool.parquet",     # pool 87
    "gkg_emb_anchors_temporal.parquet",  # anchT 80
    "gkg_emb_anchors_evtcnt.parquet", # evt 48
    "gkg_emb_riskscore.parquet",      # risk 5
    "gkg_emb_baseline_dev.parquet",   # bdev 4
    "gkg_emb_preesc.parquet",         # preesc 5
    "gdelt_enriched_events.parquet",  # gdev 26
    "gdelt_subnational.parquet",      # gdelt2 10
    "gdelt_acled_mirror.parquet",     # (mirror)
    "gdelt_relative_corrob.parquet",  # gcr 14
]
# embedding_derived(214)가 대체하는 그룹 파일 (서빙에선 이 parquet 대신 모듈 출력 사용)
EMB_DERIVED_FILES = {
    "gkg_embeddings_pca16.parquet", "gkg_emb_aux.parquet", "gkg_emb_anchors.parquet",
    "gkg_emb_anchors_temporal.parquet", "gkg_emb_anchors_evtcnt.parquet",
    "gkg_emb_riskscore.parquet", "gkg_emb_baseline_dev.parquet", "gkg_emb_preesc.parquet",
}


# gcr_(상대/동조)는 조립된 데이터셋의 SIGNALS에서 파생 → 조립 후 계산(post-assembly).
GCR_FILE = "gdelt_relative_corrob.parquet"


def _norm_date(df):
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df


def add_gcr(df, root: str | Path = "."):
    """조립 결과에 gcr_ 14피처 합류 (compute_gcr 재사용). build_dataset 가 corrob 를
    마지막에 join 하는 것과 동일 — 단 raw가 아니라 조립 df 에서 계산."""
    import sys
    sys.path.insert(0, str(Path(root)))
    from scripts.gdelt_relative_corrob import compute_gcr
    gcr = _norm_date(compute_gcr(df))
    cols = [c for c in gcr.columns if c not in {"country", "date"}]
    out = df.merge(gcr, on=["date", "country"], how="left")
    out[cols] = out[cols].fillna(0)
    return out


def assemble(base_features, group_dfs, drop_acled=True):
    """build_dataset 복제: base에 group_dfs 순차 left-join + fillna(0). drop_acled시 acled_ 제거."""
    feat = _norm_date(base_features.copy())
    if drop_acled:
        feat = feat.drop(columns=[c for c in feat.columns if c.startswith("acled_")])
    for g in group_dfs:
        g = _norm_date(g.copy())
        cols = [c for c in g.columns if c not in {"country", "date"}]
        feat = feat.merge(g, on=["date", "country"], how="left")
        for c in cols:  # build_dataset 와 동일한 결측 처리
            if c in ("gkg_missing_mask", "gkg_emb_missing_mask"):
                feat[c] = feat[c].fillna(1).astype("int64")           # 결측 = GKG/임베딩 데이터 없음
            elif c == "page_title_available_flag":
                feat[c] = (feat["date"] >= pd.Timestamp("2019-09-23")).astype("int64")
            else:
                feat[c] = feat[c].fillna(0)
    return feat


# ── 로컬 419 parity 검증 ──────────────────────────────────────────────
def verify_local(countries=("SDN", "UKR", "NGA"), root: str | Path = "."):
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import sys
    sys.path.insert(0, str(root))
    from src.serve.embedding_features import load_artifacts, embedding_derived
    root = Path(root)
    cset = list(countries)

    def load(fn, cols=None):
        return ds.dataset(root / FEAT / fn).to_table(
            filter=pc.field("country").isin(cset), columns=cols).to_pandas()

    base = load("features.parquet")
    # 임베딩 파생 214 (모듈)
    art = load_artifacts(root)
    emb = ds.dataset(root / FEAT / "gkg_embeddings.parquet").to_table(
        filter=pc.field("country").isin(cset)).to_pandas()
    emb = _norm_date(emb)
    emb_derived = embedding_derived(emb, art)

    # 조립: JOIN_ORDER 순서대로. 임베딩 그룹은 emb_derived 한 번으로 대체.
    groups, inserted_emb = [], False
    for fn in JOIN_ORDER:
        if fn in EMB_DERIVED_FILES:
            if not inserted_emb:
                groups.append(emb_derived); inserted_emb = True
            continue  # 나머지 임베딩 intermediate는 건너뜀(모듈이 대체)
        if fn == GCR_FILE:
            continue  # gcr_ 는 조립 후 compute_gcr 로 계산(post-assembly)
        p = root / FEAT / fn
        if p.exists():
            groups.append(load(fn))
    built = assemble(base, groups, drop_acled=True)
    built = add_gcr(built, root)   # gcr_ 14피처 post-assembly 합류

    # 비교 대상
    full = ds.dataset(root / "input/processed/dataset/full_pca16_aclfree.parquet").to_table(
        filter=pc.field("country").isin(cset)).to_pandas()
    full = _norm_date(full)
    feat_cols = [c for c in full.columns if c not in {"country", "date"}
                 and c not in LABEL_META and not c.startswith("acled_")]
    m = built.merge(full, on=["country", "date"], suffixes=("_b", "_f"))
    print(f"검증 국가 {cset}, 병합 {len(m)}행")
    common = [c for c in feat_cols if f"{c}_b" in m.columns and f"{c}_f" in m.columns]
    missing = [c for c in feat_cols if f"{c}_b" not in m.columns]
    bad = []
    for c in common:
        a, b = pd.to_numeric(m[f"{c}_b"], errors="coerce"), pd.to_numeric(m[f"{c}_f"], errors="coerce")
        d = np.nanmax(np.abs(a.values - b.values)) if len(m) else 0
        if d >= 1e-4:
            bad.append((c, d))
    print(f"비교 피처 {len(common)}개 / 누락 {len(missing)}개")
    if missing:
        print(f"  누락(빌더 미산출): {missing[:12]}{'...' if len(missing) > 12 else ''}")
    if bad:
        print(f"  ❌ 불일치 {len(bad)}개:")
        for c, d in sorted(bad, key=lambda x: -x[1])[:15]:
            print(f"     {c}: max|Δ|={d:.2e}")
    else:
        print(f"  ✅ 비교한 {len(common)}개 피처 전부 max|Δ| < 1e-4")


if __name__ == "__main__":
    verify_local()
