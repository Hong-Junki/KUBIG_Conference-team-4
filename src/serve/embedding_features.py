"""임베딩 기반 피처(country-day 평균 1536임베딩 → 파생) 서빙 변환.

학습 파이프라인(scripts/gkg_embed/06·07·08·11·13·14·15)을 복제하되, fit 아티팩트는
저장본을 load 후 transform만 한다(refit 금지 = 누수 차단). 입력은
(country, date, gkg_emb_0..1535, gkg_emb_n_titles_1d) country-day 평균 임베딩 DataFrame.

검증(parity_check): SDN 표본에서 학습 데이터셋 저장값과 비교 → max|Δ| < 1e-4.

구현/검증 범위:
  [VERIFIED] gkg_emb_pca_*(16), gkg_anchor_cos_*(16), gkg_anchT_*(80),
             gkg_risk_*(5), gkg_bdev_*(4), gkg_emb_delta*/cosdiss*(4), gkg_ntitles_*(4),
             gkg_cluster_dist_*(32), gkg_preesc_*(5)  = 166개 (country-day 평균으로 재현)
  [BQ 의존]  gkg_evt_*(48, train p90 임계값 필요), gkg_pool_*(87, 제목단위 BQ 임베딩 필요)
             → 별도 모듈(pool_features.py)에서 처리
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ART_DIR = "output/models/gkg_pca"
ANCHOR_NPZ = "input/processed/features/anchor_embeddings.npz"
RISK5 = ["armed_conflict", "civilian_casualties", "coup", "border_clash", "terrorism"]
EPS = 1e-6


def load_artifacts(root: str | Path = ".") -> dict:
    """서빙 시작 시 1회 로드. train 기간으로만 fit된 고정 아티팩트."""
    root = Path(root)
    p16 = pickle.load(open(root / ART_DIR / "pca_16.pkl", "rb"))
    p64 = pickle.load(open(root / ART_DIR / "pca_64.pkl", "rb"))
    km = pickle.load(open(root / ART_DIR / "kmeans_30.pkl", "rb"))
    anc = np.load(root / ANCHOR_NPZ, allow_pickle=True)
    pre = np.load(root / ART_DIR / "preesc_signature.npz")
    evt_thr = pd.read_parquet(root / ART_DIR / "evt_thresholds_p90.parquet").set_index("country")
    return {
        "pca16": p16["pca"], "emb_cols": list(p16["emb_cols"]),
        "pca64": p64["pca"],
        "kmeans": km["kmeans"], "k": km["k"],
        "anchors": anc["anchors"].astype(np.float64),
        "anchor_labels": [str(x) for x in anc["labels"]],
        "preesc_pos": pre["pos_sig"].astype(np.float64),
        "preesc_dir": pre["direction"].astype(np.float64),
        "evt_thr": evt_thr,   # (country × gkg_anchor_cos_*) train p90 임계값
    }


def _cos(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return An @ Bn.T


def _trailing_mean(X, k, include_current):
    """row 기반 trailing 평균 (cumsum). 복제: 15_preesc_signature.trailing_mean."""
    n, d = X.shape
    cs = np.vstack([np.zeros((1, d)), np.cumsum(X.astype(np.float64), axis=0)])
    out = np.zeros((n, d), dtype=np.float64)
    for i in range(n):
        lo = max(0, i - k + 1) if include_current else max(0, i - k)
        hi = i + 1 if include_current else i
        cnt = hi - lo
        if cnt > 0:
            out[i] = (cs[hi] - cs[lo]) / cnt
    return out


# ── pointwise: pca16 + anchor cosine (06, 08) ─────────────────────────
def pointwise(emb, art):
    X = emb[art["emb_cols"]].values.astype(np.float64)
    out = emb[["country", "date"]].copy()
    pca = art["pca16"].transform(emb[art["emb_cols"]].values)
    for i in range(pca.shape[1]):
        out[f"gkg_emb_pca_{i}"] = pca[:, i]
    acos = _cos(X, art["anchors"])
    for j, lab in enumerate(art["anchor_labels"]):
        out[f"gkg_anchor_cos_{lab}"] = acos[:, j]
    out["gkg_emb_n_titles_1d"] = emb["gkg_emb_n_titles_1d"].values   # pass-through (제목 수)
    return out


# ── preesc (15): proj/cos 1d·7d + accel, 국가별 row-trailing ──────────
def preesc(emb, art):
    dirv, pos = art["preesc_dir"], art["preesc_pos"]
    pos_n = np.linalg.norm(pos) + EPS
    parts = []
    for country, g in emb.groupby("country", sort=False):
        g = g.sort_values("date")
        X = g[art["emb_cols"]].values.astype(np.float64)
        V7 = _trailing_mean(X, 7, include_current=True)
        proj1, proj7 = X @ dirv, V7 @ dirv
        cos1 = (X @ pos) / (np.linalg.norm(X, axis=1) * pos_n + EPS)
        cos7 = (V7 @ pos) / (np.linalg.norm(V7, axis=1) * pos_n + EPS)
        s = pd.Series(proj7)
        accel = (s.rolling(7, min_periods=1).mean()
                 / (s.rolling(30, min_periods=1).mean().abs() + EPS)).values
        parts.append(pd.DataFrame({"country": country, "date": g["date"].values,
                                   "gkg_preesc_proj_1d": proj1, "gkg_preesc_proj_7d": proj7,
                                   "gkg_preesc_cos_1d": cos1, "gkg_preesc_cos_7d": cos7,
                                   "gkg_preesc_proj_accel": accel}))
    return pd.concat(parts, ignore_index=True)


# ── baseline dev (14): row 기반 30-window, 당일 제외 ──────────────────
def baseline_dev(emb, art):
    parts = []
    for country, g in emb.groupby("country", sort=False):
        g = g.sort_values("date"); X = g[art["emb_cols"]].values.astype(np.float32); n = len(X)
        cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)])
        l2 = np.zeros(n); cosd = np.zeros(n)
        for i in range(n):
            lo = max(0, i - 30); cnt = i - lo
            if cnt < 5:
                continue
            base = (cs[i] - cs[lo]) / cnt
            diff = X[i] - base
            l2[i] = np.sqrt(float((diff * diff).sum()))
            nb, nx = float(np.linalg.norm(base)), float(np.linalg.norm(X[i]))
            if nb > 0 and nx > 0:
                cosd[i] = 1.0 - float(X[i] @ base) / (nb * nx)
        s = pd.Series(l2)
        parts.append(pd.DataFrame({"country": country, "date": g["date"].values,
            "gkg_bdev_l2": l2, "gkg_bdev_cos": cosd,
            "gkg_bdev_l2_7dmean": s.rolling(7, min_periods=1).mean().values,
            "gkg_bdev_accel": (s.rolling(7, min_periods=1).mean()
                               / (s.rolling(30, min_periods=1).mean() + EPS)).values}))
    return pd.concat(parts, ignore_index=True)


# ── aux (07): delta/cosdiss (row-shift) + ntitles + kmeans cluster ────
def aux(emb, art):
    ec = art["emb_cols"]
    parts = []
    for country, g in emb.groupby("country", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        X = g[ec].values.astype(np.float32); n = len(g)
        rec = {"country": country, "date": g["date"].values}
        for k in (3, 7):
            l2 = np.zeros(n); cosd = np.zeros(n)
            if n > k:
                diff = X[k:] - X[:-k]
                l2[k:] = np.linalg.norm(diff, axis=1)
                a, b = X[k:], X[:-k]
                denom = np.clip(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12, None)
                cosd[k:] = 1.0 - (a * b).sum(axis=1) / denom
            rec[f"gkg_emb_delta{k}_norm"] = l2
            rec[f"gkg_emb_cosdiss_{k}"] = cosd
        nt = g["gkg_emb_n_titles_1d"].values.astype(np.float32)
        for k in (3, 7):
            d = np.zeros(n)
            if n > k:
                d[k:] = nt[k:] - nt[:-k]
            rec[f"gkg_ntitles_delta{k}"] = d
        rec["gkg_ntitles_max7d"] = pd.Series(nt).rolling(7, min_periods=1).max().values
        rmean = pd.Series(nt).rolling(30, min_periods=5).mean().values
        rstd = pd.Series(nt).rolling(30, min_periods=5).std().fillna(1.0).values
        rstd = np.clip(rstd, 1e-3, None)
        z = np.nan_to_num((nt - rmean) / rstd, nan=0.0, posinf=0.0, neginf=0.0)
        rec["gkg_ntitles_z30d"] = z
        parts.append(pd.DataFrame(rec))
    cluster = _cluster(emb, art)
    return pd.concat(parts, ignore_index=True).merge(cluster, on=["country", "date"], how="left")


def _cluster(emb, art):
    """pca64 transform → kmeans.transform → 30 centroid 거리 + min/mean (07 B4)."""
    pca64 = art["pca64"].transform(emb[art["emb_cols"]].values).astype(np.float32)
    dist = art["kmeans"].transform(pca64).astype(np.float32)  # (N,30)
    out = emb[["country", "date"]].copy()
    for i in range(dist.shape[1]):
        out[f"gkg_cluster_dist_{i}"] = dist[:, i]
    out["gkg_cluster_dist_min"] = dist.min(axis=1)
    out["gkg_cluster_dist_mean"] = dist.mean(axis=1)
    return out


# ── anchor temporal (11) + risk (13): daily-grid 0-fill ───────────────
def _daily_grid_roll(series_by_country, build_cols):
    parts = []
    for country, g in series_by_country:
        g = g.sort_values("date").set_index(pd.to_datetime(g.sort_values("date")["date"]))
        full = pd.date_range(g.index.min(), g.index.max(), freq="D")
        gg = g.reindex(full, fill_value=0.0)
        rec = build_cols(gg)
        rec["country"] = country
        rec = rec.reset_index().rename(columns={"index": "date"})
        parts.append(rec)
    return pd.concat(parts, ignore_index=True)


def anchor_temporal(anchor_cos, labels):
    cols = [f"gkg_anchor_cos_{a}" for a in labels]
    def build(gg):
        rec = pd.DataFrame(index=gg.index)
        for a in labels:
            s = gg[f"gkg_anchor_cos_{a}"]
            r7m, r30m = s.rolling(7, 1).mean(), s.rolling(30, 1).mean()
            rec[f"gkg_anchT_{a}_r7m"] = r7m
            rec[f"gkg_anchT_{a}_r7s"] = s.rolling(7, 1).std().fillna(0)
            rec[f"gkg_anchT_{a}_r30m"] = r30m
            rec[f"gkg_anchT_{a}_r30s"] = s.rolling(30, 1).std().fillna(0)
            rec[f"gkg_anchT_{a}_accel"] = r7m / (r30m + EPS)
        return rec
    g = anchor_cos[["country", "date"] + cols]
    return _daily_grid_roll(g.groupby("country", sort=False), build)


def risk(anchor_cos):
    rc = [f"gkg_anchor_cos_{a}" for a in RISK5]
    tmp = anchor_cos[["country", "date"]].copy()
    tmp["__risk"] = anchor_cos[rc].mean(axis=1)
    def build(gg):
        r = gg["__risk"]; m7, m30 = r.rolling(7, 1).mean(), r.rolling(30, 1).mean()
        return pd.DataFrame({"gkg_risk_1d": r, "gkg_risk_7dmean": m7,
                             "gkg_risk_7dmax": r.rolling(7, 1).max(),
                             "gkg_risk_30dmean": m30, "gkg_risk_accel": m7 / (m30 + EPS)},
                            index=gg.index)
    return _daily_grid_roll(tmp.groupby("country", sort=False), build)


def event_count(anchor_cos, art):
    """gkg_evt_*(48): 국가×앵커 train p90 초과 카운트. 복제: 12_anchor_eventcount.py.
    daily-grid 0-fill 후 (cos>thr) rolling sum 1d/7d/30d. [VERIFIED]"""
    labels, thr = art["anchor_labels"], art["evt_thr"]
    cols = [f"gkg_anchor_cos_{a}" for a in labels]
    parts = []
    for country, g in anchor_cos.groupby("country", sort=False):
        g = g.sort_values("date").set_index(pd.to_datetime(g.sort_values("date")["date"]))
        gg = g[cols].reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"), fill_value=0.0)
        rec = pd.DataFrame(index=gg.index)
        for a in labels:
            c = f"gkg_anchor_cos_{a}"
            t = float(thr.loc[country, c]) if country in thr.index else 0.0
            ev = (gg[c] > t).astype(float)
            rec[f"gkg_evt_{a}_1d"] = ev
            rec[f"gkg_evt_{a}_7d"] = ev.rolling(7, min_periods=1).sum()
            rec[f"gkg_evt_{a}_30d"] = ev.rolling(30, min_periods=1).sum()
        rec["country"] = country
        parts.append(rec.reset_index().rename(columns={"index": "date"}))
    return pd.concat(parts, ignore_index=True)


def embedding_derived(emb, art):
    """country-day 평균 임베딩 → 재현 가능한 214개 임베딩 파생 피처 전체.
    (gkg_pool_ 87개는 제목단위 BQ 필요 → pool_features.py 별도.)"""
    emb = emb.sort_values(["country", "date"]).reset_index(drop=True)
    pw = pointwise(emb, art)
    ac = pw[["country", "date"] + [f"gkg_anchor_cos_{a}" for a in art["anchor_labels"]]]
    # anchT/evt/risk 는 daily-grid(뉴스 없는 날도 rolling 값 존재) → outer 조인으로 그 날들 보존,
    # pca/anchor/aux/preesc/bdev(제목 있는 날만) 는 누락일 0 fill (학습 build_dataset 의 fillna(0) 와 동일).
    out = pw
    for part in (preesc(emb, art), baseline_dev(emb, art), aux(emb, art),
                 anchor_temporal(ac, art["anchor_labels"]), risk(ac), event_count(ac, art)):
        out = out.merge(part, on=["country", "date"], how="outer")
    # gkg_emb_missing_mask = 1 (제목/임베딩 없는 날). n_titles 결측 여부로 판정 (build_dataset 동일)
    out["gkg_emb_missing_mask"] = out["gkg_emb_n_titles_1d"].isna().astype("int64")
    feat = [c for c in out.columns if c not in ("country", "date", "gkg_emb_missing_mask")]
    out[feat] = out[feat].fillna(0.0)
    return out.sort_values(["country", "date"]).reset_index(drop=True)


# ── parity 자체검증 ───────────────────────────────────────────────────
def parity_check(country="SDN", root: str | Path = "."):
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    root = Path(root); art = load_artifacts(root)
    emb = ds.dataset(root / "input/processed/features/gkg_embeddings.parquet").to_table(
        filter=(pc.field("country") == country)).to_pandas()
    emb["date"] = pd.to_datetime(emb["date"]).dt.tz_localize(None).dt.normalize()
    rep = embedding_derived(emb, art)
    cols = [c for c in rep.columns if c.startswith((
        "gkg_emb_pca_", "gkg_anchor_cos_", "gkg_preesc_", "gkg_anchT_",
        "gkg_risk_", "gkg_bdev_", "gkg_emb_delta", "gkg_emb_cosdiss", "gkg_ntitles_",
        "gkg_cluster_", "gkg_evt_"))]
    full = ds.dataset(root / "input/processed/dataset/full_pca16_aclfree.parquet").to_table(
        filter=(pc.field("country") == country), columns=["date"] + cols).to_pandas()
    full["date"] = pd.to_datetime(full["date"]).dt.tz_localize(None).dt.normalize()
    m = rep.merge(full, on="date", suffixes=("_rep", "_str"))
    print(f"[{country}] 병합 {len(m)}행, 검증 {len(cols)}개 피처")
    bad = []
    for c in cols:
        d = np.nanmax(np.abs(m[f"{c}_rep"].values - m[f"{c}_str"].values))
        if d >= 1e-4:
            bad.append((c, d))
    if bad:
        print(f"  ❌ MISMATCH {len(bad)}개:")
        for c, d in sorted(bad, key=lambda x: -x[1])[:12]:
            print(f"     {c}: max|Δ|={d:.2e}")
    else:
        print(f"  ✅ PARITY OK — 전 {len(cols)}개 피처 max|Δ| < 1e-4")


if __name__ == "__main__":
    parity_check()
