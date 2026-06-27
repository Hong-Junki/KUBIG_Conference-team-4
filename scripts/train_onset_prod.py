"""onset 프로덕션 모델 학습 — 배포용 가중치 저장 (GPU 1회 실행용).

기존 seq_gru_aclfree.py 는 폴드별 학습→OOF 만 저장하고 가중치를 버린다.
이 스크립트는 배포(서빙)를 위해 각 base 를 **전체 train 으로 학습 후 가중치를 저장**한다.

산출 (output/models/onset_prod/):
  trees_lgbm.pkl, trees_xgb.pkl                     트리 base (전체학습)
  seq_<arch>_w<window>_<feats>.pt                   시퀀스 base (state_dict + scaler + 설정)
  meta.pkl                                          스태킹 메타(로지스틱) + onset 선택 base 목록
  manifest.json                                     base 목록·피처리스트·스케일러·calm 게이팅 명세

학습 컷오프: PROD_END (기본 2024-12-31, 라벨 완전 구간 전체). test 2025Q1 미사용.
메타는 기존 OOF(output/aclfree_010/*_oof.parquet)에서 onset(calm) 기준으로 fit.

사용법(GPU/CUDA 자동 감지):
  python scripts/train_onset_prod.py                       # 전체
  python scripts/train_onset_prod.py --only seq            # 시퀀스만
  python scripts/train_onset_prod.py --skip chronos        # chronos 생략(기본)
"""
from __future__ import annotations
import argparse, json, pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

DATA = "input/processed/dataset/full_pca16_aclfree.parquet"
OUTDIR = Path("output/models/onset_prod")
OOF_DIR = Path("output/aclfree_010")
PROD_END = date(2024, 12, 31)          # 배포 모델 학습 컷오프 (라벨 완전 구간)
TARGET = "y_escalation"                 # onset = escalation 학습 + calm 평가(서빙 게이팅)

# onset 프로덕션 시퀀스 base 구성 (양방향이 onset 핵심)
SEQ_BASES = [
    dict(arch="gru",    window=30, feats="lean", hidden=96, layers=1, focal=False, seeds=3),
    dict(arch="lstm",   window=45, feats="all",  hidden=96, layers=2, focal=False, seeds=3),
    dict(arch="tcn",    window=45, feats="lean", hidden=96, layers=4, focal=False, seeds=3),
    dict(arch="bigru",  window=45, feats="all",  hidden=96, layers=1, focal=False, seeds=3),
    dict(arch="bilstm", window=30, feats="lean", hidden=96, layers=1, focal=False, seeds=3),
]

LGBM_PARAMS = dict(objective="binary", n_estimators=459, learning_rate=0.0136, num_leaves=43,
                   min_child_samples=91, scale_pos_weight=22.0899, subsample=0.909,
                   colsample_bytree=0.8063, reg_lambda=1.3014, reg_alpha=2.7105)


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ── 트리 base 전체학습 ────────────────────────────────────────────────
def train_trees(df, cols_meta):
    import lightgbm as lgb
    try:
        import xgboost as xgb; has_xgb = True
    except Exception:
        has_xgb = False
    y = df[TARGET].values
    d = df["date"].dt.date
    tr = (d <= PROD_END).values
    cols = cols_meta["tree_cols"]
    Xtr, ytr = df.loc[tr, cols].values, y[tr]
    ml = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=42, n_jobs=-1, verbose=-1).fit(Xtr, ytr)
    pickle.dump({"model": ml, "cols": cols}, open(OUTDIR / "trees_lgbm.pkl", "wb"))
    print(f"  [trees] LGBM 저장 (n_train={tr.sum()}, feats={len(cols)})", flush=True)
    if has_xgb:
        pos = ytr.mean()
        mx = xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=5, subsample=0.8,
                               colsample_bytree=0.6, reg_lambda=2.0, min_child_weight=5,
                               scale_pos_weight=(1 - pos) / max(pos, 1e-6), eval_metric="aucpr",
                               tree_method="hist", n_jobs=-1, random_state=42).fit(Xtr, ytr)
        pickle.dump({"model": mx, "cols": cols}, open(OUTDIR / "trees_xgb.pkl", "wb"))
        print(f"  [trees] XGB 저장", flush=True)


# ── 시퀀스 base 전체학습 + 가중치 저장 ───────────────────────────────
def train_seq(df, cfg, device):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader
    import scripts.seq_gru_aclfree as S
    S.WINDOW = cfg["window"]; S.TARGET = TARGET
    cols = S.select_features(df, cfg["feats"])
    df = df.copy(); df[cols] = df[cols].fillna(0.0)
    d = df["date"].dt.date
    tr_mask = (d <= PROD_END).values
    Xtr = df.loc[tr_mask, cols].values.astype(np.float32)
    mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[sd == 0] = 1.0
    arrs = S.build_country_arrays(df, cols, mu, sd, diff=False)
    n_feat = len(cols)
    countries = sorted(df["country"].unique()); cidx = {c: i for i, c in enumerate(countries)}
    tr_s, tr_rows = S.make_samples(df, arrs, tr_mask)
    yesc = df[TARGET].values.astype(np.float32)
    yons = df["y_onset"].values.astype(np.float32)
    ycnt = np.log1p(df["event_count_next3d"].clip(lower=0).fillna(0).values).astype(np.float32)
    ymu, ysd = ycnt[tr_mask].mean(), ycnt[tr_mask].std() + 1e-6
    ycnt = (ycnt - ymu) / ysd
    ds = S.SeqDS(arrs, {c: torch.tensor(i) for c, i in cidx.items()}, tr_s,
                 yesc[tr_rows], yons[tr_rows], ycnt[tr_rows])
    dl = DataLoader(ds, batch_size=512, shuffle=True)
    pos = yesc[tr_rows].mean(); pw = min((1 - pos) / max(pos, 1e-6), 30.0)
    posn = yons[tr_rows].mean(); pwn = min((1 - posn) / max(posn, 1e-6), 150.0)

    seeds_state = []
    for seed in range(cfg["seeds"]):
        torch.manual_seed(seed); np.random.seed(seed)
        if cfg["arch"] == "itransformer":
            net = S.ITransformerNet(n_feat, S.WINDOW, len(countries), d_model=cfg["hidden"],
                                    layers=max(cfg["layers"], 2)).to(device)
        else:
            net = S.GRUNet(n_feat, len(countries), hidden=cfg["hidden"], layers=cfg["layers"],
                           arch=cfg["arch"]).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        pw_t = torch.tensor(pw, device=device)
        bcen = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pwn, device=device))
        mse = nn.MSELoss()
        for ep in range(15):
            net.train()
            for xb, cb, ye, yo, yc in dl:
                xb, cb = xb.to(device), cb.to(device)
                ye, yo, yc = ye.to(device), yo.to(device), yc.to(device)
                opt.zero_grad()
                pe, po, pc = net(xb, cb)
                loss = (nn.functional.binary_cross_entropy_with_logits(pe, ye, pos_weight=pw_t)
                        + 0.3 * bcen(po, yo) + 0.2 * mse(pc, yc))
                loss.backward(); opt.step()
        seeds_state.append({k: v.cpu() for k, v in net.state_dict().items()})
        print(f"  [seq:{cfg['arch']}_w{cfg['window']}_{cfg['feats']}] seed{seed} 학습완료", flush=True)

    name = f"seq_{cfg['arch']}_w{cfg['window']}_{cfg['feats']}"
    torch.save({"seeds_state": seeds_state, "cfg": cfg, "cols": cols, "mu": mu, "sd": sd,
                "countries": countries, "n_feat": n_feat}, OUTDIR / f"{name}.pt")
    print(f"  ✓ 저장 {name}.pt (seeds={cfg['seeds']}, feats={len(cols)})", flush=True)


# ── 스태킹 메타: 기존 OOF에서 onset(calm) 기준 fit ───────────────────
def fit_meta(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    import glob, os
    # OOF 병합
    base = None; cols = []
    for path in sorted(glob.glob(str(OOF_DIR / "*_oof.parquet"))):
        stem = os.path.basename(path)[:-len("_oof.parquet")]
        d = pd.read_parquet(path); d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
        for pc in [c for c in d.columns if c.startswith("p_")]:
            nm = (stem if d.filter(like="p_").shape[1] == 1 else f"{stem}_{pc[2:]}").upper()
            dd = d[["country", "date", "fold", "y", pc]].rename(columns={pc: nm})
            base = dd if base is None else base.merge(dd[["country", "date", nm]], on=["country", "date"], how="inner")
            cols.append(nm)
    meta_info = pd.read_parquet(DATA, columns=["country", "date", "past14d_event_count"])
    meta_info["date"] = pd.to_datetime(meta_info["date"]).dt.tz_localize(None).dt.normalize()
    base = base.merge(meta_info, on=["country", "date"], how="left")
    calm = (base["past14d_event_count"].fillna(0) == 0).values
    y = base["y"].values

    def logit(p): return np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    # onset(calm) 기준 greedy forward selection (fold 분리 honest)
    def ap_on(feats, sf, ef):
        tr = base["fold"].values == sf; te = base["fold"].values == ef
        Xtr, Xte = logit(base.loc[tr, feats].values), logit(base.loc[te, feats].values)
        m, s = Xtr.mean(0), Xtr.std(0) + 1e-9
        lr = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit((Xtr - m) / s, y[tr])
        p = np.full(len(base), np.nan); p[te] = lr.predict_proba((Xte - m) / s)[:, 1]
        mte = te & calm
        return average_precision_score(y[mte], p[mte])
    sel, best, rem = [], -1, list(cols)
    while rem:
        cb, cand = -1, None
        for c in rem:
            trial = sel + [c]
            ap = np.mean([ap_on(trial, "F2", "F3"), ap_on(trial, "F3", "F2")]) if len(trial) > 1 \
                else np.mean([average_precision_score(y[(base.fold == f).values & calm],
                              base[c].values[(base.fold == f).values & calm]) for f in ["F2", "F3"]])
            if ap > cb: cb, cand = ap, c
        if cb > best + 1e-4: sel.append(cand); rem.remove(cand); best = cb
        else: break
    # 선택셋으로 전체 OOF에 메타 fit (배포용)
    X = logit(base[sel].values); m, s = X.mean(0), X.std(0) + 1e-9
    lr = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit((X - m) / s, y)
    pickle.dump({"meta": lr, "selected": sel, "mu": m, "sd": s, "onset_cv": best},
                open(OUTDIR / "meta.pkl", "wb"))
    print(f"  ✓ meta 저장 — onset 선택셋={sel}, honest onset CV={best:.4f}")
    return sel, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["trees", "seq", "meta"], default=None)
    ap.add_argument("--skip-chronos", action="store_true", default=True)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    print(f"DATA {df.shape}, 학습 컷오프 <= {PROD_END}, device={_device()}")

    # 트리 피처 선택 (base_trees_oof 와 동일 lean+A+C+GDEV)
    LABEL_META = ["y", "y_onset", "y_escalation", "fatalities_next3d", "event_count_next3d",
                  "past14d_event_count", "past14d_fatalities_mean"]
    PREFIX = {"POOL": ("gkg_pool_",), "PRE": ("gkg_preesc_",), "BDEV": ("gkg_bdev_",),
              "GC2": ("gdelt2_",), "GSUB": ("gdelt_sub_",), "GDEV": ("gdev_",),
              "ANC": ("gkg_anchor_cos_",), "EMB": ("gkg_emb_pca_",), "TEMP": ("gkg_anchT_",),
              "EVT": ("gkg_evt_",), "RISK": ("gkg_risk_",)}
    KEEP = {"CORE", "T1", "POOL", "PRE", "BDEV", "GC2", "GSUB", "GDEV"}
    def assign(c):
        for b, p in PREFIX.items():
            if c.startswith(p): return b
        return "T1" if (c.startswith("gkg_") or c.startswith("page_title")) else "CORE"
    tree_cols = [c for c in df.columns if c not in (["date", "country"] + LABEL_META) and assign(c) in KEEP]
    cols_meta = {"tree_cols": tree_cols}

    if args.only in (None, "trees"):
        print("[1] 트리 base 전체학습"); train_trees(df, cols_meta)
    if args.only in (None, "seq"):
        print("[2] 시퀀스 base 전체학습 + 가중치 저장")
        dev = _device()
        for cfg in SEQ_BASES:
            train_seq(df, cfg, dev)
    if args.only in (None, "meta"):
        print("[3] 스태킹 메타 fit (onset calm 기준)"); fit_meta(df)

    # manifest
    man = {"target": TARGET, "prod_train_end": str(PROD_END),
           "onset_gating": "calm = past14d_event_count == 0",
           "seq_bases": [f"seq_{c['arch']}_w{c['window']}_{c['feats']}" for c in SEQ_BASES],
           "tree_bases": ["trees_lgbm", "trees_xgb"], "tree_cols": tree_cols}
    json.dump(man, open(OUTDIR / "manifest.json", "w"), ensure_ascii=False, indent=2)
    print(f"\n완료 → {OUTDIR}/ (가중치+meta+manifest). 이 폴더를 서빙 위치(output/models/onset_prod)로 배치.")


if __name__ == "__main__":
    main()
