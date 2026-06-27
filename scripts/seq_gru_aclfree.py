"""ACLED-free 직교 시퀀스 모델 (GRU) — 0.10 돌파 push.

트리(cross-sectional)가 못 보는 escalation의 시계열 궤적을 포착하는 직교 base learner.
- 입력: 국가별 30일 윈도우 (lean+EMB 피처, 모두 ACLED-free·실시간 가능)
- 구조: GRU(hidden) + country embedding + multi-task heads
        (escalation 주 + onset/event_count 보조). last hidden 사용 (attention 금지 = known pitfall)
- 폴드: cv_harness 와 동일 (F2 2024H1, F3 2024H2). train <= fold start.
- 출력: 폴드별 val OOF 예측 → output/aclfree_010/gru_oof.parquet (스태킹 소비용)

주의: torch 와 lightgbm 동시 import 금지(macOS libomp segfault). 이 스크립트는 torch 만.
사용법: python scripts/seq_gru_aclfree.py --epochs 15 --hidden 96
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

DATA = "input/processed/dataset/full_pca16_aclfree.parquet"
OUT = "output/aclfree_010/gru_oof.parquet"
TARGET = "y_escalation"
WINDOW = 30  # main()에서 args.window로 덮어씀
LABEL_META = ["y", "y_onset", "y_escalation", "fatalities_next3d", "event_count_next3d",
              "past14d_event_count", "past14d_fatalities_mean"]
FOLDS = [  # (name, train_end, val_start, val_end)
    ("F2", date(2023, 12, 31), date(2024, 1, 1), date(2024, 6, 30)),
    ("F3", date(2024, 6, 30), date(2024, 7, 1), date(2024, 12, 31)),
]

PREFIX_KEEP = ("gkg_pool_", "gkg_preesc_", "gkg_bdev_", "gdelt2_", "gdelt_sub_",
               "gdev_", "gkg_emb_pca_")  # lean + EMB(PCA16)


def select_features(df, mode="lean"):
    drop = set(["date", "country", "region"] + LABEL_META)
    if mode == "all":
        # 임베딩 파생 포함 전체 (시퀀스 모델은 temporal 로 활용 가능 → 트리와 달리 도움될 수 있음)
        return [c for c in df.columns if c not in drop and df[c].dtype != object]
    excl = ("gkg_anchT_", "gkg_evt_", "gkg_risk_", "gkg_anchor_cos_",
            "gkg_emb_delta", "gkg_emb_cosdiss", "gkg_ntitles_", "gkg_cluster_",
            "gkg_emb_n_titles", "gkg_emb_missing_mask", "gcr_")
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if c.startswith(excl):
            continue
        if c.startswith(PREFIX_KEEP) or c.startswith("gkg_") or c.startswith("page_title") \
           or c.startswith("econ_") or c.startswith("gdelt_") or c.startswith("hotspot") \
           or c.startswith("region_") or c.startswith("month_") or c.startswith("dow_"):
            cols.append(c)
    return cols


class SeqDS(Dataset):
    def __init__(self, arrs, cidx, samples, yesc, yons, ycnt):
        self.arrs = arrs; self.cidx = cidx; self.samples = samples
        self.yesc = yesc; self.yons = yons; self.ycnt = ycnt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        c, pos = self.samples[i]
        x = self.arrs[c][pos - WINDOW + 1:pos + 1]      # (WINDOW, F)
        return (torch.from_numpy(x), self.cidx[c],
                self.yesc[i], self.yons[i], self.ycnt[i])


class TCN(nn.Module):
    """dilated causal conv stack → 마지막 시점 표현. GRU 와 다른 귀납 편향."""
    def __init__(self, n_feat, hidden, layers=4, k=3, drop=0.3):
        super().__init__()
        chans = [n_feat] + [hidden] * layers
        blocks = []
        for i in range(layers):
            d = 2 ** i
            blocks += [nn.Conv1d(chans[i], chans[i + 1], k, padding=(k - 1) * d, dilation=d),
                       nn.ReLU(), nn.Dropout(drop)]
        self.net = nn.Sequential(*blocks)
        self.k, self.layers = k, layers
        self.out_dim = hidden

    def forward(self, x):                 # x: (B,T,F)
        T = x.size(1)
        h = self.net(x.transpose(1, 2))   # (B,H,T+pad)
        return h[:, :, T - 1]             # 마지막 유효 시점 (causal)


class GRUNet(nn.Module):
    def __init__(self, n_feat, n_country, hidden=96, emb=8, drop=0.3, layers=1, arch="gru"):
        super().__init__()
        self.arch = arch
        bi = arch.startswith("bi")          # bigru / bilstm = 양방향 (윈도우 전체가 과거라 누수 없음)
        core = arch[2:] if bi else arch     # bigru->gru, bilstm->lstm
        self.bi, self.core = bi, core
        self.cemb = nn.Embedding(n_country, emb)
        rnn_out = hidden * (2 if bi else 1)
        if core == "gru":
            self.rnn = nn.GRU(n_feat, hidden, num_layers=layers, batch_first=True,
                              dropout=drop if layers > 1 else 0.0, bidirectional=bi)
        elif core == "lstm":
            self.rnn = nn.LSTM(n_feat, hidden, num_layers=layers, batch_first=True,
                               dropout=drop if layers > 1 else 0.0, bidirectional=bi)
        elif core == "tcn":
            self.rnn = TCN(n_feat, hidden, layers=max(layers, 4), drop=drop)
            rnn_out = hidden
        self.drop = nn.Dropout(drop)
        h = rnn_out + emb
        self.head_esc = nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 1))
        self.head_ons = nn.Linear(h, 1)
        self.head_cnt = nn.Linear(h, 1)

    def forward(self, x, c):
        if self.core == "gru":
            _, hn = self.rnn(x)
            last = torch.cat([hn[-2], hn[-1]], dim=1) if self.bi else hn[-1]
        elif self.core == "lstm":
            _, (hn, _) = self.rnn(x)
            last = torch.cat([hn[-2], hn[-1]], dim=1) if self.bi else hn[-1]
        else:
            last = self.rnn(x)
        z = torch.cat([self.drop(last), self.cemb(c)], dim=1)
        return self.head_esc(z).squeeze(1), self.head_ons(z).squeeze(1), self.head_cnt(z).squeeze(1)


def focal_bce(logits, target, pos_weight, gamma=2.0):
    p = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, p, 1 - p)
    w = torch.where(target > 0.5, pos_weight, torch.ones_like(p))
    return (w * (1 - pt).pow(gamma) * nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none")).mean()


def build_country_arrays(df, cols, feat_mean, feat_std):
    arrs, positions = {}, {}
    X = ((df[cols].values.astype(np.float32) - feat_mean) / feat_std).astype(np.float32)
    df = df.reset_index(drop=True)
    for c, grp in df.groupby("country"):
        idx = grp.index.values
        arrs[c] = X[idx]
        positions[c] = grp["date"].values
    return arrs


def make_samples(df, arrs, mask):
    """mask: boolean over df rows selecting which prediction-days to include.
    returns list of (country, pos_in_country_array) with full WINDOW available."""
    df = df.reset_index(drop=True)
    out, rowids = [], []
    for c, grp in df.groupby("country"):
        idx = grp.index.values
        local_mask = mask[idx]
        for local_pos in range(WINDOW - 1, len(idx)):
            if local_mask[local_pos]:
                out.append((c, local_pos)); rowids.append(idx[local_pos])
    return out, np.array(rowids)


def run_fold(df, cols, name, tr_end, va_s, va_e, device, args):
    d = df["date"].dt.date
    tr_mask = (d <= tr_end).values
    va_mask = ((d >= va_s) & (d <= va_e)).values
    # scaler on train rows only
    Xtr = df.loc[tr_mask, cols].values.astype(np.float32)
    mu = np.nanmean(Xtr, axis=0); sd = np.nanstd(Xtr, axis=0); sd[sd == 0] = 1.0
    arrs = build_country_arrays(df, cols, mu, sd)
    countries = sorted(df["country"].unique())
    cidx = {c: i for i, c in enumerate(countries)}

    tr_s, tr_rows = make_samples(df, arrs, tr_mask)
    va_s2, va_rows = make_samples(df, arrs, va_mask)

    yesc = df[TARGET].values.astype(np.float32)
    yons = df["y_onset"].values.astype(np.float32)
    ycnt = np.log1p(df["event_count_next3d"].clip(lower=0).fillna(0).values).astype(np.float32)
    ymu, ysd = ycnt[tr_mask].mean(), ycnt[tr_mask].std() + 1e-6
    ycnt = (ycnt - ymu) / ysd

    def ds(samples, rows):
        return SeqDS(arrs, {c: torch.tensor(i) for c, i in cidx.items()}, samples,
                     yesc[rows], yons[rows], ycnt[rows])

    tr_ds, va_ds = ds(tr_s, tr_rows), ds(va_s2, va_rows)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    va_dl = DataLoader(va_ds, batch_size=1024, shuffle=False)

    pos = yesc[tr_rows].mean(); pw = min((1 - pos) / max(pos, 1e-6), 30.0)
    posn = yons[tr_rows].mean(); pwn = min((1 - posn) / max(posn, 1e-6), 150.0)

    seed_preds = []
    for seed in range(args.n_seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        net = GRUNet(len(cols), len(countries), hidden=args.hidden, layers=args.layers,
                     arch=args.arch).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
        pw_t = torch.tensor(pw, device=device)
        bcen = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pwn, device=device))
        mse = nn.MSELoss()

        def esc_loss(pe, ye):
            if args.focal:
                return focal_bce(pe, ye, pw_t, gamma=2.0)
            return nn.functional.binary_cross_entropy_with_logits(
                pe, ye, pos_weight=pw_t)

        best_ap, best_p, best_ep = -1.0, None, -1
        for ep in range(args.epochs):
            net.train()
            for xb, cb, ye, yo, yc in tr_dl:
                xb = xb.to(device); cb = cb.to(device)
                ye = ye.to(device); yo = yo.to(device); yc = yc.to(device)
                opt.zero_grad()
                pe, po, pc = net(xb, cb)
                loss = esc_loss(pe, ye) + 0.3 * bcen(po, yo) + 0.2 * mse(pc, yc)
                loss.backward(); opt.step()
            net.eval(); preds = []
            with torch.no_grad():
                for xb, cb, ye, yo, yc in va_dl:
                    pe, _, _ = net(xb.to(device), cb.to(device))
                    preds.append(torch.sigmoid(pe).cpu().numpy())
            p = np.concatenate(preds)
            ap = average_precision_score(yesc[va_rows], p)
            if ap > best_ap:
                best_ap, best_p, best_ep = ap, p, ep
            elif ep - best_ep >= args.patience:
                break
        print(f"  [{name}] seed{seed} BEST ep{best_ep} PR-AUC={best_ap:.4f}", flush=True)
        seed_preds.append(best_p)
    p_ens = np.mean(seed_preds, axis=0)
    ap_ens = average_precision_score(yesc[va_rows], p_ens)
    print(f"  [{name}] ENSEMBLE({args.n_seeds}) PR-AUC={ap_ens:.4f}", flush=True)
    res = pd.DataFrame({"country": df.loc[va_rows, "country"].values,
                        "date": df.loc[va_rows, "date"].values,
                        "fold": name, "y": yesc[va_rows], "p_gru": p_ens})
    return ap_ens, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--n_seeds", type=int, default=1)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--focal", action="store_true")
    ap.add_argument("--feats", choices=["lean", "all"], default="lean")
    ap.add_argument("--arch", choices=["gru", "lstm", "tcn", "bigru", "bilstm"], default="gru")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    global WINDOW
    WINDOW = args.window

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| window:", WINDOW, "| layers:", args.layers,
          "| focal:", args.focal, "| feats:", args.feats)
    df = pd.read_parquet(args.data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    cols = select_features(df, args.feats)
    df[cols] = df[cols].fillna(0.0)
    print(f"피처 {len(cols)}개, 행 {len(df)}, 국가 {df['country'].nunique()}")

    all_res, aps = [], []
    for name, te, vs, ve in FOLDS:
        ap_, res = run_fold(df, cols, name, te, vs, ve, device, args)
        aps.append(ap_); all_res.append(res)
    print(f"\nGRU clean CV PR-AUC (mean F2,F3) = {np.mean(aps):.4f}")
    pd.concat(all_res, ignore_index=True).to_parquet(args.out, index=False)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
