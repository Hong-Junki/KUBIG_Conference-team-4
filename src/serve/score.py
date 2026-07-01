"""onset 스코어러 — model_input(피처) → onset 점수.

프로덕션 메타 선택셋 = ['LSTM_W45_ALL'] 단독이므로, 서빙 점수는
seq_lstm_w45_all.pt(escalation-head, 3시드 sigmoid 평균) → meta(1입력 로지스틱) → onset_prob.
calm 국가에만 onset 경보 의미 부여(게이팅). 실시간엔 ACLED가 stale → calm 판정은
GDELT 기반(calm_gdelt.pkl: 드리프트-무관 비율/tone 로지스틱, val AUC 0.727)을 우선 사용,
아티팩트/피처 없으면 ACLED past14d_event_count==0 fallback (known-pitfalls: ACLED stale).

추론 CPU 충분. torch + sklearn 만 사용(트리=lightgbm 불필요 → macOS libomp 충돌 회피).
출력: country, date, base_pred, onset_prob, calm_flag.
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = "output/models/onset_prod"   # 표준 서빙 위치 (학습본 배치)


def load_model(model_dir: str | Path = MODEL_DIR):
    import torch
    md = Path(model_dir)
    meta = pickle.load(open(md / "meta.pkl", "rb"))
    sel = meta["selected"]
    assert sel == ["LSTM_W45_ALL"], f"예상과 다른 선택셋: {sel} (스코어러는 LSTM 단독 가정)"
    ckpt = torch.load(md / "seq_lstm_w45_all.pt", map_location="cpu", weights_only=False)
    return {"meta": meta, "ckpt": ckpt}


def _predict_lstm(df: pd.DataFrame, ckpt: dict, target_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LSTM escalation-head 3시드 sigmoid 평균. target_mask True 행만 예측.
    return (rowids, p_lstm). seq_gru_aclfree 의 GRUNet/윈도우 로직 재사용."""
    import torch
    import scripts.seq_gru_aclfree as S
    cfg, cols, mu, sd = ckpt["cfg"], ckpt["cols"], ckpt["mu"], ckpt["sd"]
    countries, n_feat = ckpt["countries"], ckpt["n_feat"]
    S.WINDOW = cfg["window"]
    cidx = {c: i for i, c in enumerate(countries)}
    df = df.copy()
    df[cols] = df[cols].fillna(0.0)
    arrs = S.build_country_arrays(df, cols, mu, sd)   # 국가별 정규화 윈도우 (lstm=diff 미사용)
    samples, rows = S.make_samples(df, arrs, target_mask)         # 풀윈도우 가능한 (country,pos)
    if not samples:
        return np.array([], dtype=int), np.array([])
    # 배치 텐서 구성
    W = cfg["window"]
    X = np.stack([arrs[c][p - W + 1:p + 1] for c, p in samples]).astype(np.float32)
    C = np.array([cidx.get(c, 0) for c, _ in samples], dtype=np.int64)
    xb, cb = torch.from_numpy(X), torch.from_numpy(C)
    preds = []
    for state in ckpt["seeds_state"]:
        net = S.GRUNet(n_feat, len(countries), hidden=cfg["hidden"],
                       layers=cfg["layers"], arch=cfg["arch"])
        net.load_state_dict(state); net.eval()
        with torch.no_grad():
            pe, _, _ = net(xb, cb)
            preds.append(torch.sigmoid(pe).numpy())
    return rows, np.mean(preds, axis=0)


def _calm_flag(rows_df: pd.DataFrame, cols, model_dir: str | Path) -> tuple[np.ndarray, str]:
    """calm_flag 산출. 우선순위: ① GDELT 로지스틱(calm_gdelt.pkl, 실시간 가용·드리프트 무관)
    ② ACLED past14d_event_count==0 (학습/과거 채점용, 실시간엔 stale) ③ 둘 다 없으면 1.
    return (flag[int], source)."""
    p = Path(model_dir) / "calm_gdelt.pkl"
    if p.exists():
        art = pickle.load(open(p, "rb"))
        if set(art["feats"]).issubset(set(cols)):
            X = rows_df[art["feats"]].fillna(0.0).values.astype(float)
            z = (X - np.asarray(art["mu"], float)) / np.asarray(art["sd"], float)
            pcalm = 1.0 / (1.0 + np.exp(-(z @ np.asarray(art["coef"], float) + art["intercept"])))
            return (pcalm >= art["threshold"]).astype(int), "gdelt"
    if "past14d_event_count" in cols:
        return (rows_df["past14d_event_count"].fillna(0).values == 0).astype(int), "acled"
    return np.ones(len(rows_df), dtype=int), "default"


def score(df: pd.DataFrame, model_dir: str | Path = MODEL_DIR,
          target_dates=None) -> pd.DataFrame:
    """model_input df → onset 점수. target_dates(set/list) 지정 시 그 날짜만(서빙=최신일)."""
    model = load_model(model_dir)
    ckpt, meta = model["ckpt"], model["meta"]
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    d = pd.to_datetime(df["date"]).dt.normalize()
    if target_dates is not None:
        td = {pd.Timestamp(x).normalize() for x in target_dates}
        mask = d.isin(td).values
    else:
        mask = np.ones(len(df), dtype=bool)

    rows, p_lstm = _predict_lstm(df, ckpt, mask)
    out = df.loc[rows, ["country", "date"]].copy()
    out["base_pred"] = p_lstm
    # meta(1입력 로지스틱): logit(p) → (x-mu)/sd → predict_proba
    x = np.log(np.clip(p_lstm, 1e-6, 1 - 1e-6) / (1 - np.clip(p_lstm, 1e-6, 1 - 1e-6)))
    xs = (x.reshape(-1, 1) - np.ravel(meta["mu"])) / np.ravel(meta["sd"])
    out["onset_prob"] = meta["meta"].predict_proba(xs)[:, 1]
    # calm 게이팅 — GDELT 우선(실시간), ACLED fallback
    flags, calm_src = _calm_flag(df.loc[rows], df.columns, model_dir)
    out["calm_flag"] = flags
    out = out.reset_index(drop=True)
    out.attrs["calm_source"] = calm_src
    return out


if __name__ == "__main__":
    # 스모크: 로컬 full_pca16 의 2024 일부로 스코어러 동작 확인
    import sys; sys.path.insert(0, ".")
    DATA = "input/processed/dataset/full_pca16_aclfree.parquet"
    df = pd.read_parquet(DATA)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df[(df["date"] >= "2024-06-01") & (df["date"] <= "2024-12-31")]
    res = score(df)
    print(f"스코어 {len(res)}행 | onset_prob 범위 [{res.onset_prob.min():.3f}, {res.onset_prob.max():.3f}]"
          f" 평균 {res.onset_prob.mean():.3f}")
    print(f"calm_flag: source={res.attrs.get('calm_source')} | calm {res.calm_flag.sum()}행 "
          f"({res.calm_flag.mean():.3f})")
    print(res.sort_values("onset_prob", ascending=False).head(8).to_string(index=False))
