"""GKG 임베딩 앵커 컨셉 cosine 피처 생성 (B 강화 2차).

방법:
  1. 도메인 앵커 컨셉 텍스트 N개 정의 (armed conflict, civilian casualties 등)
  2. OpenAI text-embedding-3-small 로 앵커 임베딩 (N, 1536) 생성
  3. 각 (country, date) row 의 raw 1536 임베딩과 cosine similarity 계산 → N 차원 피처
  4. parquet 저장

목적:
  - PCA 64 차원이 LGBM 에서 차원당 1/3 신호로 노이즈 작용. 64 → 12~16 해석 가능 차원으로 압축.
  - 도메인 지식(분쟁 관련 컨셉)을 사용해 supervised dim reduction 없이 타겟에 가까운 신호 보존.

출력:
  input/processed/features/gkg_emb_anchors.parquet  (date, country, anchor_cos_<concept> N개)
  input/processed/features/anchor_embeddings.npz    (앵커 임베딩 행렬 + 라벨, 실시간 추론 시 재사용)

사용법:
  python scripts/gkg_embed/08_anchor_features.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env", override=True)

MODEL = "text-embedding-3-small"
EMB_DIM = 1536

SRC_EMB_PATH = Path("input/processed/features/gkg_embeddings.parquet")
OUT_PATH = Path("input/processed/features/gkg_emb_anchors.parquet")
ANCHOR_NPZ_PATH = Path("input/processed/features/anchor_embeddings.npz")

ANCHORS: dict[str, str] = {
    "armed_conflict": "armed conflict, war, militants attacking civilians and government forces",
    "civilian_casualties": "civilian casualties, deaths and injuries from violence, mass killings",
    "military_aggression": "military aggression, troop movements, invasion, military buildup near border",
    "protest_unrest": "protests, demonstrations, riots, civil unrest, clashes with police",
    "refugee_displacement": "refugees fleeing violence, internally displaced people, humanitarian exodus",
    "political_instability": "political crisis, government collapse, leadership vacuum, parliamentary deadlock",
    "terrorism": "terrorist attack, bombing, suicide bomber, insurgent ambush, extremist violence",
    "sanctions": "economic sanctions, trade embargo, asset freeze, diplomatic isolation",
    "diplomatic_crisis": "diplomatic crisis, ambassador expelled, embassy closed, severed relations",
    "economic_shock": "economic shock, currency collapse, hyperinflation, market crash, food shortage",
    "humanitarian_crisis": "humanitarian crisis, famine, mass starvation, aid blocked, disease outbreak",
    "coup": "coup d'etat, military takeover, junta seizes power, president overthrown",
    "ethnic_tension": "ethnic tension, sectarian violence, communal clashes, religious persecution",
    "ceasefire_breakdown": "ceasefire collapses, peace deal violated, hostilities resume, truce broken",
    "weapons_proliferation": "weapons transfer, arms smuggling, missile launch, nuclear program escalation",
    "border_clash": "border clash, territorial dispute, cross-border attack, military skirmish",
}


def get_anchor_embeddings(client: OpenAI) -> tuple[np.ndarray, list[str]]:
    labels = list(ANCHORS.keys())
    texts = [ANCHORS[k] for k in labels]
    print(f"앵커 컨셉 {len(labels)}개 임베딩 요청")
    resp = client.embeddings.create(model=MODEL, input=texts)
    embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, 1e-12, None)
    print(f"  앵커 임베딩 shape={embs.shape}, mean_norm={np.linalg.norm(embs, axis=1).mean():.4f}")
    return embs, labels


def compute_cosine_features(emb_df: pd.DataFrame, emb_cols: list[str],
                             anchors: np.ndarray, labels: list[str]) -> pd.DataFrame:
    X = emb_df[emb_cols].values.astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.clip(norms, 1e-12, None)
    cos = Xn @ anchors.T
    print(f"  cosine matrix shape={cos.shape}, range=[{cos.min():.3f}, {cos.max():.3f}]")
    out = pd.DataFrame({"date": emb_df["date"].values, "country": emb_df["country"].values})
    for i, lab in enumerate(labels):
        out[f"gkg_anchor_cos_{lab}"] = cos[:, i].astype(np.float32)
    return out


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY 환경변수 필요"
    client = OpenAI(api_key=api_key)

    print(f"[1/3] 앵커 임베딩 ({len(ANCHORS)}개) 생성")
    anchors, labels = get_anchor_embeddings(client)
    np.savez(ANCHOR_NPZ_PATH, anchors=anchors, labels=np.array(labels))
    print(f"  저장: {ANCHOR_NPZ_PATH}")

    print(f"[2/3] 임베딩 로드 (raw 1536)")
    emb = pd.read_parquet(SRC_EMB_PATH)
    emb["date"] = pd.to_datetime(emb["date"], utc=True)
    emb_cols = [c for c in emb.columns if c.startswith("gkg_emb_") and c != "gkg_emb_n_titles_1d"]
    assert len(emb_cols) == EMB_DIM
    print(f"  emb shape={emb.shape}")

    print(f"[3/3] cosine 피처 계산")
    out = compute_cosine_features(emb, emb_cols, anchors, labels)
    print(f"  out shape={out.shape}")
    out.to_parquet(OUT_PATH, index=False)
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
