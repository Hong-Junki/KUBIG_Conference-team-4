# Macis LSTM Autoencoder 12년 재학습 — Colab 실행 가이드

`proc/colab/macis_retrain_12y.ipynb` 을 Colab T4 에서 돌리는 절차.

## 개요

Macis et al.(2024) "Breaking the Trend" 방법론 재현. LSTM Autoencoder 를 **평상시 (normal days) 데이터로 비지도 학습** 한 뒤, 새 입력의 재구성 오차 (Squared Error) 를 anomaly score 로 산출.

- 학습 대상: 분쟁 이벤트가 없는 평상시 30일 윈도우만 → 모델이 "평상시 패턴" 을 외움
- 추론 시: 모든 날짜 (평상시 + 사건 시) 입력 → 사건이 임박할수록 재구성 오차 (SE) 증가
- 산출물: `se_scores.parquet` — 국가×일별 SE score 1개. 이게 LSTM v4_se 의 56번째 입력 피처가 됨

## 1. Drive 사전 확인

기대 구조:
```
My Drive/
└── conflict-early-warning/
    └── input/
        └── processed/
            ├── acled/   ← *.parquet 58개 (~16MB)
            └── gdelt/   ← *.parquet 58개 (~3.3GB)
```

> 노트북 셀 1 의 `ACLED_DIR` / `GDELT_DIR` 이 위 경로를 가리키도록 설정됨. 셀 실행 시 `assert os.path.isdir(...)` 로 자동 검증.

## 2. Colab 실행

1. `proc/colab/macis_retrain_12y.ipynb` 을 Colab 으로 열기
2. **런타임 → 런타임 유형 변경 → T4 GPU** 설정
3. 셀 1~11 순서대로 실행
4. 예상 소요:
   - 데이터 로딩 + Normal days 수집: 5~10분 (Drive I/O, 12년치)
   - 학습: 60~120분 (300 epoch, early stop 으로 보통 150~250 epoch 에서 멈춤)
   - SE 산출 58국: 5~10분
   - Sanity check: 1~2분

총 **80~150분 예상**.

## 3. 모델 스펙

- **입력**: `(batch, seq_len=30, n_features=15)` — Macis 논문 기본 피처 셋 (ACLED rolling 5 + GDELT rolling 10)
- **구조**:
  - Encoder: LSTM (hidden=64) → Dense (64 → latent_dim=32)
  - Decoder: latent_dim 을 seq_len 만큼 repeat → LSTM (hidden=64) → Dense (64 → n_features)
- **Loss**: MSE (재구성 오차)
- **Optimizer**: Adam (lr=1e-3), batch=256, epochs=300, patience=20
- **Scaler**: 국가별 StandardScaler (normal days 만으로 fit, 추론 시 동일 scaler 사용)
- **학습 데이터 정의**: 사건 (`event_type ∈ {Battles, Explosions, VAC}`) 직전 30일 + 직후 30일 윈도우 제외한 나머지 (평상시)

## 4. 산출물 위치 (Google Drive)

학습 완료 후 Drive 에 다음 파일 자동 저장:

```
My Drive/conflict-early-warning/output/macis_12y/
├── model.pt              ← 글로벌 LSTM AE 가중치 (encoder + decoder)
├── se_scores.parquet     ← 핵심 산출물 (iso3, date, se_score 3열)
├── config.json           ← 하이퍼파라미터 + n_features + per-country scaler 메타
├── case_sanity.csv       ← UKR 2022.02 / SDN 2023.04 / PSE 2023.10 사건 직전 SE 상승 검증
├── train_loss.csv        ← epoch 별 train/val MSE
└── train_loss.png        ← loss 곡선
```

`se_scores.parquet` 은 LSTM Classifier v4_se 의 입력 피처로 직접 사용됨. 다른 팀원 (B / D) 의 트리 모델·스태킹 에서도 동일 파일 활용.

## 5. SE 검증 기준

학습 후 sanity check 통과 조건 (case_sanity.csv 확인):

- 3개 백테스트 케이스 모두 사건 직전 7~30일 구간에서 SE 가 상위 percentile 진입
- 평상시 평균 SE 대비 사건 직전 SE 가 2σ 이상 상승
- NaN ratio < 5% (seq_len-1 일치 만큼만 NaN)

## 6. 트러블슈팅

- **Drive 마운트 실패**: 셀 1 새로 실행 후 OAuth 재인증
- **CUDA OOM**: `CONFIG['batch_size']` 256 → 128 로 낮춤
- **patience early stop 너무 빠름**: `CONFIG['patience']` 20 → 40
- **데이터 로딩 3분 이상 멈춤**: Drive I/O 느림. `cp -r /content/drive/MyDrive/conflict-early-warning/input /content/input` 으로 로컬 디스크 복사 후 `ACLED_DIR` / `GDELT_DIR` 을 `/content/input/...` 로 바꿔서 재실행 (디스크 한도 100GB 이내)
- **SE 산출 시 새 국가 fallback warning**: 글로벌 모델 학습 당시 빠진 국가는 임시 scaler fit. 정합성 위해 해당 국가는 별도 표기 (현재 알려진 누수 위험 — 회의 안건)
