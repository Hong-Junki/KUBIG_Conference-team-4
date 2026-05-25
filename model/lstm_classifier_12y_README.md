# LSTM Classifier v4_se — Colab 실행 가이드

`proc/colab/lstm_classifier_12y_v4_se.ipynb` 를 Colab T4 에서 돌리는 절차.

## 개요

C 카테고리(딥러닝 시퀀스) 본선 모델. 단일 LSTM 분류기에 다음 3가지 기법을 결합:

1. **Country Embedding** — 58개국을 8차원 학습 가능 embedding 으로 표현, 매 timestep 입력에 concat
2. **Multi-Task Learning** — 동일 trunk 에서 `y_escalation`(메인), `y_onset`, `y` 3개 head 동시 학습 (loss 가중합 1.0 / 0.3 / 0.3)
3. **SE 피처 전이** — Macis LSTM Autoencoder (사전학습 동결) 의 일별 재구성 오차를 입력 피처 1개로 추가

후처리로 **Platt Scaling** 캘리브레이션 적용해서 raw 확률의 양성 편향 보정.

## 1. Drive 사전 확인

기대 구조:
```
My Drive/
└── conflict-early-warning/
    ├── input/processed/dataset/
    │   ├── train.parquet
    │   ├── val.parquet
    │   └── test.parquet
    └── input/processed/features/
        └── se_scores.parquet   ← v4_se 의 56번째 피처로 사용 (필수)
```

> 노트북 셀 1 의 경로 검증으로 자동 확인. 누락 시 `assert` 에러.

`se_scores.parquet` 은 `macis_retrain_12y.ipynb` 산출물. v4_se 학습 전에 먼저 생성돼있어야 함.

## 2. Colab 실행

1. `proc/colab/lstm_classifier_12y_v4_se.ipynb` 을 Colab 으로 열기
2. **런타임 → 런타임 유형 변경 → T4 GPU** 설정
3. 셀 1~14 순서대로 실행
4. 예상 소요:
   - 데이터 로딩 + SE merge + 시퀀스 빌드: 5~10분
   - 학습: 15~25분 (50 epoch, early stop 으로 보통 7~17 epoch 에서 멈춤)
   - 평가 + 백테스트: 3~5분

총 **25~40분 예상**.

## 3. 모델 스펙

- **입력**: `(batch, seq_len=30, n_features=56)` — 55 base + 1 SE
  - country index 별도 입력 `(batch,)` → embedding 조회 후 매 timestep concat (실제 LSTM 입력 차원 63)
- **구조**:
  - Country Embedding (58 → 8)
  - LSTM (input=63, hidden=128, 1 layer)
  - 마지막 hidden state → Dropout(0.3) → Linear(128 → 64) → ReLU
  - 3 head: Linear(64 → 1) × 3 (y_escalation / y_onset / y)
- **Loss**: `1.0 · BCE_esc + 0.3 · BCE_onset + 0.3 · BCE_y`
  - 각 head 별 pos_weight 자동 계산 후 clip 적용 (esc 30 / onset 100 / y 무제한)
- **Optimizer**: Adam (lr=1e-3), batch=256, epochs=50, patience=10
- **Scaler**: 국가별 StandardScaler (train 구간으로만 fit, val/test 는 transform 만)
- **Split**: train 2014-01 ~ 2023-12 / val 2024-01 ~ 2024-06 / test 2024-07 ~ 2025-03
- **평가 head**: `head_esc` 만 사용 (onset / y 는 학습 보조)

## 4. 산출물 위치 (Google Drive)

학습 완료 후 Drive 에 다음 파일 자동 저장:

```
My Drive/conflict-early-warning/output/lstm_classifier_12y_v4_se/
├── model.pt              ← state_dict + config + feature_cols + country_to_idx + per-head pos_weight
├── config.json           ← 하이퍼파라미터 + eval 결과 (raw)
├── eval.json             ← 6지표군 단독 (다른 모델과 비교용, raw)
├── predictions.parquet   ← val/test 예측 (country, date, y_true, y_prob, split)
├── train_history.csv     ← epoch 별 loss / val PR-AUC
├── train_loss.png        ← loss + val PR-AUC 곡선
└── backtest_3cases.png   ← UKR 2022.02 / SDN 2023.04 / PSE 2023.10 D-30 ~ D+7
```

> Platt Scaling 캘리브레이션은 Colab 학습 후 로컬에서 별도 스크립트 (`scripts/lstm_platt_scaling.py --version v4_se`) 로 적용. Drive 산출물에는 raw 결과만 포함.
> - 원리: val 구간 (y_prob, y_true) 로 sklearn LogisticRegression fit → test 구간 보정
> - 효과: 순위 보존이라 PR-AUC / top-K precision·recall 은 동일, ECE 만 즉시 개선 (0.293 → 0.008)
> - 결과 산출물 (calibrator.pkl + predictions_calibrated.parquet + eval_calibrated.json) 은 로컬 only

## 5. 성능 결과 (test 기준)

Platt Scaling 적용 후 (운영 점수 변환 시 반드시 이 값 사용):

| Metric | Value | 비고 |
|---|---:|---|
| PR-AUC | 0.2410 | persistence baseline 0.0354 |
| persistence_gain | +0.2056 | 합격선 > 0 통과 |
| precision @ top-5% | 0.2561 | |
| recall @ top-5% | 0.3150 | |
| recall @ P>=0.10 | 0.7006 | 운영 임계 0.10 시 양성 70% 잡음 |
| recall @ P>=0.20 | 0.3715 | |
| recall @ P>=0.30 | 0.2665 | |
| lead time (median) | 8 days | 사건 평균 8일 전 알림 가능 |
| ECE (calibrated) | 0.0075 | raw 0.293 → Platt 후 0.008 |
| Platt 함수 | `sigmoid(4.88·x − 5.52)` | |

학습 종료: best_epoch = 7 (val PR-AUC 기준 early stop)

## 6. 트러블슈팅

- **CUDA OOM**: `CONFIG['batch_size']` 256 → 128 또는 64 로 낮춤. `hidden_dim` 128 → 64
- **학습 NaN**: lr 1e-3 → 5e-4 로 낮춤, 시퀀스 내 inf/NaN 확인 (cell 5 NaN 출력)
- **SE merge 실패**: `se_scores.parquet` 이 Drive 에 있는지 확인. `macis_retrain_12y.ipynb` 먼저 실행 필요
- **val PR-AUC 가 epoch 5 이내에 갱신 멈춤**: dropout 0.3 → 0.5, lr 1e-3 → 3e-4
- **country embedding lookup 에러**: `country_to_idx` 가 train/val/test 모두 같은 매핑인지 확인 (셀 6 build_sequences_v4 에서 일관성 유지)
- **multi-task loss 가 esc 단독보다 나쁨**: β, γ 가중치 조정 (0.3 → 0.1 ~ 0.5 sweep). onset 양성 너무 sparse 하면 0 으로 끄는 것도 옵션
