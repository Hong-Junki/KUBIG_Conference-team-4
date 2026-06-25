# outputs/

## 폴더 구조

```
outputs/
├── reports/    # 실험 요약, ablation 결과, 모델 비교 문서 (보존)
└── predictions/  # ← 팀 레포에서 제외됨 (아래 설명 참고)
```

## predictions/ 제외 이유

`outputs/predictions/` 아래의 OOF / validation / test prediction CSV 파일들은
팀 레포 최종 파일 트리에서 제외했습니다.

**이유:**
- 모델과 코드가 있으면 재생성 가능한 generated artifact입니다.
- 파일 수 75개, 합계 약 90MB로 팀 레포에 불필요한 부담이 됩니다.
- GitHub 레포지토리 권장 크기 가이드라인을 고려했습니다.

## predictions/ 복원 방법

필요한 경우 아래 두 가지 방법으로 복원할 수 있습니다.

### 방법 1 — 개인 레포 히스토리에서 직접 확인

```bash
# 개인 레포 (conflict-early-warning)의 git history에 전체 파일이 보존되어 있습니다.
git -C <path-to-conflict-early-warning> log --oneline
git -C <path-to-conflict-early-warning> checkout HEAD -- outputs/predictions/
```

### 방법 2 — 모델 재실행

```bash
# stacking 앙상블 예측 재생성
python modeling/run_stacking_d_prototype.py

# LightGBM 예측 재생성
python modeling/predict_lightgbm_se.py
```

자세한 실행 방법은 `../docs/pipeline_usage.md`를 참고하세요.
