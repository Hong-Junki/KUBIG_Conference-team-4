"""실시간 서빙 모듈 — 학습 피처 파이프라인을 정확히 복제(parity)하는 서빙용 변환.

핵심 원칙:
- 모든 fit 아티팩트(PCA·앵커·preesc·kmeans·임계값)는 학습본을 load 후 transform만 (refit 금지 = 누수 차단).
- 산출 피처는 학습 데이터셋(full_pca16_aclfree.parquet) 값과 parity(부동소수점 오차 내) 보장.
"""
