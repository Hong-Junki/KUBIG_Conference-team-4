# tree-only vs LGBM+XGB+LSTM Stacking Comparison (Platt 기준)

| Metric | tree-only | +LSTM | delta |
|--------|-----------|-------|-------|
| PR-AUC | 0.2714 | 0.2656 | -0.0058 |
| P@5% | 0.2689 | 0.2670 | -0.0019 |
| P@10% | nan | 0.1818 | nan |
| Recall@Prec>=0.10 | nan | 0.7535 | nan |
| Recall@Prec>=0.20 | nan | 0.3837 | nan |
| Recall@Prec>=0.30 | nan | 0.2884 | nan |
| Brier Score | nan | 0.0359 | nan |
| ECE | 0.0083 | 0.0067 | -0.0016 |
