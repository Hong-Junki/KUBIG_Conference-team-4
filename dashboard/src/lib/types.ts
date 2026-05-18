export type RiskLevel = 'Low' | 'Moderate' | 'High' | 'Critical';

export interface Prediction {
  date: string;       // YYYY-MM-DD
  country: string;    // ISO3
  y_prob: number;     // 0–1
  risk_score: number; // y_prob × 100
  risk_level: RiskLevel;
}

export interface ModelMetrics {
  model_name: string;
  pr_auc: number;
  p_at_top5pct: number;
  recall_at_precision_010: number;
  ece: number;
  brier_score: number;
}
