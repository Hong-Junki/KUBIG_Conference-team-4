from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
PANEL_PATH = ROOT / "outputs" / "eda_downloaded_sources" / "supplementary" / "country_day_panel_v2.csv"
OUT_DIR = ROOT / "outputs" / "early_warning_lgbm_v1"

TARGETS = ["future_onset_1d", "future_onset_3d", "future_onset_7d", "future_onset_14d"]
LABEL_COLS = {
    "conflict_day",
    "conflict_onset",
    "future_onset_1d",
    "future_onset_3d",
    "future_onset_7d",
    "future_onset_14d",
}
DROP_COLS = {
    "country",
    "date",
    "event_date",
    # ACLED labels arrive with delay in a real-time system. Use shifted/rolling
    # ACLED-derived features instead of same-day event counts.
    "acled_events",
    "armed_conflict_events",
    "protest_riot_events",
    "fatalities",
}


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL_PATH)
    df["date"] = pd.to_datetime(df["date"])
    for col in LABEL_COLS:
        if col in df.columns:
            df[col] = as_bool(df[col])
    df["iso3"] = df["iso3"].astype("category")
    df["month"] = df["date"].dt.month.astype("int16")
    df["dayofyear"] = df["date"].dt.dayofyear.astype("int16")
    return df.sort_values(["date", "iso3"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in LABEL_COLS or col in DROP_COLS:
            continue
        if col == "iso3":
            cols.append(col)
            continue
        if pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[(df["date"] >= "2014-01-01") & (df["date"] <= "2022-12-31")].copy()
    val = df[(df["date"] >= "2023-01-01") & (df["date"] <= "2023-12-31")].copy()
    test = df[(df["date"] >= "2024-01-01") & (df["date"] <= "2025-03-31")].copy()
    return train, val, test


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0, 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), float(precision[idx]), float(recall[idx]), float(f1[idx])


def eval_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    out = {
        "positive_rate": float(np.mean(y_true)),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
    else:
        out["roc_auc"] = float("nan")
    return out


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    d = df.copy()
    for col in d.columns:
        if pd.api.types.is_float_dtype(d[col]):
            d[col] = d[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            d[col] = d[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(d.columns)
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in d.iterrows():
        rows.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in headers) + " |")
    return "\n".join(rows)


def fit_target(df: pd.DataFrame, target: str, features: list[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    train, val, test = split_data(df)
    y_train = train[target].astype(int)
    y_val = val[target].astype(int)
    y_test = test[target].astype(int)

    X_train = train[features]
    X_val = val[features]
    X_test = test[features]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        categorical_feature=["iso3"] if "iso3" in features else "auto",
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )

    val_scores = model.predict_proba(X_val)[:, 1]
    threshold, val_p, val_r, val_f1 = best_f1_threshold(y_val.to_numpy(), val_scores)
    test_scores = model.predict_proba(X_test)[:, 1]
    train_scores = model.predict_proba(X_train)[:, 1]

    metrics = {
        "target": target,
        "features": len(features),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "validation_threshold_search": {
            "threshold": threshold,
            "precision": val_p,
            "recall": val_r,
            "f1": val_f1,
        },
        "train": eval_scores(y_train.to_numpy(), train_scores, threshold),
        "validation": eval_scores(y_val.to_numpy(), val_scores, threshold),
        "test": eval_scores(y_test.to_numpy(), test_scores, threshold),
    }

    pred = test[["iso3", "country", "date", target]].copy()
    pred["score"] = test_scores
    pred["prediction"] = pred["score"] >= threshold

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)

    joblib.dump(model, OUT_DIR / f"model__{target}.joblib")
    return metrics, pred, importance


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_panel()
    features = feature_columns(df)
    pd.Series(features, name="feature").to_csv(OUT_DIR / "feature_list.csv", index=False, encoding="utf-8-sig")

    all_metrics = []
    prediction_frames = []
    importance_frames = []
    for target in TARGETS:
        metrics, pred, importance = fit_target(df, target, features)
        all_metrics.append(metrics)
        pred["target"] = target
        importance["target"] = target
        prediction_frames.append(pred)
        importance_frames.append(importance)
        (OUT_DIR / f"metrics__{target}.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pred.to_csv(OUT_DIR / f"predictions__{target}.csv", index=False, encoding="utf-8-sig")
        importance.to_csv(OUT_DIR / f"feature_importance__{target}.csv", index=False, encoding="utf-8-sig")

    rows = []
    for item in all_metrics:
        for split in ["train", "validation", "test"]:
            row = {"target": item["target"], "split": split, **item[split]}
            row["best_iteration"] = item["best_iteration"]
            rows.append(row)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        OUT_DIR / "predictions_all_targets.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(importance_frames, ignore_index=True).to_csv(
        OUT_DIR / "feature_importance_all_targets.csv", index=False, encoding="utf-8-sig"
    )
    (OUT_DIR / "metrics_all.json").write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    test_summary = metrics_df[metrics_df["split"].eq("test")].copy()
    lines = [
        "# Early Warning LightGBM v1",
        "",
        "Temporal split:",
        "- Train: 2014-01-01 ~ 2022-12-31",
        "- Validation: 2023-01-01 ~ 2023-12-31",
        "- Test: 2024-01-01 ~ 2025-03-31",
        "",
        "Targets: future_onset_1d, future_onset_3d, future_onset_7d, future_onset_14d",
        "",
        "## Test metrics",
        markdown_table(test_summary[["target", "positive_rate", "precision", "recall", "f1", "pr_auc", "roc_auc", "brier"]]),
        "",
        "Notes:",
        "- Current/future label columns were excluded from features.",
        "- Engineered rolling features were generated with shift(1) in the EDA pipeline.",
        "- Thresholds were selected on validation by maximum F1 and applied unchanged to test.",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved outputs to: {OUT_DIR}")
    print(test_summary[["target", "precision", "recall", "f1", "pr_auc", "roc_auc", "brier"]].to_string(index=False))


if __name__ == "__main__":
    main()
