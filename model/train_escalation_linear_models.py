from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "input" / "processed" / "dataset"
FEATURE_DIR = ROOT / "input" / "processed" / "features"
OUT_DIR = ROOT / "outputs" / "escalation_7d_linear_v1"

TARGET = "y_escalation_7d"
SOURCE_LABEL = "y_escalation"
HORIZON_DAYS = 7
ID_COLS = {"date", "country"}
DROP_COLS = {
    "y",
    "y_onset",
    "y_escalation",
    "y_escalation_7d",
    "fatalities_next3d",
    "event_count_next3d",
}


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_parquet(DATASET_DIR / f"{name}.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df


def load_full_panel() -> pd.DataFrame:
    df = pd.read_parquet(DATASET_DIR / "full.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df


def make_future_label(
    df: pd.DataFrame,
    source_col: str = SOURCE_LABEL,
    target_col: str = TARGET,
    horizon_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    if target_col == source_col and horizon_days == 3:
        return df

    out = df.sort_values(["country", "date"]).copy()
    pieces = []
    for _, g in out.groupby("country", sort=False):
        future = pd.concat(
            [g[source_col].shift(-lag) for lag in range(1, horizon_days + 1)],
            axis=1,
        )
        complete = future.notna().all(axis=1)
        label = future.max(axis=1)
        label[~complete] = np.nan
        pieces.append(label)
    out[target_col] = pd.concat(pieces).sort_index()
    return out


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[(df["date"] >= "2014-01-01") & (df["date"] <= "2023-12-31")].copy()
    val = df[(df["date"] >= "2024-01-01") & (df["date"] <= "2024-06-30")].copy()
    test = df[(df["date"] >= "2024-07-01") & (df["date"] <= "2025-03-28")].copy()
    return train, val, test


def attach_se_score(df: pd.DataFrame) -> pd.DataFrame:
    path = FEATURE_DIR / "se_scores.parquet"
    if not path.exists():
        df["se_score"] = np.nan
        return df

    se = pd.read_parquet(path)
    se["date"] = pd.to_datetime(se["date"]).dt.tz_localize(None)
    se = se.rename(columns={"iso3": "country"})
    return df.merge(se[["country", "date", "se_score"]], on=["country", "date"], how="left")


def split_xy(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    return df[features].copy(), df[TARGET].astype(int).to_numpy()


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0, 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), float(precision[idx]), float(recall[idx]), float(f1[idx])


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
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
        "brier": float(brier_score_loss(y_true, np.clip(scores, 0, 1))),
        "ece": float(expected_calibration_error(y_true, scores)),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
    else:
        out["roc_auc"] = float("nan")
    return out


def expected_calibration_error(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true).astype(float)
    p = np.clip(np.asarray(scores).astype(float), 0, 1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= bins[i]) & (p <= bins[i + 1])
        else:
            mask = (p >= bins[i]) & (p < bins[i + 1])
        if not np.any(mask):
            continue
        ece += (mask.mean()) * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def fit_platt(y_val: np.ndarray, val_scores: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    calibrator.fit(np.asarray(val_scores).reshape(-1, 1), y_val)
    return calibrator


def apply_platt(calibrator: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]


def markdown_table(df: pd.DataFrame) -> str:
    d = df.copy()
    for col in d.columns:
        if pd.api.types.is_float_dtype(d[col]):
            d[col] = d[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            d[col] = d[col].map(lambda x: "" if pd.isna(x) else str(x))
    rows = [
        "| " + " | ".join(d.columns) + " |",
        "| " + " | ".join("---" for _ in d.columns) + " |",
    ]
    for _, row in d.iterrows():
        rows.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in d.columns) + " |")
    return "\n".join(rows)


def choose_features(train: pd.DataFrame) -> list[str]:
    features = []
    for col in train.columns:
        if col in ID_COLS or col in DROP_COLS or col.startswith("y_"):
            continue
        if col == "country" or pd.api.types.is_numeric_dtype(train[col]):
            features.append(col)
    return features


def build_preprocessor(train: pd.DataFrame, features: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_cols = [c for c in features if c == "country"]
    numeric_cols = [c for c in features if c not in categorical_cols]

    skew = train[numeric_cols].skew(numeric_only=True).replace([np.inf, -np.inf], np.nan)
    mins = train[numeric_cols].min(numeric_only=True)
    log_cols = [
        c for c in numeric_cols
        if pd.notna(skew.get(c)) and skew[c] > 5 and pd.notna(mins.get(c)) and mins[c] >= 0
    ]
    other_numeric_cols = [c for c in numeric_cols if c not in log_cols]

    log_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    transformers = []
    if log_cols:
        transformers.append(("log_numeric", log_pipe, log_cols))
    if other_numeric_cols:
        transformers.append(("numeric", numeric_pipe, other_numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", cat_pipe, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0.3)
    return preprocessor, log_cols, other_numeric_cols


def persistence_scores(df: pd.DataFrame, window: int) -> np.ndarray:
    col = f"acled_event_count_{window}d"
    if col not in df.columns:
        raise KeyError(f"Missing persistence column: {col}")
    return (pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0, upper=window) / window).to_numpy()


def fit_logistic_family(
    name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    preprocessor: ColumnTransformer,
    configs: list[dict],
) -> tuple[list[dict], Pipeline, pd.DataFrame]:
    X_train, y_train = split_xy(train, features)
    X_val, y_val = split_xy(val, features)
    X_test, y_test = split_xy(test, features)

    rows = []
    best = None
    best_pipe = None
    best_val_scores = None

    for cfg in configs:
        model = SGDClassifier(
            loss="log_loss",
            max_iter=cfg.get("max_iter", 1000),
            tol=cfg.get("tol", 1e-3),
            penalty=cfg["penalty"],
            alpha=cfg["alpha"],
            l1_ratio=cfg.get("l1_ratio"),
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=42,
        )
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        val_scores = pipe.predict_proba(X_val)[:, 1]
        threshold, _, _, val_f1 = best_f1_threshold(y_val, val_scores)
        val_metrics = evaluate(y_val, val_scores, threshold)
        row = {
            "model": name,
            "score_type": "raw_tuning",
            "penalty": cfg["penalty"],
            "alpha": cfg["alpha"],
            "l1_ratio": cfg.get("l1_ratio", np.nan),
            "split": "validation",
            **val_metrics,
        }
        rows.append(row)

        if best is None or val_f1 > best["val_f1"]:
            best = {"cfg": cfg, "threshold": threshold, "val_f1": val_f1}
            best_pipe = pipe
            best_val_scores = val_scores

    assert best is not None and best_pipe is not None and best_val_scores is not None

    raw_scores_by_split = {}
    for split_name, X, y in [
        ("train", X_train, y_train),
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        scores = best_pipe.predict_proba(X)[:, 1] if split_name != "validation" else best_val_scores
        raw_scores_by_split[split_name] = scores
        metrics = evaluate(y, scores, best["threshold"])
        rows.append(
            {
                "model": name,
                "score_type": "raw",
                "penalty": best["cfg"]["penalty"],
                "alpha": best["cfg"]["alpha"],
                "l1_ratio": best["cfg"].get("l1_ratio", np.nan),
                "split": split_name,
                **metrics,
            }
        )

    calibrator = fit_platt(y_val, raw_scores_by_split["validation"])
    val_calibrated = apply_platt(calibrator, raw_scores_by_split["validation"])
    calibrated_threshold, _, _, _ = best_f1_threshold(y_val, val_calibrated)
    for split_name, y in [
        ("train", y_train),
        ("validation", y_val),
        ("test", y_test),
    ]:
        calibrated_scores = apply_platt(calibrator, raw_scores_by_split[split_name])
        metrics = evaluate(y, calibrated_scores, calibrated_threshold)
        rows.append(
            {
                "model": name,
                "score_type": "platt",
                "penalty": best["cfg"]["penalty"],
                "alpha": best["cfg"]["alpha"],
                "l1_ratio": best["cfg"].get("l1_ratio", np.nan),
                "split": split_name,
                **metrics,
            }
        )

    test_pred = test[["country", "date", TARGET]].copy()
    test_pred[f"{name}_score"] = raw_scores_by_split["test"]
    test_pred[f"{name}_platt_score"] = apply_platt(calibrator, raw_scores_by_split["test"])
    test_pred[f"{name}_prediction"] = test_pred[f"{name}_score"] >= best["threshold"]
    test_pred[f"{name}_platt_prediction"] = (
        test_pred[f"{name}_platt_score"] >= calibrated_threshold
    )
    return rows, best_pipe, test_pred


def coefficient_table(pipe: Pipeline, top_n: int = 80) -> pd.DataFrame:
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_[0]
    out = pd.DataFrame({"feature": names, "coefficient": coefs})
    out["abs_coefficient"] = out["coefficient"].abs()
    return out.sort_values("abs_coefficient", ascending=False).head(top_n)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    full = attach_se_score(load_full_panel())
    full = make_future_label(full)
    full = full.dropna(subset=[TARGET]).copy()
    full[TARGET] = full[TARGET].astype(int)
    train, val, test = temporal_split(full)

    features = choose_features(train)
    preprocessor, log_cols, other_numeric_cols = build_preprocessor(train, features)

    y_train = train[TARGET].astype(int).to_numpy()
    y_val = val[TARGET].astype(int).to_numpy()
    y_test = test[TARGET].astype(int).to_numpy()

    metrics_rows: list[dict] = []
    persistence_candidates = []
    for window in [7, 14, 30]:
        val_scores = persistence_scores(val, window)
        threshold, _, _, val_f1 = best_f1_threshold(y_val, val_scores)
        persistence_candidates.append((window, threshold, val_f1))
        raw_scores_by_split = {
            "train": persistence_scores(train, window),
            "validation": val_scores,
            "test": persistence_scores(test, window),
        }
        for split_name, df, y in [
            ("train", train, y_train),
            ("validation", val, y_val),
            ("test", test, y_test),
        ]:
            scores = raw_scores_by_split[split_name]
            metrics_rows.append(
                {
                    "model": f"Persistence_{window}d",
                    "score_type": "raw",
                    "penalty": "",
                    "alpha": np.nan,
                    "l1_ratio": np.nan,
                    "split": split_name,
                    **evaluate(y, scores, threshold),
                }
            )
        calibrator = fit_platt(y_val, val_scores)
        val_calibrated = apply_platt(calibrator, val_scores)
        calibrated_threshold, _, _, _ = best_f1_threshold(y_val, val_calibrated)
        for split_name, y in [
            ("train", y_train),
            ("validation", y_val),
            ("test", y_test),
        ]:
            calibrated_scores = apply_platt(calibrator, raw_scores_by_split[split_name])
            metrics_rows.append(
                {
                    "model": f"Persistence_{window}d",
                    "score_type": "platt",
                    "penalty": "",
                    "alpha": np.nan,
                    "l1_ratio": np.nan,
                    "split": split_name,
                    **evaluate(y, calibrated_scores, calibrated_threshold),
                }
            )

    best_persistence = max(persistence_candidates, key=lambda x: x[2])
    best_persistence_model = f"Persistence_{best_persistence[0]}d"

    model_specs = [
        (
            "Logistic_L2",
            [
                {"penalty": "l2", "alpha": 1e-4, "max_iter": 1000},
                {"penalty": "l2", "alpha": 1e-5, "max_iter": 1000},
            ],
        ),
        (
            "Logistic_L1",
            [
                {"penalty": "l1", "alpha": 1e-4, "max_iter": 1000},
                {"penalty": "l1", "alpha": 1e-5, "max_iter": 1000},
            ],
        ),
        (
            "Logistic_ElasticNet",
            [
                {"penalty": "elasticnet", "alpha": 1e-4, "l1_ratio": 0.5, "max_iter": 1000},
                {"penalty": "elasticnet", "alpha": 1e-5, "l1_ratio": 0.5, "max_iter": 1000},
            ],
        ),
    ]

    prediction_frames = []
    coef_frames = []
    for model_name, configs in model_specs:
        rows, pipe, pred = fit_logistic_family(
            model_name, train, val, test, features, preprocessor, configs
        )
        metrics_rows.extend(rows)
        prediction_frames.append(pred)
        coef = coefficient_table(pipe)
        coef["model"] = model_name
        coef_frames.append(coef)
        joblib.dump(pipe, OUT_DIR / f"model__{model_name}.joblib")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    pd.Series(features, name="feature").to_csv(OUT_DIR / "feature_list.csv", index=False, encoding="utf-8-sig")
    pd.Series(log_cols, name="log1p_feature").to_csv(OUT_DIR / "log1p_features.csv", index=False, encoding="utf-8-sig")
    pd.concat(coef_frames, ignore_index=True).to_csv(
        OUT_DIR / "coefficients_top.csv", index=False, encoding="utf-8-sig"
    )

    if prediction_frames:
        pred = prediction_frames[0]
        for frame in prediction_frames[1:]:
            pred = pred.merge(frame.drop(columns=[TARGET]), on=["country", "date"], how="left")
        pred.to_csv(OUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

    test_rows = metrics_df[metrics_df["split"] == "test"].copy()
    persistence_test_f1 = float(
        test_rows.loc[
            (test_rows["model"] == best_persistence_model)
            & (test_rows["score_type"] == "raw"),
            "f1",
        ].iloc[0]
    )
    test_rows["persistence_gain_f1"] = test_rows["f1"] - persistence_test_f1

    summary = {
        "target": TARGET,
        "source_label": SOURCE_LABEL,
        "horizon_days": HORIZON_DAYS,
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_positive_rate": float(train[TARGET].mean()),
        "validation_positive_rate": float(val[TARGET].mean()),
        "test_positive_rate": float(test[TARGET].mean()),
        "features": len(features),
        "log1p_features": log_cols,
        "best_persistence_model": best_persistence_model,
        "best_persistence_validation_f1": float(best_persistence[2]),
        "best_persistence_test_f1": persistence_test_f1,
    }
    (OUT_DIR / "metrics_all.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = [
        "# Escalation Linear Models v1",
        "",
        f"Target: `{TARGET}`",
        "",
        "## Data",
        "",
        f"- Train rows: {len(train):,}, positive rate: {train[TARGET].mean():.4f}",
        f"- Validation rows: {len(val):,}, positive rate: {val[TARGET].mean():.4f}",
        f"- Test rows: {len(test):,}, positive rate: {test[TARGET].mean():.4f}",
        f"- Features: {len(features)}",
        f"- `log1p` transformed features: {len(log_cols)}",
        f"- Best persistence baseline: `{best_persistence_model}`",
        f"- Source label: `{SOURCE_LABEL}`",
        f"- Horizon days: {HORIZON_DAYS}",
        "",
        "## Test Metrics",
        "",
        markdown_table(
            test_rows[
                [
                    "model",
                    "score_type",
                    "precision",
                    "recall",
                    "f1",
                    "pr_auc",
                    "roc_auc",
                    "brier",
                    "ece",
                    "persistence_gain_f1",
                ]
            ].sort_values("f1", ascending=False)
        ),
        "",
        "## Notes",
        "",
        "- Persistence baseline uses clipped prior 7/14/30-day ACLED event-count averages as probability scores.",
        "- Platt calibration fits a sigmoid on validation scores and reports calibrated probabilities on test.",
        "- ECE uses 10 equal-width probability bins.",
        "- Logistic models use `class_weight='balanced'` because `y_escalation` is around 4%.",
        "- Numeric features with train skew > 5 and non-negative support are transformed with `log1p` before scaling.",
        "- `country` is one-hot encoded; `se_score` is merged from `input/processed/features/se_scores.parquet`.",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Wrote outputs to {OUT_DIR}")
    print(
        test_rows[
            [
                "model",
                "score_type",
                "precision",
                "recall",
                "f1",
                "pr_auc",
                "roc_auc",
                "brier",
                "ece",
                "persistence_gain_f1",
            ]
        ]
        .sort_values("f1", ascending=False)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
