import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from scripts.run_tree_experiments import feature_group_columns
from src.config import ARTIFACT_DIR, SEED, TARGET_COL
from src.data_utils import (
    build_preprocessor,
    drop_constant_feature_columns,
    get_feature_columns,
    load_dataset_bundle,
    make_model_input,
    split_xy,
)
from src.metrics import evaluate_predictions


FINAL_TREE_CANDIDATES = [
    {
        "candidate": "lightgbm_acled_se_rank1",
        "model_name": "lightgbm",
        "estimator": LGBMClassifier(
            random_state=SEED,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            scale_pos_weight=4.7,
            n_jobs=-1,
            verbosity=-1,
        ),
        "notes": "Best LightGBM by validation PR-AUC.",
    },
    {
        "candidate": "lightgbm_acled_se_rank2",
        "model_name": "lightgbm",
        "estimator": LGBMClassifier(
            random_state=SEED,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            scale_pos_weight=4.7,
            n_jobs=-1,
            verbosity=-1,
        ),
        "notes": "Second-best LightGBM by validation PR-AUC; tied best P@top5 among LightGBM candidates.",
    },
    {
        "candidate": "xgboost_acled_se_rank1",
        "model_name": "xgboost",
        "estimator": XGBClassifier(
            random_state=SEED,
            n_estimators=500,
            learning_rate=0.04,
            max_depth=5,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=3.0,
            scale_pos_weight=1.0,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
        ),
        "notes": "Best XGBoost by validation PR-AUC and best test stability among B candidates.",
    },
]


def make_acled_se_features(train_df: pd.DataFrame) -> list[str]:
    base_features = drop_constant_feature_columns(train_df, get_feature_columns(train_df))
    return feature_group_columns(base_features, "acled_se")


def fit_predict(estimator, model_name: str, train_df: pd.DataFrame, eval_df: pd.DataFrame, feature_columns: list[str]):
    x_train, y_train = split_xy(train_df, feature_columns, target_col=TARGET_COL)
    x_eval, _ = split_xy(eval_df, feature_columns, target_col=TARGET_COL)

    x_train = make_model_input(x_train, feature_columns, date_mode="ordinal")
    x_eval = make_model_input(x_eval, feature_columns, date_mode="ordinal")

    pipeline = Pipeline(
        [
            ("prep", build_preprocessor(x_train, model_name=model_name)),
            ("model", estimator),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline.predict_proba(x_eval)[:, 1]


def save_predictions(eval_df: pd.DataFrame, y_prob, output_path: Path) -> None:
    pred = eval_df[["date", "country", TARGET_COL]].copy()
    pred["date"] = pred["date"].dt.strftime("%Y-%m-%d")
    pred["y_prob"] = y_prob
    pred.to_csv(output_path, index=False, encoding="utf-8")


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    submission_dir = ARTIFACT_DIR / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_dataset_bundle("se")
    feature_columns = make_acled_se_features(train_df)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    rows = []
    for spec in FINAL_TREE_CANDIDATES:
        candidate = spec["candidate"]
        model_name = spec["model_name"]

        val_prob = fit_predict(spec["estimator"], model_name, train_df, val_df, feature_columns)
        val_metrics = evaluate_predictions(val_df, val_prob, history_df=train_df, target_col=TARGET_COL)
        rows.append(
            {
                "candidate": candidate,
                "split": "val",
                "trained_on": "train",
                "notes": spec["notes"],
                **val_metrics,
            }
        )
        save_predictions(val_df, val_prob, submission_dir / f"val_predictions__{candidate}.csv")

        test_prob = fit_predict(spec["estimator"], model_name, train_val_df, test_df, feature_columns)
        test_metrics = evaluate_predictions(test_df, test_prob, history_df=train_val_df, target_col=TARGET_COL)
        rows.append(
            {
                "candidate": candidate,
                "split": "test",
                "trained_on": "train+val",
                "notes": spec["notes"],
                **test_metrics,
            }
        )
        save_predictions(test_df, test_prob, submission_dir / f"test_predictions__{candidate}.csv")

    out = pd.DataFrame(rows)
    metrics_path = ARTIFACT_DIR / "final_tree_candidate_metrics.csv"
    out.to_csv(metrics_path, index=False, encoding="utf-8")

    display_cols = [
        "candidate",
        "split",
        "trained_on",
        "pr_auc",
        "persistence_pr_auc",
        "persistence_gain",
        "p_at_1pct",
        "p_at_5pct",
        "r_at_p_10",
        "ece_10bin",
    ]
    print(out[display_cols].to_string(index=False))
    print()
    print(f"saved metrics: {metrics_path}")
    print(f"saved predictions: {submission_dir}")


if __name__ == "__main__":
    main()
