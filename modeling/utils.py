import os
import json
import pandas as pd

# ── Project root (one level up from this file's directory) ──────────────────
_MODELING_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_MODELING_DIR)

# ── Column constants ─────────────────────────────────────────────────────────
TARGET_COL = "y_escalation"
DATE_COL = "date"
CATEGORICAL_COLS = ["country"]

LABEL_META_COLS = [
    "y", "y_onset", "y_escalation",
    "fatalities_next3d", "event_count_next3d",
    "past14d_event_count", "past14d_fatalities_mean",
]

# ── Path constants ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(_ROOT, "input", "processed", "dataset")
MODEL_DIR = os.path.join(_ROOT, "outputs", "models")
PRED_DIR = os.path.join(_ROOT, "outputs", "predictions")
REPORT_DIR = os.path.join(_ROOT, "outputs", "reports")

TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
VAL_PATH = os.path.join(DATA_DIR, "val.parquet")
TEST_PATH = os.path.join(DATA_DIR, "test.parquet")
MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm_baseline.pkl")
PRED_PATH = os.path.join(PRED_DIR, "predictions__lightgbm__byeonghyeon.csv")
REPORT_PATH = os.path.join(REPORT_DIR, "lightgbm_val_metrics.json")


def ensure_output_dirs():
    for d in [MODEL_DIR, PRED_DIR, REPORT_DIR]:
        os.makedirs(d, exist_ok=True)


def load_datasets():
    """Load train, val, test parquet files. Returns (train_df, val_df, test_df)."""
    print("Loading datasets...")
    train = pd.read_parquet(TRAIN_PATH)
    val = pd.read_parquet(VAL_PATH)
    test = pd.read_parquet(TEST_PATH)
    print(f"  train : {train.shape[0]:>6,} rows × {train.shape[1]} cols  "
          f"({train[DATE_COL].min().date()} ~ {train[DATE_COL].max().date()})")
    print(f"  val   : {val.shape[0]:>6,} rows × {val.shape[1]} cols  "
          f"({val[DATE_COL].min().date()} ~ {val[DATE_COL].max().date()})")
    print(f"  test  : {test.shape[0]:>6,} rows × {test.shape[1]} cols  "
          f"({test[DATE_COL].min().date()} ~ {test[DATE_COL].max().date()})")
    return train, val, test


def get_feature_columns(df):
    """Return feature column names by excluding label/meta/date columns from df."""
    exclude = set(LABEL_META_COLS) | {DATE_COL}
    return [c for c in df.columns if c not in exclude]


def make_xy(df, feature_cols):
    """
    Split df into (X, y).
    - country is converted to pandas Categorical for LightGBM.
    - y is None when TARGET_COL is absent (e.g. stripped test sets).
    """
    X = df[feature_cols].copy()
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    y = df[TARGET_COL].copy() if TARGET_COL in df.columns else None
    return X, y


def validate_prediction_file(pred_df, test_df):
    """
    Validate the submission CSV against test_df.
    Raises ValueError on any violation; prints a summary on success.
    """
    errors = []

    # Column order and names
    expected_cols = ["date", "country", "y_prob"]
    if list(pred_df.columns) != expected_cols:
        errors.append(
            f"Column mismatch — expected {expected_cols}, got {list(pred_df.columns)}"
        )

    # Row count
    if len(pred_df) != len(test_df):
        errors.append(
            f"Row count mismatch — pred={len(pred_df):,}, test={len(test_df):,}"
        )

    # Missing values in y_prob
    n_null = pred_df["y_prob"].isnull().sum()
    if n_null > 0:
        errors.append(f"y_prob has {n_null:,} missing values")

    # Probability range
    if (pred_df["y_prob"] < 0).any() or (pred_df["y_prob"] > 1).any():
        out_of_range = ((pred_df["y_prob"] < 0) | (pred_df["y_prob"] > 1)).sum()
        errors.append(f"y_prob has {out_of_range:,} values outside [0, 1]")

    # date × country key match
    test_keys = set(
        zip(pd.to_datetime(test_df[DATE_COL]).dt.strftime("%Y-%m-%d"), test_df["country"])
    )
    pred_keys = set(zip(pred_df["date"], pred_df["country"]))
    if test_keys != pred_keys:
        n_missing = len(test_keys - pred_keys)
        n_extra = len(pred_keys - test_keys)
        errors.append(
            f"date×country mismatch — {n_missing:,} missing, {n_extra:,} extra keys"
        )

    if errors:
        raise ValueError("Validation FAILED:\n" + "\n".join(f"  • {e}" for e in errors))

    print("Validation passed:")
    print(f"  rows      : {len(pred_df):,}")
    print(f"  y_prob    : [{pred_df['y_prob'].min():.6f}, {pred_df['y_prob'].max():.6f}]")
    print(f"  date range: {pred_df['date'].min()} ~ {pred_df['date'].max()}")
