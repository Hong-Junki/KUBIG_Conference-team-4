import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def compute_pr_auc(y_true, y_prob):
    return float(average_precision_score(y_true, y_prob))


def compute_p_at_top_k(y_true, y_prob, k=0.05):
    """Precision among the top-k% highest-scored samples."""
    n = len(y_true)
    top_k = max(1, int(np.ceil(n * k)))
    top_idx = np.argsort(y_prob)[::-1][:top_k]
    return float(np.mean(y_true[top_idx]))


def compute_recall_at_precision(y_true, y_prob, min_precision=0.10):
    """Maximum recall achievable while keeping precision >= min_precision."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve appends (1.0, 0.0) as the last point (no threshold).
    # Include it: precision=1.0 satisfies any min_precision, but recall=0.0 there.
    valid = precision >= min_precision
    if not valid.any():
        return 0.0
    return float(recall[valid].max())


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (equal-width bins)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Last bin is closed on the right
        if hi == 1.0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += mask.sum() / n * abs(bin_conf - bin_acc)
    return float(ece)


def compute_all_metrics(y_true, y_prob):
    """Compute all evaluation metrics and return as a dict."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return {
        "pr_auc": compute_pr_auc(y_true, y_prob),
        "p_at_top5pct": compute_p_at_top_k(y_true, y_prob, k=0.05),
        "recall_at_precision_010": compute_recall_at_precision(
            y_true, y_prob, min_precision=0.10
        ),
        "ece": compute_ece(y_true, y_prob),
        "positive_rate": float(y_true.mean()),
        "n_samples": int(len(y_true)),
    }
