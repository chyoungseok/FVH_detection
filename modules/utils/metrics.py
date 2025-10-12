from typing import Dict
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score
)

def compute_binary_metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    out = {}

    # --- ROC, Pprecision ---
    if len(np.unique(y_true)) == 1:
        out["auc"] = float("nan")
        out["precision"]  = float("nan")
    else:
        out["auc"] = roc_auc_score(y_true, y_prob)
        out["precision"]  = average_precision_score(y_true, y_prob)

    # --- Threshold = 0.5 ---
    y_pred_05 = (y_prob >= 0.5).astype(np.int32)
    out["acc"]     = accuracy_score(y_true, y_pred_05)
    out["f1"]      = f1_score(y_true, y_pred_05, zero_division=0)
    out["precision"] = precision_score(y_true, y_pred_05, zero_division=0)
    out["recall"]    = recall_score(y_true, y_pred_05, zero_division=0)

    # --- Precision-Recall curve ---
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    denom = (precision + recall)
    denom[denom == 0] = 1e-9
    f1 = 2 * precision * recall / denom

    # --- F1 최댓값 기준 ---
    best_idx = int(np.argmax(f1))
    out["f1_best"] = float(f1[best_idx])
    out["thr_best"] = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    # --- 최적 threshold 시 precision, recall 저장 ---
    out["precision_best"] = float(precision[best_idx])
    out["recall_best"] = float(recall[best_idx])

    return out