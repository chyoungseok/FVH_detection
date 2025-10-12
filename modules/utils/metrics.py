from typing import Dict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_recall_curve


def compute_binary_metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    out = {}
    if len(np.unique(y_true)) == 1:
        out["auc"] = float("nan")
        out["ap"]  = float("nan")
    else:
        out["auc"] = roc_auc_score(y_true, y_prob)
        out["ap"]  = average_precision_score(y_true, y_prob)

    y_pred_05 = (y_prob >= 0.5).astype(np.int32)
    out["acc@0.5"] = accuracy_score(y_true, y_pred_05)
    out["f1@0.5"]  = f1_score(y_true, y_pred_05, zero_division=0)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    denom = (precision + recall)
    denom[denom == 0] = 1e-9
    f1 = 2 * precision * recall / denom
    best_idx = int(np.argmax(f1))
    out["f1_best"] = float(f1[best_idx])
    out["thr_best"] = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    return out
