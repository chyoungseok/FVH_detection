import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import roc_curve, auc as sk_auc

def plot_training_curves(history: dict, out_dir: str):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    # Loss
    plt.figure(figsize=(7,5))
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss over epochs"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "curve_loss.png")); plt.close()
    
    # Accuracy (optional if present)
    if "train_acc" in history and "val_acc" in history:
        plt.figure(figsize=(7,5))
        plt.plot(epochs, history["train_acc"], label="train acc")
        plt.plot(epochs, history["val_acc"], label="val acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy over epochs"); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "curve_acc.png")); plt.close()
        
    # AUC over epochs (optional if present)
    if "train_auc" in history and "val_auc" in history and len(history["train_auc"]) == len(epochs):
        plt.figure(figsize=(7,5))
        plt.plot(epochs, history["train_auc"], label="train AUC")
        plt.plot(epochs, history["val_auc"], label="val AUC")
        plt.xlabel("Epoch"); plt.ylabel("AUC"); plt.title("AUC over epochs"); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "curve_auc.png")); plt.close()
        
    # LR over epochs (optional if present)
    if "lr" in history and len(history["lr"]) == len(epochs):
        plt.figure(figsize=(7,5))
        plt.plot(epochs, history["lr"], label="learning rate")
        plt.xlabel("Epoch"); plt.ylabel("LR"); plt.title("Learning rate over epochs")
        plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "curve_lr.png")); plt.close()
    
    print(f"[Saved] curves -> {os.path.join(out_dir, 'curve_loss.png')}, curve_acc.png, curve_auc.png, curve_lr.png")

def save_history_csv(history: dict, out_dir: str, fname: str = "history_metrics.csv"):
    """history 딕셔너리를 (epoch 단위 행) CSV로 저장."""
    os.makedirs(out_dir, exist_ok=True)
    n = len(history.get("train_loss", []))
    epochs = np.arange(1, n + 1)
    # 필요한 키들 목록
    keys = ["train_loss","val_loss","train_acc","val_acc","train_auc","val_auc","lr"]
    data = {"epoch": epochs}
    for k in keys:
        v = history.get(k, [])
        # 길이 맞추기 (없으면 NaN)
        if len(v) < n:
            vv = list(v) + [np.nan] * (n - len(v))
        else:
            vv = v
        data[k] = vv
    pd.DataFrame(data).to_csv(os.path.join(out_dir, fname), index=False)

def save_eval_plots(y_true: np.ndarray, y_prob: np.ndarray, out_dir: str, prefix: str = "test", thr: float = 0.5):
    os.makedirs(out_dir, exist_ok=True)
    # ROC
    try:
        fpr, tpr, thr_roc = roc_curve(y_true, y_prob)
        roc_auc = sk_auc(fpr, tpr)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0,1],[0,1], linestyle="--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC curve")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_roc.png")); plt.close()
        # Save ROC raw data to CSV for Prism
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr_roc}).to_csv(
            os.path.join(out_dir, f"{prefix}_roc.csv"), index=False
        )
    except Exception:
        pass
    
    # PR curve
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(figsize=(6,6))
        plt.plot(rec, prec)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curve")
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_pr.png")); plt.close()
        # Save PR raw data to CSV for Prism (note: threshold length is len-1, pad with NaN)
        thr_pr = np.append(_, np.nan)  # `_` holds thresholds from precision_recall_curve
        pd.DataFrame({"recall": rec, "precision": prec, "threshold": thr_pr}).to_csv(
            os.path.join(out_dir, f"{prefix}_pr.csv"), index=False
        )
    except Exception:
        pass

    # Confusion matrix at chosen threshold
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    # Save CM as CSV with labels
    pd.DataFrame(cm, index=["Actual_Neg","Actual_Pos"], columns=["Pred_Neg","Pred_Pos"]).to_csv(
        os.path.join(out_dir, f"{prefix}_cm.csv")
    )

    # Save per-sample predictions for Prism or audit
    pd.DataFrame({
        "index": np.arange(len(y_true)),
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "threshold": np.full_like(y_true, fill_value=thr, dtype=float)
    }).to_csv(os.path.join(out_dir, f"{prefix}_preds.csv"), index=False)

    plt.figure(figsize=(4.5,4))
    plt.imshow(cm, cmap="Blues")
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xticks([0,1],["Neg","Pos"]); plt.yticks([0,1],["Neg","Pos"])
    plt.title(f"Confusion Matrix (thr={thr:.2f})"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_cm.png")); plt.close()
    