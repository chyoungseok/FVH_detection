import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from modules.engine.train_loop import evaluate

def save_confusion_matrix(y_true, y_pred, outdir):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["True Neg", "True Pos"],
        columns=["Pred Neg", "Pred Pos"]
    )

    cm_path = os.path.join(outdir, "confusion_matrix.csv")
    cm_df.to_csv(cm_path, index=True)
    print(f"[INFO] Confusion matrix saved at {cm_path}")

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred Neg", "Pred Pos"],
        yticklabels=["True Neg", "True Pos"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    png_path = os.path.join(outdir, "confusion_matrix.png")
    plt.savefig(png_path)
    plt.close()
    print(f"[INFO] Confusion matrix heatmap saved at {png_path}")
    

def save_roc_curve(y_true, y_prob, outdir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    png_path = os.path.join(outdir, "roc_curve.png")
    plt.savefig(png_path)
    plt.close()
    print(f"[INFO] ROC curve saved at {png_path}")

def run_eval_on_testset(cfg, model, device, pos_weight, test_loader):
    # --- Load checkpoint ---
    ckpt_path = os.path.join(cfg["out"]["output_dir"], "best.pth")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    print(f"[INFO] Loaded checkpoint: {ckpt_path} (epoch {ckpt['epoch']+1})")

    # --- Evaluate ---
    metrics = evaluate(model,
                       test_loader,
                       device,
                       pos_weight=pos_weight,
                       type_loss=cfg["train"].get("loss", "bce"),
                       return_preds=True
                       )

    print("\n===== TEST SET RESULTS =====")
    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            print(f"{k}: {v:.4f}")

    y_true, y_pred, y_prob = metrics["y_true"], metrics["y_pred"], metrics["y_prob"]
    outdir = cfg["out"]["output_dir"]
    save_confusion_matrix(y_true, y_pred, outdir)
    save_roc_curve(y_true, y_prob, outdir)