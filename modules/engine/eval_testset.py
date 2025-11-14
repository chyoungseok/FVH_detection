import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from modules.engine.train_loop import evaluate
from modules.visualization.gradcam_utils import GradCAM, generate_and_save_cam


# ============================================================
# 📊 1️⃣ Confusion Matrix & ROC Curve & Metrics CSV
# ============================================================
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


def save_test_metrics_to_csv(metrics: dict, outdir: str):
    """metrics 딕셔너리에서 숫자형 항목만 추출해 CSV로 저장"""
    numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (float, int))}
    df = pd.DataFrame([numeric_metrics])
    out_path = os.path.join(outdir, "test_metrics.csv")
    df.to_csv(out_path, index=False)
    print(f"[INFO] Test metrics saved to: {out_path}")


# ============================================================
# 🔥 2️⃣ Grad-CAM Application
# ============================================================
def apply_gradcam_on_testset(model, test_loader, device, cfg):
    """
    모든 테스트 이미지(batch 포함)에 대해 Grad-CAM을 생성하고 저장
    """
    model.eval()
    gradcam_outdir = os.path.join(cfg["out"]["output_dir"], "gradCAM")
    os.makedirs(gradcam_outdir, exist_ok=True)

    # --- Try to locate target layer automatically ---
    if isinstance(model, torch.nn.DataParallel):
        backbone = getattr(model.module, "backbone", None)
    else:
        backbone = getattr(model, "backbone", None)

    if backbone is None:
        print("[WARN] Grad-CAM not supported: model has no backbone attribute.")
        return

    target_layer = getattr(backbone, "layer4", None)
    if target_layer is None:
        print("[WARN] Grad-CAM not supported for this model type.")
        return

    gradcam = GradCAM(model.module if isinstance(model, torch.nn.DataParallel) else model,
                      target_layer[-1])

    print(f"[INFO] Generating Grad-CAM results to: {gradcam_outdir}")

    for batch in test_loader:
        imgs = batch["image"].to(device)      # [B, 1, H, W]
        labels = batch["label"].cpu().numpy() # [B]
        sids = batch["sid"]

        # 배치 전체를 하나씩 순회
        for i in range(imgs.size(0)):
            img = imgs[i].unsqueeze(0)        # [1, 1, H, W]
            label = int(labels[i])
            sid = sids[i]

            # 예측 수행
            logits = model(img)
            prob = torch.sigmoid(logits).item()
            pred = int(prob >= 0.5)

            # Grad-CAM 생성 및 저장
            cam_path = generate_and_save_cam(
                model=model.module if isinstance(model, torch.nn.DataParallel) else model,
                gradcam=gradcam,
                img_tensor=img,
                raw_img_np=batch["image"][i].squeeze().numpy(),
                sid=sid,
                gt_label=label,
                pred_label=pred,
                save_dir=gradcam_outdir
            )
            print(f"[CAM] Saved: {os.path.basename(cam_path)}")

    print(f"[INFO] All Grad-CAM results saved under: {gradcam_outdir}")




# ============================================================
# 🧪 3️⃣ Main evaluation entrypoint
# ============================================================
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
    save_test_metrics_to_csv(metrics, outdir)

    # --- Apply Grad-CAM on all test samples ---
    apply_gradcam_on_testset(model, test_loader, device, cfg)
