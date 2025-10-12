# test.py
import argparse
import os, cv2
import yaml
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from tqdm.auto import tqdm   # ✅ progress bar 추가

from modules.datasets.flair2d_binary import FlairH5SliceDataset, compute_pos_weight_from_dataset
from modules.models.plainnet import resnet, resnet_transformer
from modules.models.vit import ViTBinaryClassifier
from swin_transformer_pytorch import SwinTransformer
from modules.engine.train_loop import evaluate
from modules.utils.common import seed_everything, is_main_process

# -----------------------------
# 헬퍼 함수
# -----------------------------
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


# -----------------------------
# main
# -----------------------------
def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["train"].get("gpus", "0"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg["train"].get("seed", 42))

    # --- Dataset (test split) ---
    dcfg = cfg["data"]
    test_ds = FlairH5SliceDataset(
        root_dir=dcfg.get("root_dir", ""),
        h5_file=dcfg.get("h5_file", "flair_slice_dataset.h5"),
        split="test",
        seed=cfg["train"].get("seed", 42),
        stratified_split=dcfg.get("stratified_split", True),
        undersample=dcfg.get("undersample", False),
        undersample_ratio=dcfg.get("undersample_ratio", 1.0),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=max(1, cfg["train"]["batch_size"] // 2),
        sampler=SequentialSampler(test_ds),
        num_workers=dcfg.get("num_workers", 4),
        pin_memory=True
    )

    is_gradCam = False
    # --- Model ---
    mcfg = cfg["model"]
    if mcfg.get("swin", {}).get("enabled", False):
        swin_cfg = mcfg.get("swin", {})
        model = SwinTransformer(
            hidden_dim=swin_cfg.get("hidden_dim", 96),
            layers=swin_cfg.get("layers", [2, 2, 6, 2]),
            heads=swin_cfg.get("heads", [3, 6, 12, 24]),
            channels=swin_cfg.get("channels", 1),
            num_classes=swin_cfg.get("num_classes", 1),
            head_dim=swin_cfg.get("head_dim", 32),
            window_size=swin_cfg.get("window_size", 7),
            downscaling_factors=swin_cfg.get("downscaling_factors", [4, 2, 2, 2]),
            relative_pos_embedding=swin_cfg.get("relative_pos_embedding", True)
        )
    elif mcfg.get("vit", {}).get("enabled", False):
        model = ViTBinaryClassifier(pretrained=mcfg.get("pretrained", True))
    elif mcfg.get("transformer", {}).get("enabled", False):
        model = resnet_transformer(
            backbone=mcfg.get("backbone", "resnet18"),
            in_channels=mcfg.get("in_channels", 1),
            pretrained=mcfg.get("pretrained", False),
            transformer_cfg=mcfg.get("transformer", {})
        )
    else:
        model = resnet(
            backbone=mcfg.get("backbone", "resnet18"),
            in_channels=mcfg.get("in_channels", 1),
            pretrained=mcfg.get("pretrained", False),
        )
        is_gradCam = True

    model = model.to(device)

    # --- Load checkpoint ---
    ckpt_path = os.path.join(cfg["out"]["output_dir"], "best.pth")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    print(f"[INFO] Loaded checkpoint: {ckpt_path} (epoch {ckpt['epoch']+1})")

    # --- pos_weight ---
    pos_weight = None
    if cfg["train"].get("pos_weight_auto", True):
        pos_weight = compute_pos_weight_from_dataset(test_ds)

    # --- Evaluate ---
    metrics = evaluate(
        model, test_loader, device,
        pos_weight=pos_weight,
        type_loss=cfg["train"].get("loss", "bce"),
        return_preds=True
    )

    if is_main_process():
        print("\n===== TEST SET RESULTS =====")
        for k, v in metrics.items():
            if isinstance(v, (float, int)):
                print(f"{k}: {v:.4f}")

        y_true, y_pred, y_prob = metrics["y_true"], metrics["y_pred"], metrics["y_prob"]
        outdir = cfg["out"]["output_dir"]
        save_confusion_matrix(y_true, y_pred, outdir)
        save_roc_curve(y_true, y_prob, outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)