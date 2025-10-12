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

from modules.datasets.flair2d_binary import FlairH5SliceDataset, compute_pos_weight_from_dataset, load_dataset, _dataloader
from modules.models.resnet import resnet
from modules.models.vit import ViTBinaryClassifier
from swin_transformer_pytorch import SwinTransformer
from modules.engine.train_loop import evaluate
from modules.utils.common import seed_everything, is_main_process

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
    test_ds = load_dataset(cfg=cfg, split_type="test")
    test_loader = _dataloader(cfg, test_ds, "test")

    # --- Model ---
    # --- Model Selection
    model = ModelSelection(cfg, device=device)


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