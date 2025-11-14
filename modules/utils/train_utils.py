import os, csv, math
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from swin_transformer_pytorch import SwinTransformer
from modules.models.vit import ViTBinaryClassifier
from modules.models.resnet import resnet
from modules.models.vitgru import ViTGRU

# -- Model Selection ---
def ModelSelection(cfg, device):
    mcfg = cfg["model"]
    
    # -- Swin Transformer
    if mcfg.get("swin", {}).get("enabled", False):
        swin_cfg = mcfg["swin"]
        model = SwinTransformer(
            hidden_dim=swin_cfg.get("hidden_dim", 96),
            layers=swin_cfg.get("layers", [2, 2, 6, 2]),
            heads=swin_cfg.get("heads", [3, 6, 12, 24]),
            channels=swin_cfg.get("channels", 1),
            num_classes=swin_cfg.get("num_classes", 2),
            head_dim=swin_cfg.get("head_dim", 32),
            window_size=swin_cfg.get("window_size", 7),
            downscaling_factors=swin_cfg.get("downscaling_factors", [4, 2, 2, 2]),
            relative_pos_embedding=swin_cfg.get("relative_pos_embedding", True)
        ).to(device)
        model_name = "swin"
    
    # -- Vision Transformer
    elif mcfg.get("vit", {}).get("enabled", False):
        vit_cfg = mcfg["vit"]
        model = ViTBinaryClassifier(pretrained=vit_cfg.get("pretrained", True)).to(device)
        model_name = "vit"
    
    # -- Resnet
    elif mcfg.get("resnet", {}).get("enabled", False):
        resnet_cfg = mcfg["resnet"]
        model = resnet(
            backbone=resnet_cfg.get("backbone", "resnet18"),
            in_channels=resnet_cfg.get("in_channels", 1),
            pretrained=resnet_cfg.get("pretrained", False)
        ).to(device)
        model_name = "resnet"
        
    # --- ViT-GRU
    elif mcfg.get("vitgru", {}).get("enabled", False):
        vcfg = mcfg["vitgru"]
        model = ViTGRU(
            in_channels=vcfg.get("in_channels", 1),
            img_size=vcfg.get("img_size", 672),
            patch_size=vcfg.get("patch_size", 8),       # 논문 설정
            embed_dim=vcfg.get("embed_dim", 64),        # 논문 설정
            num_heads=vcfg.get("num_heads", 4),
            num_layers=vcfg.get("num_layers", 8),
            gru_hidden=vcfg.get("gru_hidden", 1024),
            bidirectional=vcfg.get("bidirectional", False),
            dropout=vcfg.get("dropout", 0.3),
            num_classes=vcfg.get("num_classes", 1)
        ).to(device)
        model_name = "vitgru"

    return model, model_name

# --- Optimizer 설정 ---
def build_optimizer(cfg, model):
    tcfg = cfg["train"]
    opt_name = (tcfg.get("optimizer", "adamw") or "adamw").lower()
    if opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=tcfg["lr"],
                              momentum=0.9, weight_decay=tcfg["weight_decay"])
    elif opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=tcfg["lr"],
                               weight_decay=tcfg["weight_decay"])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=tcfg["lr"],
                                weight_decay=tcfg["weight_decay"])
        
    return optimizer
        

# --- Scheduler 설정 ---
def build_scheduler(cfg, optimizer):
    tcfg = cfg["train"]
    name = (tcfg.get("scheduler", "none") or "none").lower()

    if name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])

    elif name == "step":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(
            optimizer,
            step_size=tcfg.get("step_size", 20),
            gamma=tcfg.get("gamma", 0.1)
        )

    elif name == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=tcfg.get("factor", 0.5),
            patience=tcfg.get("patience", 5),
            verbose=True
        )

    return None

# --- outdir initialization ---
def init_outdir(cfg):
    outdir = cfg["out"]["output_dir"]
    os.makedirs(outdir, exist_ok=True)
    return outdir


# --- csv initialization ---
def init_csv(outdir):
    metrics_file = os.path.join(outdir, "metrics.csv")
    if not os.path.exists(metrics_file):
        with open(metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss", "train_acc", "train_precision", "train_recall", "train_auc", "train_f1", "train_f1_best",
                "val_loss",   "val_acc",   "val_precision",   "val_recall",   "val_auc",   "val_f1",   "val_f1_best"
            ])
    return metrics_file

# --- Step scheduler
def step_scheduler(scheduler, val_metrics):
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics["loss"])
        else:
            scheduler.step()
         
# --- Logger    
def log_epoch_metrics(epoch: int,
                      epochs: int,
                      tr_metrics: dict,
                      val_metrics: dict):
    """
    각 epoch마다 학습 및 검증 결과를 콘솔에 출력하는 함수.

    Args:
        epoch (int): 현재 epoch index (0-based)
        epochs (int): 전체 epoch 수
        tr_metrics (dict): train_one_epoch() 결과 딕셔너리
        val_metrics (dict): evaluate() 결과 딕셔너리
    """
    print(
        f"[Epoch {epoch+1:03d}/{epochs}] "
        f"\n"
        f"train_loss={tr_metrics.get('loss', float('nan')):.4f}  "
        f"train_acc={tr_metrics.get('acc', float('nan')):.4f}  "
        f"train_precision={tr_metrics.get('precision', float('nan')):.4f}  "
        f"train_recall={tr_metrics.get('recall', float('nan')):.4f}  "
        f"train_auc={tr_metrics.get('auc', float('nan')):.4f}  "
        f"train_f1={tr_metrics.get('f1', float('nan')):.4f}  "
        f"train_f1_best={tr_metrics.get('f1_best', float('nan')):.4f}  "
        f"\n"
        f"  val_loss={val_metrics.get('loss', float('nan')):.4f}  "
        f"  val_acc={val_metrics.get('acc', float('nan')):.4f}  "
        f"  val_precision={val_metrics.get('precision', float('nan')):.4f}  "
        f"  val_recall={val_metrics.get('recall', float('nan')):.4f}  "
        f"  val_auc={val_metrics.get('auc', float('nan')):.4f}  "
        f"  val_f1={val_metrics.get('f1', float('nan')):.4f}  "
        f"  val_f1_best={val_metrics.get('f1_best', float('nan')):.4f}  "
    )
    

def update_metrics_csv(metrics_file: str,
                       epoch: int,
                       tr_metrics: dict,
                       val_metrics: dict):
    """
    학습 및 검증 결과를 metrics.csv 파일에 추가하는 함수.

    Args:
        metrics_file (str): CSV 파일 경로
        epoch (int): 현재 epoch (0-based)
        tr_metrics (dict): train_one_epoch() 결과
        val_metrics (dict): evaluate() 결과
    """
    # NaN-safe 변환 함수
    def safe_val(value, default=float("nan")):
        return value if value is not None and not (isinstance(value, float) and math.isnan(value)) else default

    row = [
        epoch + 1,
        safe_val(tr_metrics.get("loss")),
        safe_val(tr_metrics.get("acc")),
        safe_val(tr_metrics.get("precision")),
        safe_val(tr_metrics.get("recall")),
        safe_val(tr_metrics.get("auc")),
        safe_val(tr_metrics.get("f1")),
        safe_val(tr_metrics.get("f1_best")),
        
        safe_val(val_metrics.get("loss")),
        safe_val(val_metrics.get("acc")),
        safe_val(val_metrics.get("precision")),
        safe_val(val_metrics.get("recall")),
        safe_val(val_metrics.get("auc")),
        safe_val(val_metrics.get("f1")),
        safe_val(val_metrics.get("f1_best")),
    ]

    with open(metrics_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        
# --- 학습 곡선 시각화 ---
def plot_curves(metrics_file, outdir):
    df = pd.read_csv(metrics_file)

    # Loss
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(outdir, "loss_curve.png"))
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(df["epoch"], df["train_acc"], label="Train Acc")
    plt.plot(df["epoch"], df["val_acc"], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(); plt.title("Training and Validation Accuracy")
    plt.savefig(os.path.join(outdir, "accuracy_curve.png"))
    plt.close()

    # F1
    plt.figure()
    plt.plot(df["epoch"], df["train_f1"], label="Train F1")
    plt.plot(df["epoch"], df["val_f1"], label="Val F1")
    plt.xlabel("Epoch"); plt.ylabel("F1 Score")
    plt.legend(); plt.title("Training and Validation F1 Score")
    plt.savefig(os.path.join(outdir, "f1_curve.png"))
    plt.close()