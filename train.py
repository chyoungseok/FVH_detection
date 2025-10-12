import os
# --- CPU 스레드 제한 (과부하 방지) ---
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import torch
import torch
torch.set_num_threads(2)

import argparse
import yaml
import csv

import pandas as pd
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from modules.datasets.flair2d_binary import FlairNPYSliceDataset, compute_pos_weight_from_dataset, FlairH5SliceDataset
from modules.models.plainnet import resnet, resnet_transformer
from modules.models.vit import ViTBinaryClassifier
from modules.engine.train_loop import train_one_epoch, evaluate
from modules.utils.common import (
    seed_everything, setup_distributed, cleanup_distributed,
    is_main_process, save_checkpoint
)
from swin_transformer_pytorch import SwinTransformer


def build_scheduler(optimizer, cfg_train):
    name = (cfg_train.get("scheduler", "none") or "none").lower()

    if name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=cfg_train["epochs"])

    elif name == "step":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(
            optimizer,
            step_size=cfg_train.get("step_size", 20),
            gamma=cfg_train.get("gamma", 0.1)
        )

    elif name == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(
            optimizer,
            mode="min",                  # validation loss 최소화 기준
            factor=cfg_train.get("factor", 0.5),
            patience=cfg_train.get("patience", 5),
            verbose=True
        )

    return None

def plot_curves(metrics_file, outdir):
    df = pd.read_csv(metrics_file)

    # 1) Loss plot
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(outdir, "loss_curve.png"))
    plt.close()

    # 2) Accuracy plot
    plt.figure()
    plt.plot(df["epoch"], df["train_acc"], label="Train Acc")
    plt.plot(df["epoch"], df["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.savefig(os.path.join(outdir, "accuracy_curve.png"))
    plt.close()

    # 3) F1 score plot
    plt.figure()
    plt.plot(df["epoch"], df["train_f1"], label="Train F1")
    plt.plot(df["epoch"], df["val_f1"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.title("Training and Validation F1 Score")
    plt.savefig(os.path.join(outdir, "f1_curve.png"))
    plt.close()

def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # GPU 선택 (config에서 읽음)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["train"].get("gpus", "0"))

    seed_everything(cfg["train"].get("seed", 42))

    # --- DDP 여부 확인 ---
    ddp_enabled = cfg.get("ddp", {}).get("enabled", True)

    if ddp_enabled:
        setup_distributed(cfg["ddp"].get("backend", "nccl"))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset ---
    dcfg = cfg["data"]

    # --- Dataset ---
    dcfg = cfg["data"]

    train_ds = FlairH5SliceDataset(
        root_dir=dcfg.get("root_dir", ""),
        split="train",
        seed=cfg["train"].get("seed", 42),
        stratified_split=dcfg.get("stratified_split", True),
        undersample=dcfg.get("undersample", True),
        undersample_ratio=dcfg.get("undersample_ratio", 1.0)   # ✅ ratio 전달
    )
    val_ds = FlairH5SliceDataset(
        root_dir=dcfg.get("root_dir", ""),
        split="val",
        seed=cfg["train"].get("seed", 42),
        stratified_split=dcfg.get("stratified_split", True),
        undersample=dcfg.get("undersample", True),
        undersample_ratio=dcfg.get("undersample_ratio", 1.0)   # ✅ ratio 전달
    )
    test_ds = FlairH5SliceDataset(
        root_dir=dcfg.get("root_dir", ""),
        split="test",
        seed=cfg["train"].get("seed", 42),
        stratified_split=dcfg.get("stratified_split", True),
        undersample=dcfg.get("undersample", True),
        undersample_ratio=dcfg.get("undersample_ratio", 1.0)   # ✅ ratio 전달
    )
    
    
    if ddp_enabled:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(test_ds, shuffle=False, drop_last=False)
    else:
        train_sampler = RandomSampler(train_ds, replacement=False)
        val_sampler = SequentialSampler(val_ds)
        test_sampler = SequentialSampler(test_ds)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], sampler=train_sampler,
        num_workers=dcfg.get("num_workers", 4), pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, cfg["train"]["batch_size"] // 2), sampler=val_sampler,
        num_workers=dcfg.get("num_workers", 4), pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=max(1, cfg["train"]["batch_size"] // 2), sampler=test_sampler,
        num_workers=dcfg.get("num_workers", 4), pin_memory=True
    )

    # --- Model ---
    mcfg = cfg["model"]
    if mcfg.get("swin", {}).get("enabled", False):
        swin_cfg = mcfg.get("swin", {})
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
        
    elif mcfg.get("vit", {}).get("enabled", False):
        model = ViTBinaryClassifier(pretrained=mcfg.get("pretrained", True)).to(device)
        
    elif mcfg.get("transformer", {}).get("enabled", False):
        model = resnet_transformer(
            backbone=mcfg.get("backbone", "resnet18"),
            in_channels=mcfg.get("in_channels", 1),
            pretrained=mcfg.get("pretrained", False),
            transformer_cfg=mcfg.get("transformer", {})
        ).to(device)
        
    else:
        model = resnet(
            backbone=mcfg.get("backbone", "resnet18"),
            in_channels=mcfg.get("in_channels", 1),
            pretrained=mcfg.get("pretrained", False),
        ).to(device)

    if ddp_enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # --- Optimizer / Scheduler ---
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

    scheduler = build_scheduler(optimizer, tcfg)

    # --- Class imbalance weight ---
    pos_weight = None
    if tcfg.get("pos_weight_auto", True):
        pos_weight = compute_pos_weight_from_dataset(train_ds)

    # --- Output dir ---
    outdir = cfg["out"]["output_dir"]
    if is_main_process() or not ddp_enabled:
        os.makedirs(outdir, exist_ok=True)

    # --- CSV logger 준비 ---
    metrics_file = os.path.join(outdir, "metrics.csv")
    if (is_main_process() or not ddp_enabled) and not os.path.exists(metrics_file):
        with open(metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss", "train_acc", "train_f1", "train_f1_best",
                "val_loss", "val_acc", "val_auc", "val_ap",
                "val_f1", "val_f1_best", "val_thr_best"
            ])

    # --- Training loop ---
    best_auc = -1.0
    epochs = tcfg["epochs"]
    amp = bool(tcfg.get("amp", True))
    type_loss = tcfg.get("loss", 'bce')

    for epoch in range(epochs):
        if ddp_enabled:
            train_sampler.set_epoch(epoch)

        tr = train_one_epoch(model, train_loader, optimizer, device, epoch,
                             amp=amp, pos_weight=pos_weight, type_loss=type_loss)
        val_metrics = evaluate(model, val_loader, device, pos_weight=pos_weight, type_loss=type_loss)

        # --- Scheduler step ---
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

            if is_main_process() or not ddp_enabled:
                print(f"[Epoch {epoch+1:03d}/{epochs}] "
                    f"train_loss={tr['loss']:.4f}  "
                    f"train_acc={tr.get('acc@0.5', float('nan')):.4f}  "
                    f"train_f1={tr.get('f1@0.5', float('nan')):.4f}  "
                    f"train_f1_best={tr.get('f1_best', float('nan')):.4f}  "
                    f"val_loss={val_metrics['loss']:.4f}  "
                    f"val_acc={val_metrics.get('acc@0.5', float('nan')):.4f}  "
                    f"val_auc={val_metrics.get('auc', float('nan')):.4f}  "
                    f"val_ap={val_metrics.get('ap', float('nan')):.4f}  "
                    f"val_f1={val_metrics.get('f1@0.5', float('nan')):.4f}  "
                    f"val_f1_best={val_metrics.get('f1_best', float('nan')):.4f}  "
                    f"val_thr_best={val_metrics.get('thr_best', 0.5):.3f}")

                with open(metrics_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch+1,
                        tr.get("loss", float("nan")),
                        tr.get("acc@0.5", float("nan")),
                        tr.get("f1@0.5", float("nan")),
                        tr.get("f1_best", float("nan")),
                        val_metrics.get("loss", float("nan")),
                        val_metrics.get("acc@0.5", float("nan")),
                        val_metrics.get("auc", float("nan")),
                        val_metrics.get("ap", float("nan")),
                        val_metrics.get("f1@0.5", float("nan")),
                        val_metrics.get("f1_best", float("nan")),
                        val_metrics.get("thr_best", 0.5),
                    ])

                if epoch == epochs - 1:
                    plot_curves(metrics_file, outdir)

                ckpt_dict = {
                    "model": model.module.state_dict() if ddp_enabled else model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "cfg": cfg,
                }
                save_checkpoint(ckpt_dict, os.path.join(outdir, "last.pth"))

                auc_score = val_metrics.get("auc", -1.0)
                if auc_score is not None and auc_score > best_auc:
                    best_auc = auc_score
                    save_checkpoint(ckpt_dict, os.path.join(outdir, "best.pth"))

    if ddp_enabled:
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args)
