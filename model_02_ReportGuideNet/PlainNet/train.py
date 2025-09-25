import argparse
import os
import yaml

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler

from datasets.flair2d_binary import FlairNPYSliceDataset, compute_pos_weight_from_dataset
from models.plainnet import PlainNet
from engine.train_loop import train_one_epoch, evaluate
from utils.common import seed_everything, setup_distributed, cleanup_distributed, is_main_process, save_checkpoint


def build_scheduler(optimizer, cfg_train):
    name = (cfg_train.get("scheduler", "none") or "none").lower()
    if name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=cfg_train["epochs"])
    elif name == "step":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=cfg_train.get("step_size", 20), gamma=cfg_train.get("gamma", 0.1))
    else:
        return None


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["train"].get("seed", 42))

    setup_distributed(cfg["ddp"].get("backend", "nccl"))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    dcfg = cfg["data"]
    train_ds = FlairNPYSliceDataset(
        root_dir=dcfg.get("root_dir", ""),
        split="train",
    )

    val_ds = FlairNPYSliceDataset(
        root_dir=dcfg.get("root_dir", ""),
        split="val",
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
    val_sampler   = DistributedSampler(val_ds, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], sampler=train_sampler,
        num_workers=dcfg.get("num_workers", 4), pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, cfg["train"]["batch_size"] // 2), sampler=val_sampler,
        num_workers=dcfg.get("num_workers", 4), pin_memory=True, drop_last=False,
    )

    mcfg = cfg["model"]
    model = PlainNet(
        backbone=mcfg.get("backbone", "resnet18"),
        in_channels=mcfg.get("in_channels", 1),
        pretrained=mcfg.get("pretrained", False),
    ).to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    tcfg = cfg["train"]
    opt_name = (tcfg.get("optimizer","adamw") or "adamw").lower()
    if opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=tcfg["lr"], momentum=0.9, weight_decay=tcfg["weight_decay"])
    elif opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    scheduler = build_scheduler(optimizer, tcfg)

    pos_weight = None
    if tcfg.get("pos_weight_auto", True):
        pos_weight = compute_pos_weight_from_dataset(train_ds)

    outdir = cfg["out"]["output_dir"]
    if is_main_process():
        os.makedirs(outdir, exist_ok=True)

    best_auc = -1.0
    epochs = tcfg["epochs"]
    amp = bool(tcfg.get("amp", True))

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)

        tr = train_one_epoch(model, train_loader, optimizer, device, epoch, amp=amp, pos_weight=pos_weight)

        val_metrics = evaluate(model, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        if is_main_process():
            print(f"[Epoch {epoch+1:03d}/{epochs}] "
                  f"train_loss={tr['loss']:.4f}  "
                  f"val_auc={val_metrics.get('auc', float('nan')):.4f}  "
                  f"val_ap={val_metrics.get('ap', float('nan')):.4f}  "
                  f"f1_best={val_metrics.get('f1_best', float('nan')):.4f}  "
                  f"thr_best={val_metrics.get('thr_best', 0.5):.3f}")

            save_checkpoint({
                "model": model.module.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "cfg": cfg,
            }, os.path.join(outdir, "last.pth"))

            auc_score = val_metrics.get("auc", -1.0)
            if auc_score is not None and auc_score > best_auc:
                best_auc = auc_score
                save_checkpoint({
                    "model": model.module.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "cfg": cfg,
                }, os.path.join(outdir, "best.pth"))

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args)
