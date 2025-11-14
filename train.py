import os
# --- CPU 스레드 제한 (과부하 방지) ---
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import torch
torch.set_num_threads(2)

import argparse, yaml

from modules.datasets.flair2d_binary import get_dataset, get_dataloader, FlairH5SliceDataset, compute_pos_weight_from_dataset
from modules.utils.train_utils import *
from modules.engine.train_loop import train_one_epoch, evaluate
from modules.engine.eval_testset import run_eval_on_testset
from modules.utils.common import seed_everything, save_checkpoint, str2bool

# --- 메인 루프 ---
def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["train"].get("gpus", "0"))
    seed_everything(cfg["train"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset / DataLoader ---
    train_ds, val_ds, test_ds = get_dataset(cfg)
    train_loader, val_loader, test_loader = get_dataloader(cfg, train_ds, val_ds, test_ds)

    # --- Class imbalance weight ---
    pos_weight = compute_pos_weight_from_dataset(train_ds) if cfg["train"].get("pos_weight_auto", True) else None

    # --- Model Selection ---
    model = ModelSelection(cfg, device=device)

    # --- Optimizer / Scheduler ---
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    # --- Output directory ---
    outdir = init_outdir(cfg)

    # --- Metric csv init. ---
    metrics_file = init_csv(outdir)

    # --- Training setup ---
    best_auc = -1.0
    tcfg = cfg["train"]
    epochs = tcfg["epochs"]
    amp = bool(tcfg.get("amp", True))
    type_loss = tcfg.get("loss", 'bce')

    # ==========================================================
    # Resume checkpoint (auto-detect last.pth if resume_ckpt not given)
    # ==========================================================
    resume_path = cfg["out"].get("resume_ckpt", None)
    start_epoch = 0

    # 자동 탐지 추가
    if not resume_path:
        candidate = os.path.join(outdir, "last.pth")
        if os.path.exists(candidate):
            resume_path = candidate
            print(f"[INFO] Auto-detected resume checkpoint: {resume_path}")

    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"[INFO] Resumed model weights from {resume_path}")

        # Optimizer / Scheduler state 복원
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print(f"[INFO] Optimizer state restored.")
        if "scheduler" in ckpt and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
            print(f"[INFO] Scheduler state restored.")

        start_epoch = ckpt.get("epoch", 0) + 1
        best_auc = ckpt.get("val_metrics", {}).get("auc", -1.0)
        print(f"[INFO] Resuming training from epoch {start_epoch} with best AUC = {best_auc:.4f}")

    # ==========================================================
    # Training loop
    # ==========================================================
    if not args.only_test:
        for epoch in range(start_epoch, epochs):
            tr_metrics = train_one_epoch(
                model, train_loader, optimizer, device, epoch,
                amp=amp, pos_weight=pos_weight, type_loss=type_loss
            )

            val_metrics = evaluate(
                model, val_loader, device,
                pos_weight=pos_weight, type_loss=type_loss
            )

            # Scheduler step
            step_scheduler(scheduler=scheduler, val_metrics=val_metrics)

            # Logging
            log_epoch_metrics(epoch, epochs, tr_metrics, val_metrics)

            # Save metrics per epoch
            update_metrics_csv(metrics_file, epoch, tr_metrics, val_metrics)

            # Plot metrics after last epoch
            if epoch == epochs - 1:
                plot_curves(metrics_file, outdir)

            # Save checkpoint
            ckpt_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "cfg": cfg,
            }
            save_checkpoint(ckpt_dict, os.path.join(outdir, "last.pth"))

            # Save best model
            auc_score = val_metrics.get("auc", -1.0)
            if auc_score is not None and auc_score > best_auc:
                best_auc = auc_score
                save_checkpoint(ckpt_dict, os.path.join(outdir, "best.pth"))
                print(f"[INFO] New best AUC: {best_auc:.4f} at epoch {epoch}")

    # ==========================================================
    # Evaluation on test dataset
    # ==========================================================
    pos_weight_testset = compute_pos_weight_from_dataset(test_ds) if cfg["train"].get("pos_weight_auto", True) else None
    run_eval_on_testset(cfg=cfg,
                        model=model,
                        device=device,
                        pos_weight=pos_weight_testset,
                        test_loader=test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--only_test", type=str2bool, default=False)
    args = parser.parse_args()
    main(args)