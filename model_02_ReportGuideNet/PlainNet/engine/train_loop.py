from typing import Dict
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from utils.common import AverageMeter
from utils.metrics import compute_binary_metrics_from_probs
from utils.losses import FocalLoss  # utils/losses.py에 정의한다고 가정

def get_criterion(type_loss, device, pos_weight=None):
    loss_name = type_loss.lower()
    if loss_name == "focal":
        return FocalLoss(alpha=1.0, gamma=2.0)
    else:
        return torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.to(device) if pos_weight is not None else None
            )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    amp: bool = True,
    pos_weight: torch.Tensor = None,
    type_loss: str = "bce" 
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    prob_all, y_all = [], []
    
    criterion = get_criterion(type_loss=type_loss, device=device, pos_weight=pos_weight)
    scaler = GradScaler(enabled=amp)

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            logits = model(x)
            
            # --- 자동 shape 맞추기 ---
            if logits.ndim == 2 and logits.size(1) == 1:
                logits = logits.view(-1)     # (B,1) → (B,)
            if y.ndim == 2 and y.size(1) == 1:
                y = y.view(-1)               # (B,1) → (B,)
            
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), n=x.size(0))

        probs = torch.sigmoid(logits)
        prob_all.append(probs.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())

    # --- 전체 epoch metric 계산 ---
    y_prob = np.concatenate(prob_all, axis=0).reshape(-1)
    y_true = np.concatenate(y_all, axis=0).reshape(-1)
    metrics = compute_binary_metrics_from_probs(y_true, y_prob)
    metrics["loss"] = loss_meter.avg

    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device, pos_weight=None, type_loss="bce", return_preds=False):
    model.eval()
    y_true, y_prob = [], []

    criterion = get_criterion(type_loss=type_loss, device=device, pos_weight=pos_weight)

    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            logits = model(x)
            
            # --- 자동 shape 맞추기 ---
            if logits.ndim == 2 and logits.size(1) == 1:
                logits = logits.view(-1)     # (B,1) → (B,)
            if y.ndim == 2 and y.size(1) == 1:
                y = y.view(-1)               # (B,1) → (B,)
            
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(y.cpu().numpy().tolist())

    # numpy 변환
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # --- threshold 최적화 ---
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    # F1 최적 threshold 탐색
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.1, 0.9, 17):
        y_pred_thr = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred_thr, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    y_pred_final = (y_prob >= best_thr).astype(int)

    metrics = {
        "loss": total_loss / len(dataloader.dataset),
        "auc": auc,
        "ap": ap,
        "f1@0.5": f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0),
        "f1_best": best_f1,
        "thr_best": best_thr,
        "acc@0.5": accuracy_score(y_true, (y_prob >= 0.5).astype(int)),
    }

    # --- return_preds 옵션 처리 ---
    if return_preds:
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred_final
        metrics["y_prob"] = y_prob

    return metrics