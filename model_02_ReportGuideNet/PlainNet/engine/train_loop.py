from typing import Dict
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from utils.common import AverageMeter
from utils.metrics import compute_binary_metrics_from_probs


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    amp: bool = True,
    pos_weight: torch.Tensor = None,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    scaler = GradScaler(enabled=amp)

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), n=x.size(0))

    return {"loss": loss_meter.avg}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    prob_all, y_all = [], []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits = model(x)
        probs = torch.sigmoid(logits)
        prob_all.append(probs.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())

    y_prob = np.concatenate(prob_all, axis=0).reshape(-1)
    y_true = np.concatenate(y_all, axis=0).reshape(-1)
    metrics = compute_binary_metrics_from_probs(y_true, y_prob)

    return metrics
