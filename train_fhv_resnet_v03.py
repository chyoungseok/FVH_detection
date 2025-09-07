# train_fhv_resnet.py
"""
FHV(+) / FHV(-) 이진 분류 (slice-level) - ResNet18 통합 스크립트
- 데이터 트리: data/subjectX/{slice_img.npy, slice_label.npy}
- 입력: 각 subject의 slice_img.npy: (N_slices, 672, 672), slice_label.npy: (N_slices,) 또는 (N_slices,1)
- 출력: 학습된 모델(.pt), 테스트 리포트(.txt), 콘솔 메트릭

학습 전략
- subject-wise 정보는 사용하지 않음(요청사항 반영)
- slice-wise 전체 concat → 7:3 stratified split
- 클래스 불균형: 음성(0) 언더샘플링 (양성:음성 = 1:R)
- 손실: BCEWithLogitsLoss(+ pos_weight 옵션)
- 추론 함수: infer_one_slice() (확률 반환)
"""

import os
import glob
import math
import random
import argparse
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
# =========================================================
# Optional: Reduce shared memory pressure for >0 workers
# =========================================================
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

# =========================================================
# 유틸: 시드 고정
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# 데이터 로더: subject 트리 순회 (권장)
# root/subjectX/{slice_img.npy, slice_label.npy}
# =========================================================
def load_all_slices_from_tree(root_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    root_dir/subjectX/ 에서
      - 이미지: *_AxialSlices_padded.npy
      - 라벨 : *_label_sliceLevel.npy
    를 찾아 모두 concat하여 (N, 672, 672), (N,) 반환
    """
    import os, glob, numpy as np

    subj_dirs = sorted([p for p in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(p)])
    assert len(subj_dirs) > 0, f"No subject folders under {root_dir}"

    all_imgs, all_labs = [], []
    loaded_subjects = 0
    skipped = []

    for d in subj_dirs:
        img_hits = sorted(glob.glob(os.path.join(d, "*_AxialSlices_padded.npy")))
        lab_hits = sorted(glob.glob(os.path.join(d, "*_label_sliceLevel.npy")))

        if not img_hits or not lab_hits:
            skipped.append((os.path.basename(d), "missing *_AxialSlices_padded.npy or *_label_sliceLevel.npy"))
            continue

        # 보통 하나씩일 텐데, 여러 개 있으면 첫 번째 사용 (원하면 에러 처리로 바꿔도 됨)
        img_path = img_hits[0]
        lab_path = lab_hits[0]

        try:
            imgs = np.load(img_path)            # (n, 672, 672)
            labs = np.load(lab_path).squeeze()  # (n,) or (n,1) -> (n,)
        except Exception as e:
            skipped.append((os.path.basename(d), f"load error: {e}"))
            continue

        if imgs.ndim != 3:
            skipped.append((os.path.basename(d), f"imgs.ndim={imgs.ndim}, expected 3"))
            continue
        if labs.ndim != 1:
            skipped.append((os.path.basename(d), f"labs.ndim={labs.ndim}, expected 1 after squeeze"))
            continue
        if imgs.shape[0] != labs.shape[0]:
            skipped.append((os.path.basename(d), f"length mismatch: imgs {imgs.shape}, labs {labs.shape}"))
            continue
        # 이진 라벨 확인(0/1)
        uniq = np.unique(labs)
        if not np.all(np.isin(uniq, [0, 1])):
            skipped.append((os.path.basename(d), f"labels not binary-like: uniq={uniq[:5]}"))
            continue

        all_imgs.append(imgs.astype(np.float32))
        all_labs.append(labs.astype(np.int64))
        loaded_subjects += 1

    assert loaded_subjects > 0, \
        "No valid subject matched the expected patterns (*_AxialSlices_padded.npy / *_label_sliceLevel.npy)."

    X = np.concatenate(all_imgs, axis=0)
    y = np.concatenate(all_labs, axis=0)
    print(f"[Loaded] subjects={loaded_subjects}, total_slices={len(y)}, pos={(y==1).sum()}, neg={(y==0).sum()}")

    if skipped:
        print(f"[Note] skipped {len(skipped)} subjects (showing up to 10):")
        for name, why in skipped[:10]:
            print(f"  - {name}: {why}")

    return X, y
# =========================================================
# 불균형 처리: 음성(0) 언더샘플링
# =========================================================
def undersample_negatives(X: np.ndarray, y: np.ndarray, neg_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    양성(1) 수를 P라 할 때, 음성(0)은 min(neg_ratio * P, 현재 음성 수) 개 랜덤 추출.
    neg_ratio=1.0 → 1:1, 2.0 → 1:2
    """
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    P = len(pos_idx)
    keep_neg = min(int(neg_ratio * P), len(neg_idx))
    if keep_neg <= 0 or P == 0:
        # 양성이 없거나 계산이 비정상인 경우, 원본 반환
        return X, y
    sel_neg = np.random.choice(neg_idx, size=keep_neg, replace=False)
    keep_idx = np.concatenate([pos_idx, sel_neg])
    np.random.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx]


# =========================================================
# Train/Test split (slice-level, class별 층화)
# =========================================================
def train_test_split_stratified(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.3, seed: int = 42):
    rng = np.random.RandomState(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    def split_indices(indices):
        rng.shuffle(indices)
        n_test = int(math.floor(test_ratio * len(indices)))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        return train_idx, test_idx

    pos_train, pos_test = split_indices(idx_pos.copy())
    neg_train, neg_test = split_indices(idx_neg.copy())

    train_idx = np.concatenate([pos_train, neg_train])
    test_idx = np.concatenate([pos_test, neg_test])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])


# =========================================================
# Stratified train/val split helper
# =========================================================
def train_val_split_stratified(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.1, seed: int = 42):
    """Stratified split of (X, y) into train/val by class with given ratio."""
    rng = np.random.RandomState(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    def split_indices(indices):
        rng.shuffle(indices)
        n_val = int(math.floor(val_ratio * len(indices)))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        return train_idx, val_idx

    pos_train, pos_val = split_indices(idx_pos.copy())
    neg_train, neg_val = split_indices(idx_neg.copy())

    train_idx = np.concatenate([pos_train, neg_train])
    val_idx = np.concatenate([pos_val, neg_val])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx])


# =========================================================
# Dataset / Transform
# =========================================================
class NpyToFloat01(object):
    """ (H,W) → [0,1] 스케일 (0.5~99.5 퍼센타일 클리핑) """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        vmin, vmax = np.percentile(x, 0.5), np.percentile(x, 99.5)
        if vmax <= vmin:
            vmax = float(x.max())
            vmin = float(x.min())
        if vmax > vmin:
            x = (x - vmin) / (vmax - vmin)
        else:
            x = np.zeros_like(x, dtype=np.float32)
        x = np.clip(x, 0.0, 1.0)
        return x

class SliceDataset(Dataset):
    """
    단일 slice(1채널, 672x672)와 이진 라벨 제공.
    train=True일 때만 약한 augmentation(HFlip) 적용.
    """
    def __init__(self, images: np.ndarray, labels: np.ndarray, train: bool = True):
        self.images = images
        self.labels = labels.astype(np.float32)
        aug_list = []
        if train:
            aug_list += [transforms.RandomHorizontalFlip(p=0.5)]
        self.transform = transforms.Compose([
            NpyToFloat01(),          # (H,W) → [0,1]
            transforms.ToTensor(),   # (1,H,W)
            *aug_list,
        ])

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        lab = self.labels[idx]
        img = self.transform(img)   # torch.FloatTensor (1,672,672)
        return img, lab


# =========================================================
# 모델: ResNet18 (1채널 입력)
# =========================================================
def build_resnet18_1ch(num_classes: int = 1) -> nn.Module:
    """
    torchvision resnet18: 첫 conv를 1채널로, fc를 이진 로짓(1)로 교체.
    """
    model = models.resnet18(weights=None)  # 의학영상 특성 상 사전학습 비사용 기본
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# =========================================================
# 학습/평가 루프
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    for imgs, labs in loader:
        imgs = imgs.to(device, non_blocking=True)
        labs = labs.to(device, non_blocking=True).view(-1, 1)  # (B,1)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, labs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labs)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)



@torch.no_grad()
def evaluate(model, loader, device, criterion: Optional[nn.Module] = None):
    model.eval()
    all_probs, all_labels = [], []
    total_loss, total_n = 0.0, 0
    for imgs, labs in loader:
        imgs = imgs.to(device, non_blocking=True)
        labs_f = labs.to(device, non_blocking=True).view(-1, 1)
        logits = model(imgs)
        if criterion is not None:
            loss = criterion(logits, labs_f)
            total_loss += loss.item() * imgs.size(0)
            total_n += imgs.size(0)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labs.numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels).astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')
    cm = confusion_matrix(y_true, y_pred)
    out = {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc, "cm": cm, "y_prob": y_prob, "y_true": y_true}
    if criterion is not None and total_n > 0:
        out["loss"] = total_loss / total_n
    return out


# =========================================================
# 단일 slice 추론
# =========================================================
@torch.no_grad()
def infer_one_slice(model_path: str, slice_npy: np.ndarray, device: Optional[str] = None) -> float:
    """
    저장된 모델(.pt)을 불러와 단일 slice(672x672) FHV(+) 확률을 반환.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_resnet18_1ch(num_classes=1)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    x = NpyToFloat01()(slice_npy)
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(device)  # (1,1,672,672)
    logit = model(x)
    prob = torch.sigmoid(logit).item()
    return float(prob)


# =========================================================
# Plotting utilities
# =========================================================
def plot_training_curves(history: dict, out_dir: str):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    # Loss
    plt.figure(figsize=(7,5))
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss over epochs"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "curve_loss.png")); plt.close()
    # Accuracy (optional if present)
    if "train_acc" in history and "val_acc" in history:
        plt.figure(figsize=(7,5))
        plt.plot(epochs, history["train_acc"], label="train acc")
        plt.plot(epochs, history["val_acc"], label="val acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy over epochs"); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "curve_acc.png")); plt.close()
        
    # AUC over epochs (optional if present)
    if "train_auc" in history and "val_auc" in history and len(history["train_auc"]) == len(epochs):
        plt.figure(figsize=(7,5))
        plt.plot(epochs, history["train_auc"], label="train AUC")
        plt.plot(epochs, history["val_auc"], label="val AUC")
        plt.xlabel("Epoch"); plt.ylabel("AUC"); plt.title("AUC over epochs"); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "curve_auc.png")); plt.close()
        
    # LR over epochs (optional if present)
    if "lr" in history and len(history["lr"]) == len(epochs):
        plt.figure(figsize=(7,5))
        plt.plot(epochs, history["lr"], label="learning rate")
        plt.xlabel("Epoch"); plt.ylabel("LR"); plt.title("Learning rate over epochs")
        plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "curve_lr.png")); plt.close()


def save_eval_plots(y_true: np.ndarray, y_prob: np.ndarray, out_dir: str, prefix: str = "test", thr: float = 0.5):
    os.makedirs(out_dir, exist_ok=True)
    import pandas as pd
    # ROC
    try:
        from sklearn.metrics import roc_curve, auc as sk_auc
        fpr, tpr, thr_roc = roc_curve(y_true, y_prob)
        roc_auc = sk_auc(fpr, tpr)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0,1],[0,1], linestyle="--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC curve")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_roc.png")); plt.close()
        # Save ROC raw data to CSV for Prism
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr_roc}).to_csv(
            os.path.join(out_dir, f"{prefix}_roc.csv"), index=False
        )
    except Exception:
        pass
    
    # PR curve
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(figsize=(6,6))
        plt.plot(rec, prec)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curve")
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_pr.png")); plt.close()
        # Save PR raw data to CSV for Prism (note: threshold length is len-1, pad with NaN)
        thr_pr = np.append(_, np.nan)  # `_` holds thresholds from precision_recall_curve
        pd.DataFrame({"recall": rec, "precision": prec, "threshold": thr_pr}).to_csv(
            os.path.join(out_dir, f"{prefix}_pr.csv"), index=False
        )
    except Exception:
        pass

    # Confusion matrix at chosen threshold
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    # Save CM as CSV with labels
    pd.DataFrame(cm, index=["Actual_Neg","Actual_Pos"], columns=["Pred_Neg","Pred_Pos"]).to_csv(
        os.path.join(out_dir, f"{prefix}_cm.csv")
    )

    # Save per-sample predictions for Prism or audit
    pd.DataFrame({
        "index": np.arange(len(y_true)),
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "threshold": np.full_like(y_true, fill_value=thr, dtype=float)
    }).to_csv(os.path.join(out_dir, f"{prefix}_preds.csv"), index=False)

    plt.figure(figsize=(4.5,4))
    plt.imshow(cm, cmap="Blues")
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xticks([0,1],["Neg","Pos"]); plt.yticks([0,1],["Neg","Pos"])
    plt.title(f"Confusion Matrix (thr={thr:.2f})"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_cm.png")); plt.close()
    
def save_history_csv(history: dict, out_dir: str, fname: str = "history_metrics.csv"):
    """history 딕셔너리를 (epoch 단위 행) CSV로 저장."""
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    n = len(history.get("train_loss", []))
    epochs = np.arange(1, n + 1)
    # 필요한 키들 목록
    keys = ["train_loss","val_loss","train_acc","val_acc","train_auc","val_auc","lr"]
    data = {"epoch": epochs}
    for k in keys:
        v = history.get(k, [])
        # 길이 맞추기 (없으면 NaN)
        if len(v) < n:
            vv = list(v) + [np.nan] * (n - len(v))
        else:
            vv = v
        data[k] = vv
    pd.DataFrame(data).to_csv(os.path.join(out_dir, fname), index=False)


def save_curves_csv(y_true: np.ndarray, y_prob: np.ndarray, out_dir: str, prefix: str = "test"):
    """ROC/PR 곡선 원자료를 CSV로 저장하고, 혼동행렬/임계값 스윕도 옵션 저장."""
    import pandas as pd
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, f1_score, precision_score, recall_score

    os.makedirs(out_dir, exist_ok=True)

    # ROC
    fpr, tpr, thr_roc = roc_curve(y_true, y_prob)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr_roc}).to_csv(
        os.path.join(out_dir, f"{prefix}_roc.csv"), index=False
    )

    # PR (threshold 길이가 -1 짧음 → NaN padding)
    prec, rec, thr_pr = precision_recall_curve(y_true, y_prob)
    thr_pr = np.append(thr_pr, np.nan)  # 길이 맞추기
    pd.DataFrame({"recall": rec, "precision": prec, "threshold": thr_pr}).to_csv(
        os.path.join(out_dir, f"{prefix}_pr.csv"), index=False
    )

    # 혼동행렬(기본 임계값 0.5)
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=["TN","FP"], columns=["FN","TP"]).to_csv(
        os.path.join(out_dir, f"{prefix}_cm.csv")
    )

    # (선택) 임계값 스윕 테이블
    ts = np.linspace(0, 1, 1001)
    precs, recs, f1s = [], [], []
    for t in ts:
        yp = (y_prob >= t).astype(int)
        precs.append(precision_score(y_true, yp, zero_division=0))
        recs.append(recall_score(y_true, yp, zero_division=0))
        f1s.append(f1_score(y_true, yp, zero_division=0))
    pd.DataFrame({"threshold": ts, "precision": precs, "recall": recs, "f1": f1s}).to_csv(
        os.path.join(out_dir, f"{prefix}_threshold_sweep.csv"), index=False
    )


# =========================================================
# 메인
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="FHV slice-level binary classification (ResNet18)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="subject 폴더들이 모여있는 루트 (subject/slice_img.npy, subject/slice_label.npy)")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="모델/리포트 저장 폴더")
    parser.add_argument("--neg_ratio", type=float, default=1.0, help="언더샘플링 비율: 양성 1에 대해 음성 R (기본 1:1)")
    parser.add_argument("--test_ratio", type=float, default=0.3, help="train:test 비율 (기본 7:3 → 0.3)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="val ratio split from TRAIN (default 0.1)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_pos", type=float, default=1.0, help="BCE pos_weight (양성 가중치)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="자동 혼합 정밀도 사용")
    # LR scheduler options
    parser.add_argument("--lr_scheduler", type=str, default="none",
                        choices=["none", "step", "multistep", "cosine", "plateau"],
                        help="learning rate scheduler type")
    parser.add_argument("--lr_step_size", type=int, default=5, help="StepLR: step_size")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="StepLR/MultiStepLR: gamma")
    parser.add_argument("--lr_milestones", type=str, default="",
                        help="MultiStepLR: comma-separated epochs, e.g., 10,15")
    parser.add_argument("--lr_T_max", type=int, default=15, help="CosineAnnealingLR: T_max (epochs)")
    parser.add_argument("--lr_eta_min", type=float, default=0.0, help="CosineAnnealingLR: eta_min")
    parser.add_argument("--plateau_mode", type=str, default="min", choices=["min", "max"],
                        help="ReduceLROnPlateau mode")
    parser.add_argument("--plateau_factor", type=float, default=0.5, help="ReduceLROnPlateau factor")
    parser.add_argument("--plateau_patience", type=int, default=3, help="ReduceLROnPlateau patience (epochs)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # 1) 데이터 로드
    X, y = load_all_slices_from_tree(args.data_root)
    print(f"[Loaded] total={len(y)} | pos={(y==1).sum()} | neg={(y==0).sum()}")

    # 2) 음성 언더샘플링 (요청: FHV(-)을 적절히 랜덤 선택)
    X_bal, y_bal = undersample_negatives(X, y, neg_ratio=args.neg_ratio)
    print(f"[Balanced] total={len(y_bal)} | pos={(y_bal==1).sum()} | neg={(y_bal==0).sum()} (neg_ratio={args.neg_ratio})")

    # 3) train/test split then train/val split
    (Xtr_full, ytr_full), (Xte, yte) = train_test_split_stratified(X_bal, y_bal, test_ratio=args.test_ratio, seed=args.seed)
    (Xtr, ytr), (Xval, yval) = train_val_split_stratified(Xtr_full, ytr_full, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[Split] train={len(ytr)} | val={len(yval)} | test={len(yte)} (test_ratio={args.test_ratio}, val_ratio={args.val_ratio})")
    np.save(os.path.join(args.out_dir, "test_data.npy"), Xte)

    # 4) Datasets & Dataloaders
    ds_train = SliceDataset(Xtr, ytr, train=True)
    ds_train_eval = SliceDataset(Xtr, ytr, train=False)  # for clean train metrics
    ds_val   = SliceDataset(Xval, yval, train=False)
    ds_test  = SliceDataset(Xte, yte,  train=False)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=False,
                          persistent_workers=False if args.num_workers == 0 else True)
    dl_train_eval = DataLoader(ds_train_eval, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=False,
                               persistent_workers=False if args.num_workers == 0 else True)
    dl_val   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=False,
                          persistent_workers=False if args.num_workers == 0 else True)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=False,
                          persistent_workers=False if args.num_workers == 0 else True)

    # 5) 모델/손실/최적화
    model = build_resnet18_1ch(num_classes=1).to(device)
    pos_weight = torch.tensor([args.weight_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # LR Scheduler init
    scheduler = None
    if args.lr_scheduler == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "multistep":
        milestones = [int(m) for m in args.lr_milestones.split(',') if m.strip().isdigit()]
        if len(milestones) == 0:
            milestones = [int(max(1, args.epochs * 0.6)), int(max(2, args.epochs * 0.85))]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosine":
        T_max = args.lr_T_max if args.lr_T_max > 0 else args.epochs
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=args.lr_eta_min)
    elif args.lr_scheduler == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=args.plateau_mode, factor=args.plateau_factor,
            patience=args.plateau_patience, verbose=False
        )

    scaler = torch.amp.GradScaler('cuda') if (args.amp and device == "cuda") else None
    
    
    # 6) Training loop with validation tracking
    # history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    # history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_auc": [], "val_auc": []}
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_auc": [], "val_auc": [], "lr": []}
    best_f1, best_ckpt_path, best_metrics = -1.0, os.path.join(args.out_dir, "best_resnet18_fhv.pt"), None

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, dl_train, criterion, optimizer, device, scaler=scaler)
        tr_metrics = evaluate(model, dl_train_eval, device, criterion)
        val_metrics = evaluate(model, dl_val, device, criterion)
        history["train_loss"].append(tr_metrics.get("loss", tr_loss))
        history["val_loss"].append(val_metrics.get("loss", float('nan')))
        history["train_acc"].append(tr_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])
        history["train_auc"].append(tr_metrics.get("auc", float('nan')))
        history["val_auc"].append(val_metrics.get("auc", float('nan')))
        
        current_lr = optimizer.param_groups[0]['lr']
        history["lr"].append(current_lr)

        print(f"[Epoch {epoch:02d}] lr={current_lr:.3e} | tr_loss={tr_metrics.get('loss', tr_loss):.4f} | "
              f"val_loss={val_metrics.get('loss', float('nan')):.4f} | VAL: ACC={val_metrics['acc']:.3f} "
              f"PRE={val_metrics['prec']:.3f} REC={val_metrics['rec']:.3f} F1={val_metrics['f1']:.3f} "
              f"AUC={val_metrics['auc']:.3f}")

        # print(f"[Epoch {epoch:02d}] tr_loss={tr_metrics.get('loss', tr_loss):.4f} | val_loss={val_metrics.get('loss', float('nan')):.4f} | "
        #       f"VAL: ACC={val_metrics['acc']:.3f} PRE={val_metrics['prec']:.3f} REC={val_metrics['rec']:.3f} F1={val_metrics['f1']:.3f} AUC={val_metrics['auc']:.3f}")

        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_metrics.get('loss', None))  # val loss 기준
            else:
                scheduler.step()
        
        # Save best by validation F1
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_metrics = val_metrics
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "args": vars(args),
                "seed": args.seed
            }, best_ckpt_path)

    # Plot training curves
    plot_training_curves(history, args.out_dir)
    save_history_csv(history, args.out_dir)  # ← 프리즘용 학습곡선 CSV
    print(f"[Saved] curves -> {os.path.join(args.out_dir, 'curve_loss.png')}, curve_acc.png, curve_auc.png, curve_lr.png")

    # 7) Final evaluation on TEST using best checkpoint
    ckpt = torch.load(
        best_ckpt_path,
        map_location=device if isinstance(device, str) else None,
        weights_only=False  # 추후 기본값 변경 대비(원동작 유지)
        ) if os.path.exists(best_ckpt_path) else None
    
    if ckpt and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    test_metrics = evaluate(model, dl_test, device, criterion)

    # Save predictions and metrics
    np.save(os.path.join(args.out_dir, "test_y_true.npy"), test_metrics["y_true"]) 
    np.save(os.path.join(args.out_dir, "test_y_prob.npy"), test_metrics["y_prob"]) 

    # Plots: ROC, PR, Confusion Matrix
    save_eval_plots(test_metrics["y_true"], test_metrics["y_prob"], args.out_dir, prefix="test", thr=0.5)

    # Write detailed report
    rep_path = os.path.join(args.out_dir, "test_report.txt")
    with open(rep_path, "w") as f:
        f.write("=== Validation (best) Metrics ===\n")
        if best_metrics:
            f.write(f"ACC: {best_metrics['acc']:.4f}\nPREC: {best_metrics['prec']:.4f}\nREC: {best_metrics['rec']:.4f}\nF1: {best_metrics['f1']:.4f}\nAUC: {best_metrics['auc']:.4f}\n")
            f.write(f"CM:\n{best_metrics['cm']}\n\n")
        f.write("=== Test Metrics ===\n")
        f.write(f"ACC: {test_metrics['acc']:.4f}\n")
        f.write(f"PREC: {test_metrics['prec']:.4f}\n")
        f.write(f"REC: {test_metrics['rec']:.4f}\n")
        f.write(f"F1: {test_metrics['f1']:.4f}\n")
        f.write(f"AUC: {test_metrics['auc']:.4f}\n")
        f.write(f"CM:\n{test_metrics['cm']}\n\n")
        # Classification report at 0.5
        from sklearn.metrics import classification_report
        y_pred = (test_metrics['y_prob'] >= 0.5).astype(int)
        f.write("=== Classification Report (thr=0.5) ===\n")
        f.write(classification_report(test_metrics['y_true'], y_pred, digits=4))

    print(f"[Saved] curves -> {os.path.join(args.out_dir, 'curve_loss.png')} (+ curve_acc.png)")
    print(f"[Saved] test plots -> ROC/PR/CM under {args.out_dir}")
    print(f"[Saved] report -> {rep_path}")
    print(f"[Saved] best model -> {best_ckpt_path}")


if __name__ == "__main__":
    main()