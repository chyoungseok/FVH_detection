from typing import Tuple, Optional
import random, math
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
def load_all_slices_from_tree(root_dir: str, select_N = None) -> Tuple[np.ndarray, np.ndarray]:
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

    if select_N is None:
        select_N = len(subj_dirs)
    
    for d in subj_dirs[:select_N]:
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
class SliceDataset(Dataset):
    """
    단일 slice(1채널, 672x672)와 이진 라벨 제공.
    - 증강(augmentation) 없음
    - 슬라이스 생성 단계에서 이미 [0,1] 정규화가 끝났다고 가정
    """
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels.astype(np.float32)

        # Compose는 여러 transform을 순서대로 적용하기 위한 래퍼.
        # 여기서는 ToTensor()만 사용하지만, 이후 필요 시 Normalize 등을 쉽게 추가할 수 있도록 형태 유지.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # ToTensor():
            # - NumPy → torch.Tensor로 변환 (모델/역전파/GPU 연산은 torch.Tensor만 지원)
            # - (H, W) → (1, H, W) 로 채널 차원 추가 (Conv2d는 입력을 (N, C, H, W)로 기대)
            # - 값이 0~255일 경우 0~1로 스케일링. 이미 [0,1]이면 그대로 유지됨.
        ])

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]           # (H, W), 이미 [0,1] 범위라고 가정
        lab = self.labels[idx]           # 스칼라(0/1) 또는 (,) 형태

        # (H, W) numpy → (1, H, W) float tensor
        # PyTorch Conv2d는 입력을 (N, C, H, W)로 받으므로, DataLoader가 배치로 쌓으면 (N, 1, H, W)가 됨.
        img = self.transform(img)        # torch.FloatTensor (1, 672, 672)

        # 라벨도 torch.Tensor로 명시 변환 (BCEWithLogitsLoss는 float 타깃을 기대)
        lab = torch.as_tensor(lab, dtype=torch.float32)

        return img, lab

