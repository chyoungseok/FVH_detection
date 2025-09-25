import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


def _robust_minmax(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.0, eps: float = 1e-6) -> np.ndarray:
    lo, hi = np.percentile(x, [p_low, p_high])
    x = np.clip(x, lo, hi)
    return (x - lo) / max(hi - lo, eps)


class FlairNPYSliceDataset(Dataset):
    """
    Slice-level binary classification dataset for FLAIR MRI.

    Expected structure:
      root_dir/
        subject1/
          subject1_AxialSlices_padded.npy   # (S, H, W)
          subject1_label_sliceLevel.npy     # (S, 1) or (S,)
        subject2/
          subject2_AxialSlices_padded.npy
          subject2_label_sliceLevel.npy
        ...

    Args:
        root_dir: path to the data directory
        split: "train", "val", "test", or None (all subjects)
        split_ratio: tuple for splitting (train, val, test), must sum to 1.0
        robust_norm: apply percentile-based robust min-max normalization
        seed: random seed for reproducible splits
    """

    def __init__(self, root_dir: str,
                 split: str = None,
                 split_ratio=(0.7, 0.15, 0.15),
                 robust_norm: bool = True,
                 seed: int = 42):
        super().__init__()
        self.root_dir = root_dir
        self.robust_norm = robust_norm

        # --- Step 1: subject 목록 수집 ---
        all_subjects = []
        for subj_dir in sorted(os.listdir(root_dir)):
            subj_path = os.path.join(root_dir, subj_dir)
            if not os.path.isdir(subj_path):
                continue

            img_files = glob.glob(os.path.join(subj_path, "*AxialSlices_padded.npy"))
            lab_files = glob.glob(os.path.join(subj_path, "*label_sliceLevel.npy"))
            if len(img_files) != 1 or len(lab_files) != 1:
                continue  # skip if missing files

            all_subjects.append({
                "sid": subj_dir,
                "image": img_files[0],
                "label": lab_files[0],
            })

        # --- Step 2: subject split ---
        rng = np.random.RandomState(seed)
        rng.shuffle(all_subjects)

        n_total = len(all_subjects)
        n_train = int(split_ratio[0] * n_total)
        n_val = int(split_ratio[1] * n_total)
        n_test = n_total - n_train - n_val

        split_subjects = {
            "train": all_subjects[:n_train],
            "val": all_subjects[n_train:n_train + n_val],
            "test": all_subjects[n_train + n_val:]
        }

        if split is None:
            self.subjects = all_subjects
        else:
            if split not in split_subjects:
                raise ValueError(f"Invalid split={split}, must be one of {list(split_subjects.keys())}")
            self.subjects = split_subjects[split]

        # --- Step 3: slice-level index 생성 ---
        self._index = []
        for si, rec in enumerate(self.subjects):
            imgs = np.load(rec["image"], mmap_mode="r")  # shape only 확인
            labs = np.load(rec["label"], mmap_mode="r")

            if labs.ndim == 2 and labs.shape[1] == 1:
                labs = labs[:, 0]

            assert imgs.shape[0] == labs.shape[0], f"Shape mismatch {imgs.shape} vs {labs.shape}"

            for k in range(imgs.shape[0]):
                self._index.append((si, k))

    def __len__(self):
        return len(self._index)

    @property
    def labels(self):
        all_labels = []
        for si, k in self._index:
            labs = np.load(self.subjects[si]["label"], mmap_mode="r")
            if labs.ndim == 2 and labs.shape[1] == 1:
                labs = labs[:, 0]
            all_labels.append(float(labs[k]))
        return all_labels

    def __getitem__(self, idx: int):
        si, k = self._index[idx]   # subject index, slice index

        # 필요한 slice만 on-the-fly 로드
        imgs = np.load(self.subjects[si]["image"], mmap_mode="r")
        labs = np.load(self.subjects[si]["label"], mmap_mode="r")
        if labs.ndim == 2 and labs.shape[1] == 1:
            labs = labs[:, 0]

        img = imgs[k].astype(np.float32)
        lab = float(labs[k])

        # normalization
        if self.robust_norm:
            img = _robust_minmax(img)
        else:
            vmin, vmax = float(img.min()), float(img.max())
            img = (img - vmin) / max(vmax - vmin, 1e-6)

        x = torch.from_numpy(img)[None, ...]  # (1,672,672)
        x = (x - 0.5) / 0.5

        return {
            "image": x.float(),
            "label": torch.tensor(lab, dtype=torch.float32),
            "subject_id": self.subjects[si]["sid"],   # subject 이름
            "slice_idx": k                           # slice 번호
        }


def compute_pos_weight_from_dataset(dataset: FlairNPYSliceDataset) -> torch.Tensor:
    labels = np.array(dataset.labels, dtype=np.float32)
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if n_pos <= 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)
