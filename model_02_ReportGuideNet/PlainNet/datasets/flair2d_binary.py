import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit


def _robust_minmax(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.0, eps: float = 1e-6) -> np.ndarray:
    """Percentile 기반 robust min-max normalization"""
    lo, hi = np.percentile(x, [p_low, p_high])
    x = np.clip(x, lo, hi)
    return (x - lo) / max(hi - lo, eps)


class FlairNPYSliceDataset(Dataset):
    """FLAIR slice-level binary classification dataset (subject-level npy 구조)"""
    _cached_split = None  # 모든 인스턴스가 동일 split을 공유

    def __init__(self, root_dir: str,
                 split: str = "train",
                 split_ratio=(0.7, 0.15, 0.15),
                 robust_norm: bool = True,
                 seed: int = 42,
                 stratified_split: bool = False,
                 undersample: bool = False):
        super().__init__()
        self.root_dir = root_dir
        self.robust_norm = robust_norm
        self.split = split
        self.seed = seed
        self.stratified_split = stratified_split
        self.undersample = undersample

        # --- Step 1: subject-level 파일 수집 ---
        all_subjects = []
        for subj_dir in sorted(os.listdir(root_dir)):
            subj_path = os.path.join(root_dir, subj_dir)
            if not os.path.isdir(subj_path):
                continue

            img_files = glob.glob(os.path.join(subj_path, "*AxialSlices_padded.npy"))
            lab_files = glob.glob(os.path.join(subj_path, "*label_sliceLevel.npy"))
            if len(img_files) != 1 or len(lab_files) != 1:
                continue  # 파일이 없으면 스킵

            all_subjects.append({
                "sid": subj_dir,
                "image": img_files[0],
                "label": lab_files[0],
            })

        # --- Step 2: split (stratified or random) ---
        if stratified_split:
            # slice-level label을 기반으로 stratify
            all_labels = []
            for rec in all_subjects:
                labs = np.load(rec["label"], mmap_mode="r")
                if labs.ndim == 2 and labs.shape[1] == 1:
                    labs = labs[:, 0]
                all_labels.extend(labs.tolist())
            all_labels = np.array(all_labels)

            # slice 단위 index 생성
            n_total = len(all_labels)
            idx_all = np.arange(n_total)

            # 1차: train vs (val+test)
            sss1 = StratifiedShuffleSplit(n_splits=1,
                                          test_size=split_ratio[1] + split_ratio[2],
                                          random_state=seed)
            train_idx, tmp_idx = next(sss1.split(idx_all, all_labels))

            # 2차: val vs test
            rel_test = split_ratio[2] / (split_ratio[1] + split_ratio[2])
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel_test, random_state=seed)
            val_idx, test_idx = next(sss2.split(idx_all[tmp_idx], all_labels[tmp_idx]))
            val_idx = tmp_idx[val_idx]
            test_idx = tmp_idx[test_idx]

            FlairNPYSliceDataset._cached_split = {
                "train": train_idx, "val": val_idx, "test": test_idx
            }

            split_idx = FlairNPYSliceDataset._cached_split[split]
            self._index = split_idx.tolist()

        else:
            rng = np.random.default_rng(seed)
            n_total = sum(np.load(rec["label"], mmap_mode="r").shape[0] for rec in all_subjects)
            idx_all = np.arange(n_total)
            rng.shuffle(idx_all)

            n_train = int(split_ratio[0] * n_total)
            n_val = int(split_ratio[1] * n_total)

            FlairNPYSliceDataset._cached_split = {
                "train": idx_all[:n_train],
                "val": idx_all[n_train:n_train + n_val],
                "test": idx_all[n_train + n_val:],
            }

            self._index = FlairNPYSliceDataset._cached_split[split].tolist()

        # --- Step 3: subject index 매핑 ---
        self.samples = []
        self.labels = []
        for rec in all_subjects:
            imgs = np.load(rec["image"], mmap_mode="r")
            labs = np.load(rec["label"], mmap_mode="r")
            if labs.ndim == 2 and labs.shape[1] == 1:
                labs = labs[:, 0]
            for k in range(imgs.shape[0]):
                self.samples.append((rec["image"], rec["label"], k, rec["sid"]))
                self.labels.append(int(labs[k]))

        self.samples = [self.samples[i] for i in self._index]
        self.labels = np.array([self.labels[i] for i in self._index])

        # --- Step 4: undersampling (train split에서만 적용) ---
        if self.undersample and self.split == "train":
            rng = np.random.default_rng(seed)
            pos_idx = np.where(self.labels == 1)[0]
            neg_idx = np.where(self.labels == 0)[0]

            if len(pos_idx) > 0:
                neg_idx_down = rng.choice(neg_idx, size=len(pos_idx), replace=False)
                keep_idx = np.concatenate([pos_idx, neg_idx_down])
                rng.shuffle(keep_idx)

                self.samples = [self.samples[i] for i in keep_idx]
                self.labels = self.labels[keep_idx]

        # --- Step 5: 로그 출력 ---
        n_pos = int((self.labels == 1).sum())
        n_neg = int((self.labels == 0).sum())
        ratio = n_pos / max(n_pos + n_neg, 1)
        print(f"[{split.upper()}] total={len(self.labels)}, pos={n_pos}, neg={n_neg}, pos_ratio={ratio:.3f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_file, lab_file, k, sid = self.samples[idx]
        imgs = np.load(img_file, mmap_mode="r")
        labs = np.load(lab_file, mmap_mode="r")
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

        x = torch.from_numpy(img)[None, ...]  # (1, H, W)
        x = (x - 0.5) / 0.5

        return {
            "image": x.float(),
            "label": torch.tensor(lab, dtype=torch.float32),
            "subject_id": sid,
            "slice_idx": k
        }


def compute_pos_weight_from_dataset(dataset: FlairNPYSliceDataset) -> torch.Tensor:
    labels = np.array(dataset.labels, dtype=np.float32)
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if n_pos <= 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)
