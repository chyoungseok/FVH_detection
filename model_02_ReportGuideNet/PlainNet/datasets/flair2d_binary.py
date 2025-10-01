import os
import glob
import numpy as np
import torch
import h5py
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit


def _robust_minmax(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.0, eps: float = 1e-6) -> np.ndarray:
    """Percentile 기반 robust min-max normalization"""
    lo, hi = np.percentile(x, [p_low, p_high])
    x = np.clip(x, lo, hi)
    return (x - lo) / max(hi - lo, eps)


class FlairNPYSliceDataset(Dataset):
    """FLAIR slice-level binary classification dataset (subject-level npy 구조, 메모리 캐싱 적용)"""
    _cached_split = None  # 모든 인스턴스가 동일 split을 공유

    def __init__(self, root_dir: str,
                 split: str = "train",
                 split_ratio=(0.7, 0.15, 0.15),
                 robust_norm: bool = False,
                 seed: int = 42,
                 stratified_split: bool = True,
                 undersample: bool = True,
                 undersample_ratio: float = 1.0):  # ✅ undersample 비율 추가
        super().__init__()
        self.root_dir = root_dir
        self.robust_norm = robust_norm
        self.split = split
        self.seed = seed
        self.stratified_split = stratified_split
        self.undersample = undersample
        self.undersample_ratio = undersample_ratio

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
            all_labels = []
            for rec in all_subjects:
                labs = np.load(rec["label"], mmap_mode="r")
                if labs.ndim == 2 and labs.shape[1] == 1:
                    labs = labs[:, 0]
                all_labels.extend(labs.tolist())
            all_labels = np.array(all_labels)

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

        # --- Step 3: subject index 매핑 + 메모리 캐싱 ---
        self.samples = []
        self.labels = []
        self.data_cache = {}  # ✅ 캐싱

        for rec in tqdm(all_subjects, desc="Load npy data"):
            imgs = np.load(rec["image"]).astype(np.float32)
            labs = np.load(rec["label"]).astype(np.float32)
            if labs.ndim == 2 and labs.shape[1] == 1:
                labs = labs[:, 0]
            self.data_cache[rec["sid"]] = (imgs, labs)  # ✅ 캐싱 저장
            for k in range(imgs.shape[0]):
                self.samples.append((rec["sid"], k))
                self.labels.append(int(labs[k]))

        self.samples = [self.samples[i] for i in self._index]
        self.labels = np.array([self.labels[i] for i in self._index])

        # --- Step 4: undersampling ---
        if self.undersample:
            rng = np.random.default_rng(seed)
            pos_idx = np.where(self.labels == 1)[0]
            neg_idx = np.where(self.labels == 0)[0]

            if len(pos_idx) > 0 and len(neg_idx) > 0:
                target_neg = int(len(pos_idx) * self.undersample_ratio)
                target_neg = min(target_neg, len(neg_idx))

                neg_idx_down = rng.choice(neg_idx, size=target_neg, replace=False)
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
        sid, k = self.samples[idx]
        imgs, labs = self.data_cache[sid]  # ✅ 캐시에서 바로 접근

        img = imgs[k]
        lab = labs[k]

        # if self.robust_norm:
        #     img = _robust_minmax(img)
        # else:
        #     vmin, vmax = float(img.min()), float(img.max())
        #     img = (img - vmin) / max(vmax - vmin, 1e-6)

        x = torch.from_numpy(img)[None, ...]
        # x = (x - 0.5) / 0.5

        return {
            "image": x.float(),
            "label": torch.tensor(float(lab), dtype=torch.float32),
            "subject_id": sid,
            "slice_idx": k
        }


import os
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit


class FlairH5SliceDataset(Dataset):
    """
    FLAIR slice-level binary classification dataset (HDF5 기반)
    구조:
      root_dir/data.h5
        - images: (N, H, W) float32  또는  (N, C, H, W) float32
        - labels: (N,) int {0,1}
        - subject_ids: (N,) string (optional)
    """
    _cached_split = None  # 모든 인스턴스에서 동일 split 공유

    def __init__(self, root_dir: str,
                 h5_file: str = "flair_slice_dataset.h5",
                 split: str = "train",
                 split_ratio=(0.7, 0.15, 0.15),
                 seed: int = 42,
                 stratified_split: bool = True,
                 undersample: bool = False,
                 undersample_ratio: float = 1.0):

        super().__init__()
        self.split = split
        self.seed = seed
        self.stratified_split = stratified_split
        self.undersample = undersample
        self.undersample_ratio = undersample_ratio

        # --- Step 1: h5 파일 열기 ---
        self.h5_path = os.path.join(root_dir, h5_file)
        self.h5 = h5py.File(self.h5_path, "r")
        self.images = self.h5["images"]   # (N,H,W) or (N,C,H,W)
        self.labels = self.h5["labels"]
        self.sids = self.h5["subject_ids"] if "subject_ids" in self.h5 else None  # ✅ 추가됨

        n_total = self.labels.shape[0]
        all_labels = np.array(self.labels[:])  # ✅ numpy array로 변환 (중요)

        # --- Step 2: split (stratified or random) ---
        if stratified_split:
            idx_all = np.arange(n_total)

            # 1차: train vs (val+test)
            sss1 = StratifiedShuffleSplit(
                n_splits=1, test_size=split_ratio[1] + split_ratio[2], random_state=seed
            )
            train_idx, tmp_idx = next(sss1.split(idx_all, all_labels))

            # 2차: val vs test
            rel_test = split_ratio[2] / (split_ratio[1] + split_ratio[2])
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel_test, random_state=seed)
            val_idx, test_idx = next(sss2.split(tmp_idx, all_labels[tmp_idx]))

            FlairH5SliceDataset._cached_split = {
                "train": np.array(train_idx),
                "val": np.array(tmp_idx[val_idx]),
                "test": np.array(tmp_idx[test_idx]),
            }

        else:
            rng = np.random.default_rng(seed)
            idx_all = np.arange(n_total)
            rng.shuffle(idx_all)

            n_train = int(split_ratio[0] * n_total)
            n_val = int(split_ratio[1] * n_total)

            FlairH5SliceDataset._cached_split = {
                "train": idx_all[:n_train],
                "val": idx_all[n_train:n_train + n_val],
                "test": idx_all[n_train + n_val:],
            }

        self.indices = FlairH5SliceDataset._cached_split[split]

        # --- Step 3: undersampling (선택적) ---
        self.labels_np = all_labels[self.indices]  # ✅ numpy indexing 안전하게
        if undersample:
            rng = np.random.default_rng(seed)
            pos_idx = np.where(self.labels_np == 1)[0]
            neg_idx = np.where(self.labels_np == 0)[0]

            if len(pos_idx) > 0 and len(neg_idx) > 0:
                target_neg = int(len(pos_idx) * undersample_ratio)
                target_neg = min(target_neg, len(neg_idx))
                neg_idx_down = rng.choice(neg_idx, size=target_neg, replace=False)
                keep_idx = np.concatenate([pos_idx, neg_idx_down])
                rng.shuffle(keep_idx)
                self.indices = self.indices[keep_idx]
                self.labels_np = self.labels_np[keep_idx]

        # --- Step 4: 데이터 차원 감지 ---
        sample = self.images[0]
        if sample.ndim == 2:
            self.is_2d = True   # (H,W)
        elif sample.ndim == 3:
            self.is_2d = False  # (C,H,W)
        else:
            raise ValueError(f"지원하지 않는 데이터 차원: {sample.shape}")

        # --- Step 5: 로그 출력 ---
        n_pos = int((self.labels_np == 1).sum())
        n_neg = int((self.labels_np == 0).sum())
        ratio = n_pos / max(n_pos + n_neg, 1)
        print(f"[{split.upper()}] total={len(self.indices)}, pos={n_pos}, neg={n_neg}, "
              f"pos_ratio={ratio:.3f}, data_mode={'2D' if self.is_2d else '3D'}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        img = self.images[i].astype(np.float32)  # (H,W) or (C,H,W)
        lab = float(self.labels[i])

        if self.is_2d:
            x = torch.from_numpy(img)[None, ...]  # (1,H,W)
        else:
            x = torch.from_numpy(img)             # (C,H,W)

        # subject id 불러오기
        sid = None
        if self.sids is not None:
            sid = self.sids[i].decode("utf-8") if isinstance(self.sids[i], bytes) else str(self.sids[i])

        return {
            "image": x.float(),
            "label": torch.tensor(lab, dtype=torch.float32),
            "slice_idx": int(i),
            "sid": sid if sid is not None else f"sample_{i}"
        }

    def close(self):
        """HDF5 파일 닫기"""
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None


def compute_pos_weight_from_dataset(dataset) -> torch.Tensor:
    labels = np.array(dataset.labels, dtype=np.float32)
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if n_pos <= 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)
