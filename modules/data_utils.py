from typing import Tuple
import random, math, os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit

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


def expand_consecutive_ranges(nums, pad=1, min_val=0, max_val=None):
    """
    nums: 정수 리스트 또는 numpy array
    pad : 각 연속 구간을 좌우로 얼마나 확장할지(기본 1칸)
    min_val : 포함할 최소 값 (default=0)
    max_val : 포함할 최대 값 (default=None → 제한 없음)
    
    사용 예시
    print(expand_consecutive_ranges([14, 15, 16, 19, 20], pad=1))
    → [13, 14, 15, 16, 17, 18, 19, 20, 21]

    print(expand_consecutive_ranges([8, 9, 12, 13, 16, 19], pad=1))
    → [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    """
    import numpy as np
    if isinstance(nums, np.ndarray):
        nums = nums.tolist()  # numpy array → list
    
    if not nums:
        return []

    nums = sorted(set(nums))

    # 연속 구간 찾기
    ranges = []
    start = prev = nums[0]
    for x in nums[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append((start, prev))
            start = prev = x
    ranges.append((start, prev))

    # pad 적용 + min/max 제한
    expanded = []
    for a, b in ranges:
        new_a = max(min_val, a - pad)
        new_b = b + pad
        if max_val is not None:
            new_b = min(max_val, new_b)
        expanded.append((new_a, new_b))

    # 겹치는 구간 병합
    merged = []
    for a, b in sorted(expanded):
        if not merged or a > merged[-1][1] + 1:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)

    # 최종 결과
    out = []
    for a, b in merged:
        out.extend(range(a, b + 1))
    return out


# =========================================================
# 데이터 로더: subject 트리 순회 (권장)
# root/subjectX/{slice_img.npy, slice_label.npy}
# =========================================================
def load_all_slices_from_tree(root_dir: str, select_N = None, choose_major_slice=False) -> Tuple[np.ndarray, np.ndarray]:
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
    all_subj_ids = []
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
        
        if choose_major_slice:
            # print(d)
            if not any(labs):
                # fhv가 없는 경우, fhv가 대부분 발견되는 slice만 추출
                imgs = imgs[12:15]
                labs = labs[12:15]
            else:
                # fhv가 있는 경우, fhv가 존재하는 구간 +, - 1만큼의 slice만 추출
                idx_fhv = np.argwhere(labs==1).flatten().tolist()
                idx_choose = expand_consecutive_ranges(idx_fhv, max_val=len(labs)-1)
                imgs = imgs[idx_choose]
                labs = labs[idx_choose]

        all_imgs.append(imgs.astype(np.float32))
        all_labs.append(labs.astype(np.int64))
        all_subj_ids.append(np.full(imgs.shape[0], loaded_subjects, dtype=np.int64))  # ✅ subject 고유 ID

        loaded_subjects += 1

    assert loaded_subjects > 0, \
        "No valid subject matched the expected patterns (*_AxialSlices_padded.npy / *_label_sliceLevel.npy)."

    X = np.concatenate(all_imgs, axis=0)
    y = np.concatenate(all_labs, axis=0)
    all_subj_ids = np.concatenate(all_subj_ids, axis=0)
    print(f"[Loaded] subjects={loaded_subjects}, total_slices={len(y)}, pos={(y==1).sum()}, neg={(y==0).sum()}")

    if skipped:
        print(f"[Note] skipped {len(skipped)} subjects (showing up to 10):")
        for name, why in skipped[:10]:
            print(f"  - {name}: {why}")

    return X, y, all_subj_ids

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

def set_dataloader(data_root, select_N, neg_ratio, test_ratio, val_ratio, batch_size, num_workers, out_dir, seed):
    # 1) 데이터 로드
    X, y, _ = load_all_slices_from_tree(data_root, select_N)
    print(f"[Loaded] total={len(y)} | pos={(y==1).sum()} | neg={(y==0).sum()}")
    
    # 2) 음성 언더샘플링 (요청: FHV(-)을 적절히 랜덤 선택)
    X_bal, y_bal = undersample_negatives(X, y, neg_ratio=neg_ratio)
    print(f"[Balanced] total={len(y_bal)} | pos={(y_bal==1).sum()} | neg={(y_bal==0).sum()} (neg_ratio={neg_ratio})")
    
    # 3) train/test split then train/val split
    (Xtr_full, ytr_full), (Xte, yte) = train_test_split_stratified(X_bal, y_bal, test_ratio=test_ratio, seed=seed)
    (Xtr, ytr), (Xval, yval) = train_val_split_stratified(Xtr_full, ytr_full, val_ratio=val_ratio, seed=seed)
    print(f"[Split] train={len(ytr)} | val={len(yval)} | test={len(yte)} (test_ratio={test_ratio}, val_ratio={val_ratio})")
    np.save(os.path.join(out_dir, "test_data.npy"), Xte)
    
    # 4) Datasets & Dataloaders
    ds_train = SliceDataset(Xtr, ytr)
    ds_val   = SliceDataset(Xval, yval)
    ds_test  = SliceDataset(Xte, yte)
    
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=False,
                          persistent_workers=False if num_workers == 0 else True)
    dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=False,
                          persistent_workers=False if num_workers == 0 else True)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=False,
                          persistent_workers=False if num_workers == 0 else True)
    
    return dl_train, dl_val, dl_test


# functions for trio input
def _compute_runs(sep_mask: np.ndarray):
    """
    sep_mask가 일정하게 유지되는 구간(= 같은 subject 구간)을 (start, end)로 반환.
    sep_mask가 0/1로 번갈아 들어있어도, 값이 바뀌는 지점이 경계가 된다.
    
    - ex) sep_mask = array([0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1])
    - ex) runs = [(0, 2), (3, 5), (6, 8), (9, 11)]
    - ex) run_id = array([ 0,  0,  0,  1]
    """
    
    sep_mask = sep_mask.astype(np.int64)
    if len(sep_mask) == 0:
        return []
    # 경계 지점 찾기: 값이 바뀌는 인덱스
    boundaries = np.where(np.diff(sep_mask) != 0)[0]
    starts = np.r_[0, boundaries + 1]
    ends   = np.r_[boundaries, len(sep_mask) - 1]
    runs = list(zip(starts, ends))  # 각 (start, end) 포함 구간
    # 각 인덱스가 어느 run에 속하는지 맵핑 배열도 같이 반환하면 효율적
    run_id = np.empty(len(sep_mask), dtype=np.int64)
    for rid, (s, e) in enumerate(runs):
        run_id[s:e+1] = rid
    return runs, run_id

class SliceDataset2p5DMasked(Dataset):
    """
    Custom PyTorch Dataset for 2.5D slice-level classification with subject boundary masking.

    This dataset constructs a 2.5D input by stacking neighboring slices around a center slice index i.
    For each sample, the input has shape (C=2k+1, H, W), where k is the context size (number of slices
    before and after the center slice). The target label corresponds only to the center slice.

    Subject boundaries are enforced using `sep_mask`, so that slices from different subjects are never
    combined in one input window.

    Parameters
    ----------
    X : np.ndarray
        Array of all slices concatenated across subjects. Shape = (N, H, W).
    y : np.ndarray
        Corresponding labels for each slice. Shape = (N,), values ∈ {0, 1}.
    sep_mask : np.ndarray
        Array marking subject membership for each slice. Shape = (N,). Slices with the same value
        belong to the same subject. Subject boundaries are inferred where values change.
    k : int, optional (default=1)
        Context size. The number of neighboring slices to include before and after the center.
        The total input channels = 2k + 1.
    drop_mixed : bool, optional (default=False)
        If True, only keep samples where all slices in the window (center ±k) share the same label
        as the center slice. If False, allow label mixing and only the center label is used as target.
    pad_mode : str, {"edge", "zero"}, optional (default="edge")
        Strategy for handling subject boundary cases:
            - "edge": replicate the nearest valid slice from the same subject.
            - "zero": fill with zeros for out-of-bound indices.

    Returns
    -------
    __getitem__(idx) → (x, y)
        x : torch.FloatTensor
            Shape = (C=2k+1, H, W). 2.5D stacked slice tensor.
        y : torch.FloatTensor
            Shape = (), scalar label for the center slice.

    Notes
    -----
    - This dataset ensures slices from different subjects are never combined.
    - `drop_mixed=True` is stricter but reduces sample count. Typically `False` is preferred.
    - `pad_mode="edge"` preserves intensity statistics better than zero padding.
    - Use together with DataLoader for batching.
    """

    def __init__(self,
                 X: np.ndarray,         # (N, H, W) all slices
                 y: np.ndarray,         # (N,) binary labels
                 sep_mask: np.ndarray,  # (N,) subject mask (same subject = same value)
                 k: int = 1,
                 drop_mixed: bool = False,
                 pad_mode: str = "edge"):

        assert X.ndim == 3, "X must be (N,H,W)"
        assert y.ndim == 1 and len(y) == len(X), "y must be (N,) aligned with X"
        assert len(sep_mask) == len(X), "sep_mask must be (N,) aligned with X"
        assert pad_mode in ("edge", "zero")

        self.X = X
        self.y = y.astype(np.float32)
        self.sep_mask = sep_mask
        self.k = int(k)
        self.pad_mode = pad_mode

        # subject 구간 단위로 (start, end) 인덱스를 구해두고, 각 slice가 속한 subject id 매핑
        self.runs, self.run_id = _compute_runs(self.sep_mask)

        # 중심 인덱스 후보들을 모음 (drop_mixed 여부에 따라 필터링)
        centers = []
        for i in range(len(self.X)):
            rid = self.run_id[i]
            s, e = self.runs[rid]  # subject의 slice 구간 범위
            # 중심 i를 기준으로 좌우 k 슬라이스 인덱스 생성
            idxs = np.arange(i - self.k, i + self.k + 1)

            if self.pad_mode == "edge":
                # subject 경계를 넘으면 가장 가까운 slice를 복제
                idxs = np.clip(idxs, s, e)

            # drop_mixed 옵션 처리
            if drop_mixed:
                if self.pad_mode == "edge":
                    # 윈도우 내 모든 라벨이 중심 라벨과 동일해야 유지
                    if np.all(self.y[idxs] == self.y[i]):
                        centers.append(i)
                else:  # zero padding일 경우
                    valid = (idxs >= s) & (idxs <= e)
                    if np.all(self.y[idxs[valid]] == self.y[i]):
                        centers.append(i)
            else:
                centers.append(i)

        self.centers = np.array(centers, dtype=np.int64)

    def __len__(self):
        # Dataset의 크기 = 필터링 후 중심 인덱스 개수
        return len(self.centers)

    def __getitem__(self, j):
        """
        Args
        ----
        j : int
            Index into the dataset's valid centers.

        Returns
        -------
        x : torch.FloatTensor
            Shape = (2k+1, H, W), stack of center slice and neighbors.
        y : torch.FloatTensor
            Scalar (0.0 or 1.0), label of the center slice.
        """
        i = int(self.centers[j])        # 중심 인덱스
        rid = self.run_id[i]            # 해당 slice의 subject id
        s, e = self.runs[rid]           # subject slice 구간 [start, end]

        # 윈도우 인덱스 생성
        idxs = np.arange(i - self.k, i + self.k + 1)

        frames = []
        if self.pad_mode == "edge":
            idxs = np.clip(idxs, s, e)
            frames = [torch.from_numpy(self.X[t]).float() for t in idxs]
        else:  # zero padding 모드
            for t in idxs:
                if t < s or t > e:
                    # subject 경계 밖 → 제로 패딩
                    frames.append(torch.zeros_like(torch.from_numpy(self.X[0]).float()))
                else:
                    frames.append(torch.from_numpy(self.X[t]).float())

        # (C, H, W) 텐서로 합치기
        x = torch.stack(frames, dim=0)
        # 중심 slice 라벨
        y = torch.tensor(self.y[i], dtype=torch.float32)
        return x, y

def split_subjectwise_indices_ordered(subject_id, test_ratio=0.3, val_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    uniq_sid, first_pos = np.unique(subject_id, return_index=True)
    sid_in_order = uniq_sid[np.argsort(first_pos)]
    sid_shuffled = sid_in_order.copy(); rng.shuffle(sid_shuffled)

    n = len(sid_shuffled)
    n_test = int(round(n*test_ratio))
    n_val  = int(round((n - n_test)*val_ratio))

    test_s  = set(sid_shuffled[:n_test])
    remain  = sid_shuffled[n_test:]
    val_s   = set(remain[:n_val])
    train_s = set(remain[n_val:])

    idx_all = np.arange(len(subject_id))
    def expand(sids):
        mask = np.isin(subject_id, list(sids))
        return idx_all[mask]  # ✅ 원본 전역 인덱스 오름차순

    return expand(train_s), expand(val_s), expand(test_s)

def split_subjectwise_indices_stratified(subject_id: np.ndarray,
                                         y: np.ndarray,
                                         test_ratio=0.3, val_ratio=0.1, seed=42):
    """
    subject 단위로 stratified split을 수행.
    - subject 라벨: 해당 subject안에 1이 하나라도 있으면 1, 아니면 0
    - 먼저 subjects를 stratified로 trainval/test
    - 이후 trainval 안에서 stratified로 train/val
    - 각 split이 단일 클래스가 되면 seed를 바꿔 재시도
    """
    rng = np.random.RandomState(seed)
    # 1) subject 리스트와 각 subject의 라벨 만들기
    uniq_sid, first_pos = np.unique(subject_id, return_index=True)
    # subject 라벨 = 그 subject에 속한 slice 중 y==1이 하나라도 있으면 1
    subj_label = []
    for sid in uniq_sid:
        mask = (subject_id == sid)
        subj_label.append(int(np.any(y[mask] == 1)))
    subj_label = np.array(subj_label)

    def _indices_from_subjects(sel_subjects):
        m = np.isin(subject_id, sel_subjects)
        return np.where(m)[0]

    # 여러 번 재시도해서 단일 클래스 회피
    for attempt in range(50):
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed + attempt)
        (trainval_subj_idx, test_subj_idx), = sss1.split(uniq_sid, subj_label)
        trainval_subjects = uniq_sid[trainval_subj_idx]
        test_subjects     = uniq_sid[test_subj_idx]

        # train/val 층화
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed + 100 + attempt)
        (train_subj_idx, val_subj_idx), = sss2.split(trainval_subjects,
                                                     subj_label[trainval_subj_idx])
        train_subjects = trainval_subjects[train_subj_idx]
        val_subjects   = trainval_subjects[val_subj_idx]

        train_idx = _indices_from_subjects(train_subjects)
        val_idx   = _indices_from_subjects(val_subjects)
        test_idx  = _indices_from_subjects(test_subjects)

        # 각 split에 두 클래스가 모두 존재하는지 체크
        def _has_both_classes(idx):
            uy = np.unique(y[idx]) if len(idx) > 0 else []
            return set(uy) == {0, 1}
        if _has_both_classes(val_idx) and _has_both_classes(test_idx):
            return train_idx, val_idx, test_idx

    # 최악의 경우: 재시도 실패 → 마지막 결과라도 반환(경고 출력)
    print("[Warn] Could not make both classes appear in val/test after retries.")
    return train_idx, val_idx, test_idx

def apply_indices(X, y, sep_mask, idx):
    return X[idx], y[idx], sep_mask[idx]

def make_valid_centers(X, y, sep_mask, k=1, pad_mode="edge", drop_mixed=False):
    """subject 경계/라벨 혼합 규칙을 만족하는 유효 중심 인덱스 배열 생성."""
    # runs: [(start,end), ...], run_id[i]: i가 속한 run 번호
    runs, run_id = _compute_runs(sep_mask)
    centers = []
    for i in range(len(X)):
        rid = run_id[i]
        s, e = runs[rid]
        idxs = np.arange(i-k, i+k+1)

        # 경계 처리
        if pad_mode == "edge":
            idxs = np.clip(idxs, s, e)
            valid_mask = np.ones_like(idxs, dtype=bool)
        else:  # zero padding
            valid_mask = (idxs >= s) & (idxs <= e)

        if drop_mixed:
            # zero padding이면 경계 밖은 제외하고 비교
            cand = idxs[valid_mask]
            if not np.all(y[cand] == y[i]):
                continue

        centers.append(i)

    return np.array(centers, dtype=np.int64)

def undersample_on_centers(centers, y, neg_ratio=1.0, seed=42):
    """센터 인덱스 배열에 대해 라벨 기준 언더샘플링(재현성 보장)."""
    rng = np.random.RandomState(seed)
    yc = y[centers]
    pos_c = centers[yc == 1]
    neg_c = centers[yc == 0]

    P = len(pos_c)
    keep_neg = min(int(neg_ratio * P), len(neg_c))
    if P == 0 or keep_neg <= 0:
        return centers  # 그대로 반환(혹은 예외 처리)

    sel_neg = rng.choice(neg_c, size=keep_neg, replace=False)
    keep = np.concatenate([pos_c, sel_neg])
    rng.shuffle(keep)
    return keep

def map_centers_to_subset_idx(dataset: SliceDataset2p5DMasked, selected_centers: np.ndarray):
    """
    dataset.centers (원본 인덱스 배열) 안에서 selected_centers(원본 인덱스)의 위치를 찾아
    Subset 용 인덱스 리스트로 변환.
    
    -------------------------------
    헬퍼: 선택한 centers(원본 인덱스 기준)를 Dataset 내부 인덱스로 매핑
    - SliceDataset2p5DMasked는 __init__에서 self.centers(원본 인덱스) 배열을 만든다.
    - Subset은 "dataset 내부 인덱스"를 받으므로, 우리가 고른 centers(원본 인덱스)를
      dataset.centers에서의 위치로 변환해야 한다.
    -------------------------------
    """
    # 빠른 매핑을 위해 원본 index -> dataset 내부 위치 dict 생성
    pos_in_ds = {int(c): i for i, c in enumerate(dataset.centers.tolist())}
    subset_idx = [pos_in_ds[int(c)] for c in selected_centers if int(c) in pos_in_ds]
    return subset_idx

def trio_loader(data_root,
                select_N,
                choose_major_slice,
                pad_mode,
                drop_mixed,
                test_ratio,
                val_ratio,
                k,
                neg_ratio,
                batch_size,
                num_workers,
                out_dir,
                seed):
    """
    subject-level split + 2.5D(이웃 slice 윈도우) 입력을 위한 DataLoader 생성 유틸.

    개요:
        1) 데이터 로드 (X, y, subject_id)
        2) subject 단위로 train/val/test 분할  → 피험자 누수 방지
        3) 각 split에서 '유효 center 인덱스' 계산 (subject 경계/라벨혼합 정책 반영)
        4) train split의 center들에 한해 언더샘플링(neg_ratio) 적용
        5) 2.5D Dataset 구성 (윈도우: i-k … i … i+k)
        6) 언더샘플링으로 선별된 center만 Subset으로 학습에 사용
        7) DataLoader 반환

    Args:
        data_root (str)        : subject 폴더 루트
        select_N (int or None) : 앞에서부터 N명의 subject만 사용 (디버그/소규모 실험용). None이면 전체.
        choose_major_slice (bool): subject 내부에서 주요 slice만 추출할지 여부(전처리 단계 옵션)
        pad_mode (str)         : ["edge","zero"] 중 택1. 윈도우가 경계를 넘어갈 때 패딩 방식.
        drop_mixed (bool)      : True면 (i±k) 윈도우 내부 라벨이 모두 중심 라벨과 동일한 경우만 center로 허용.
                                 False면 라벨이 섞여도 중심 라벨만 타깃으로 사용(문맥 활용).
        test_ratio (float)     : subject-level test 비율
        val_ratio (float)      : subject-level val 비율 (train에서 분할)
        k (int)                : 윈도우 반경 (총 채널 수 = 2k+1). 예: k=1 → (i-1,i,i+1)
        neg_ratio (float)      : 언더샘플링 비율. 양성 1개당 음성 R개 유지(예: 1.0 → 1:1)
        batch_size (int)       : DataLoader 배치 크기
        num_workers (int)      : DataLoader worker 수
        out_dir (str)          : 사용된 center 등 메타 저장 경로
        seed (int)             : 재현성 seed

    Returns:
        dl_train (DataLoader), dl_val (DataLoader), dl_test (DataLoader), meta (dict)
            - meta["train_idx"], ["val_idx"], ["test_idx"] : subject-level split 인덱스(원본 기준)
            - meta["train_centers_bal"] : 언더샘플링 후 학습에 쓰기로 선택한 center(원본 기준)
            - meta["used_centers"]      : 실제 Subset으로 사용된 center(= train_centers_bal와 동일 의미)
    
    보장 사항:
        - (i-k … i … i+k) 윈도우는 Dataset 내부에서 **subject 경계를 넘지 않도록** sep_mask(=subject_id)를 이용해 구성됨.
        - 학습 시 `shuffle=True`를 써도 sample 정의(윈도우)가 깨지지 않음.
        - 언더샘플링은 train center에만 적용되며, val/test에는 손대지 않음.
    """
    # 1) Load dataset with subject_id
    X, y, subject_id = load_all_slices_from_tree(root_dir=data_root, 
                                                 select_N=select_N,
                                                 choose_major_slice=choose_major_slice)
    
    # 2) subject-level split (그룹 단위로 분할 → 누수 방지)
    #    - 같은 subject가 train/val/test에 동시에 들어가지 않도록 group-wise split
    train_idx, val_idx, test_idx = split_subjectwise_indices_stratified(
        subject_id, y, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed
    )
    
    # 3) 각 세트에서 '유효 center' 인덱스 계산
    #    - SliceDataset2p5DMasked가 사용할 수 있는 center만 모음
    #    - subject 경계를 넘지 않도록 보정 + (옵션) 라벨혼합(drop_mixed) 정책 반영
    tr_centers_all = make_valid_centers(X[train_idx], y[train_idx], subject_id[train_idx],
                                        k=k, pad_mode=pad_mode, drop_mixed=drop_mixed)
    # (val/test는 undersampling 대상이 아니므로, 여기서는 center 리스트만 Dataset 쪽에서 자체 보유)
    # va_centers_all = make_valid_centers(X[val_idx],   y[val_idx],   subject_id[val_idx],   k=k, pad_mode=pad_mode, drop_mixed=drop_mixed)
    # te_centers_all = make_valid_centers(X[test_idx],  y[test_idx],  subject_id[test_idx],  k=k, pad_mode=pad_mode, drop_mixed=drop_mixed)
    
    # 4) (중요) 언더샘플링은 train center에만 적용
    #    - y[train_idx] 기준으로, 양성/음성 비율을 맞춰 center를 선택
    #    - seed 고정으로 재현성 확보
    tr_centers_bal = undersample_on_centers(tr_centers_all, y[train_idx], neg_ratio=neg_ratio, seed=seed)
    
    # 5) Dataset 구성
    #    - 이 시점부터 X, y, subject_id는 split된 배열만 Dataset에 전달됨
    #    - Dataset 내부에서 center 배열을 사용해 (i-k … i … i+k) 윈도우를 구성
    ds_train_full = SliceDataset2p5DMasked(
        X=X[train_idx], y=y[train_idx], sep_mask=subject_id[train_idx],
        k=k, drop_mixed=drop_mixed, pad_mode=pad_mode
    )
    ds_val = SliceDataset2p5DMasked(
        X=X[val_idx], y=y[val_idx], sep_mask=subject_id[val_idx],
        k=k, drop_mixed=drop_mixed, pad_mode=pad_mode
    )
    ds_test = SliceDataset2p5DMasked(
        X=X[test_idx], y=y[test_idx], sep_mask=subject_id[test_idx],
        k=k, drop_mixed=drop_mixed, pad_mode=pad_mode
    )

    # 6) 우리가 고른 train centers만 사용하도록 Subset 구성
    #    - ds_train_full.centers: 유효 center(원본 'train split 기준' 인덱스) → Dataset 내부가 보유
    #    - tr_centers_bal       : undersampling으로 '선택된 center'(동일 좌표계)
    #    - map_centers_to_subset_idx: ds_train_full.centers 배열에서 tr_centers_bal 위치를 찾아 Subset 인덱스 생성
    subset_idx = map_centers_to_subset_idx(ds_train_full, tr_centers_bal)
    ds_train = Subset(ds_train_full, subset_idx)
    
    print(f"[Centers] train(full)={len(ds_train_full)} | train(bal)={len(ds_train)} "
          f"| val={len(ds_val)} | test={len(ds_test)}")
    
    # ===== 7) DataLoader 생성 =====
    # 재현성: DataLoader용 난수 생성기 설정
    g = torch.Generator()
    g.manual_seed(seed)

    # workers가 있으면 persistent_workers=True가 대체로 효율적(환경 따라 False가 더 안정적일 수도 있음)
    pw_train = (num_workers > 0)
    pw_eval  = (num_workers > 0)

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,              # 학습은 셔플 권장(편향 완화)
        num_workers=num_workers,
        pin_memory=False,          # CUDA면 True도 고려 가능(호스트→디바이스 전송 최적화)
        persistent_workers=pw_train,
        generator=g
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,             # 검증/테스트는 재현성을 위해 고정
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=pw_eval
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=pw_eval
    )

    # (옵션) 추적/재현을 위해 학습에 실제로 사용된 center 저장
    os.makedirs(out_dir, exist_ok=True)
    used_centers = np.array(ds_train_full.centers)[subset_idx]
    np.save(os.path.join(out_dir, "train_centers_used.npy"), used_centers)

    # 메타 반환(분할 인덱스 및 center 선택 정보)
    meta = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "train_centers_bal": tr_centers_bal,
        "used_centers": used_centers
    }
    
    # 예: train 셋 몇 개 샘플을 꺼내 윈도우의 실제 인덱스와 라벨 확인
    # dl_debug = DataLoader(ds_train, batch_size=1, shuffle=False)
    # for n, batch in enumerate(dl_debug):
    #     x, y = batch           # x: (B=1, C=2k+1, H, W), y: (B,)
    #     # ds_train은 Subset이므로, 원본 center는 used_centers[n]로 추적 가능
    #     center_global = tr_centers_bal[n]
    #     print(f"[sample {n}] center_global={center_global}, label={y.item()}")
    #     if n == 5:
    #         break

    return dl_train, dl_val, dl_test, meta
    
    
    