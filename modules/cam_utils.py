import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model: nn.Module, target_module: nn.Module):
        self.model = model
        self.target_module = target_module
        self.features = None
        self.gradients = None
        self.hook_handles = []

    def _register(self):
        def fwd_hook(module, inp, out):
            self.features = out
        def bwd_hook(module, grad_in, grad_out):
            # grad_out is a tuple; take gradients wrt module output
            self.gradients = grad_out[0]
        self.hook_handles.append(self.target_module.register_forward_hook(fwd_hook))
        self.hook_handles.append(self.target_module.register_full_backward_hook(bwd_hook))

    def _remove(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []

    @torch.no_grad()
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # min-max normalize per-tensor to [0,1]
        x_min = torch.min(x)
        x_max = torch.max(x)
        if (x_max - x_min) > 1e-8:
            x = (x - x_min) / (x_max - x_min)
        else:
            x = torch.zeros_like(x)
        return x

    def compute_cam(self, x: torch.Tensor, target: str = 'pos') -> torch.Tensor:
        """
        x: (1,1,H,W)
        returns heatmap tensor (1,1,H,W) in [0,1]
        """
        assert x.dim() == 4 and x.size(0) == 1, "Grad-CAM expects a single-sample tensor"
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self._register()
        try:
            with torch.enable_grad():
                # forward
                out = self.model(x)  # (1,1)
                score = out[0,0] if target == 'pos' else -out[0,0]
                # backward to get gradients at target module
                score.backward()
                A = self.features           # (1,C,h,w)
                dYdA = self.gradients       # (1,C,h,w)
                # weights: GAP over spatial dims
                weights = torch.mean(dYdA, dim=(2,3), keepdim=True)  # (1,C,1,1)
                cam = torch.sum(weights * A, dim=1, keepdim=True)    # (1,1,h,w)
                cam = F.relu(cam)
                # upsample to input size
                cam = F.interpolate(cam, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
                cam = self._normalize(cam)
        finally:
            self._remove()
            self.model.zero_grad(set_to_none=True)
        return cam  # (1,1,H,W)

def save_cam_overlay(img_np: np.ndarray, heat_01: np.ndarray, out_path: str, alpha: float=0.35):
    """Save grayscale image overlaid with heatmap.
    img_np: (H,W) original npy image (before transform)
    heat_01: (H,W) float32 in [0,1]
    """
    base = img_np  # [0,1]
    plt.figure(figsize=(5,5))
    plt.imshow(base, cmap='gray')
    plt.imshow(heat_01, cmap='jet', alpha=alpha)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()


def cam_analysis(cam_dir, out_dir, model, cam_layer, test_metrics, dl_test, device, cam_target):    
    cam_out = cam_dir if cam_dir else os.path.join(out_dir, "cam_test")
    os.makedirs(cam_out, exist_ok=True)
    
    # extract Xte, yte
    Xte = dl_test.dataset.images
    yte = dl_test.dataset.labels

    # select target module
    target_module = model.layer4 if cam_layer == "layer4" else model.layer3
    cam_engine = GradCAM(model, target_module)

    # Use probabilities computed earlier to name files consistently
    y_true_te = test_metrics["y_true"]
    y_prob_te = test_metrics["y_prob"]

    # Iterate in the same order as Xte/yte used to build ds_test (no shuffle)
    print(f"[CAM] Exporting overlays for TEST set -> {cam_out}")
    for i in range(len(Xte)):
        img_np = Xte[i]
        lab = int(yte[i])
        prob = float(y_prob_te[i]) if i < len(y_prob_te) else float('nan')

        # build input tensor with same preprocessing
        x = img_np
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(device)

        # compute CAM (disable AMP to stabilize grads)
        with torch.amp.autocast('cuda', enabled=False):
            heat = cam_engine.compute_cam(x, target=cam_target)  # (1,1,H,W)
        heat_np = heat[0,0].detach().cpu().numpy().astype(np.float32)

        # save overlay and raw heatmap
        fname = f"cam_test_{i:05d}_y{lab}_p{prob:.3f}.png"
        save_cam_overlay(img_np, heat_np, os.path.join(cam_out, fname))
        # also save raw arrays (optional, useful for later)
        np.save(os.path.join(cam_out, f"cam_test_{i:05d}_heat.npy"), heat_np)
    print(f"[CAM] Done. Saved {len(Xte)} overlays.")
    
# cam_utils.py



def cam_analysis_trio(cam_dir,                
                      dl_test,
                      model,
                      device,
                      test_metrics,
                      cam_layer="layer4",
                      cam_target="pos",
                      overlay_alpha=0.35,
                      out_dir=None):
    """
    테스트 세트에 대해 Grad-CAM을 생성/저장하는 함수.
    - dl_test.dataset은 SliceDataset2p5DMasked 인스턴스(Shuffle=False 가정)
    - 원본 중앙 슬라이스 이미지는 dataset.X[dataset.centers[i]]에서 가져옴
    - test_metrics["y_true"], ["y_prob"]는 dl_test 순서와 동일하다고 가정

    Args:
        cam_dir (str): CAM 결과 저장 디렉토리
        dl_test (DataLoader): 테스트 데이터 로더 (shuffle=False)
        model (nn.Module): 학습된 모델
        device (str): "cuda" or "cpu"
        test_metrics (dict): evaluate() 반환 딕셔너리 (y_true, y_prob 포함)
        cam_layer (str): "layer4" 또는 "layer3" (ResNet 타깃 레이어)
        cam_target (str): "pos" (logit z) 또는 "neg" (-z)
        overlay_alpha (float): CAM 오버레이 알파값
    """
    # --- 기본 경로 보정 ---
    if not cam_dir:  # None, "" 모두 처리
        base = out_dir if out_dir else "./02_results"
        cam_dir = os.path.join(base, "cam_test_trio")
    os.makedirs(cam_dir, exist_ok=True)

    # --------- 1) 테스트 데이터셋에서 원본 이미지/센터 인덱스 확보 ---------
    ds = dl_test.dataset  # SliceDataset2p5DMasked
    # SliceDataset2p5DMasked 내부에 다음 속성이 있다고 가정:
    #   - ds.X: np.ndarray, (N, H, W) 원본 슬라이스
    #   - ds.centers: np.ndarray[int], 길이=len(ds). 각 샘플의 중앙 슬라이스 전역 인덱스
    if hasattr(ds, "X"):
        X_source = ds.X
    elif hasattr(ds, "images"):
        # 혹시 클래스에 images로 되어 있으면 fallback
        X_source = ds.images
    else:
        raise AttributeError("SliceDataset2p5DMasked has neither 'X' nor 'images' attribute.")

    if not hasattr(ds, "centers"):
        raise AttributeError("SliceDataset2p5DMasked must expose 'centers' (center indices per sample).")

    centers = ds.centers
    y_true_te = test_metrics.get("y_true", None)
    y_prob_te = test_metrics.get("y_prob", None)

    # --------- 2) Grad-CAM 준비 (대상 모듈 선택) ---------
    target_module = getattr(model, cam_layer)  # e.g., model.layer4

    # 간단한 Grad-CAM 엔진 (필요 시 기존 GradCAM 클래스 사용 가능)
    def compute_cam(x_one: torch.Tensor, target: str = "pos"):
        """
        x_one: (1, C, H, W)  ← 여기서 C=2k+1 (2.5D 입력)
        target: 'pos' or 'neg'
        returns: (1, 1, H, W) in [0,1]
        """
        model.eval()
        model.zero_grad(set_to_none=True)

        feats = {}
        grads = {}

        def fwd_hook(m, i, o): feats["f"] = o
        def bwd_hook(m, gi, go): grads["g"] = go[0]

        h1 = target_module.register_forward_hook(fwd_hook)
        h2 = target_module.register_full_backward_hook(bwd_hook)

        try:
            with torch.enable_grad():
                out = model(x_one)            # (1, 1) logit
                score = out[0, 0] if target == "pos" else -out[0, 0]
                score.backward()

                A = feats["f"]                  # (1, Ck, h, w)
                dYdA = grads["g"]               # (1, Ck, h, w)
                w = torch.mean(dYdA, dim=(2, 3), keepdim=True)   # (1, Ck, 1, 1)
                cam = torch.sum(w * A, dim=1, keepdim=True)      # (1,1,h,w)
                cam = F.relu(cam)
                cam = F.interpolate(cam, size=(x_one.size(2), x_one.size(3)),
                                    mode="bilinear", align_corners=False)
                # normalize to [0,1]
                cam_min, cam_max = cam.min(), cam.max()
                if (cam_max - cam_min) > 1e-8:
                    cam = (cam - cam_min) / (cam_max - cam_min)
                else:
                    cam = torch.zeros_like(cam)
        finally:
            h1.remove(); h2.remove()
            model.zero_grad(set_to_none=True)

        return cam

    # --------- 3) CAM 생성 루프 ---------
    print(f"[CAM] Exporting overlays for TEST set -> {cam_dir}")

    # DataLoader 순서 == dataset 순서 (shuffle=False)
    # 배치 단위로 순회하면서, 배치 내 각 샘플의 center 인덱스를 이용해 원본 중앙 슬라이스를 얻음
    idx_global = 0
    with torch.no_grad():  # CAM 역전파는 compute_cam 내부에서 enable_grad로 처리됨. 여기선 추론 텐서를 float로 준비할 때만 사용.
        for batch, (x_batch, _) in enumerate(dl_test):
            B = x_batch.size(0)
            # x_batch: (B, 2k+1, H, W)  ← 2.5D 스택 텐서
            for b in range(B):
                center_idx = int(centers[idx_global])  # 전역 중앙 슬라이스 인덱스
                img_np = X_source[center_idx]          # (H, W) 원본 중앙 슬라이스 (float32 권장)
                lab = int(y_true_te[idx_global]) if y_true_te is not None else -1
                prob = float(y_prob_te[idx_global]) if y_prob_te is not None else float("nan")

                # Grad-CAM 입력은 배치에서 해당 샘플만 꺼내 사용
                x_one = x_batch[b:b+1].to(device)  # (1, 2k+1, H, W)

                # CAM 계산 (AMP 비활성화)
                with torch.amp.autocast("cuda", enabled=False):
                    heat = compute_cam(x_one, target=cam_target)  # (1,1,H,W)
                heat_np = heat[0, 0].detach().cpu().numpy().astype(np.float32)

                # 오버레이 저장
                # (show/save 함수는 프로젝트에 이미 있으리라 가정: save_cam_overlay(img_np, heat_np, path, alpha))
                fname = f"cam_test_{idx_global:05d}_center{center_idx}_y{lab}_p{prob:.3f}.png"
                out_path = os.path.join(cam_dir, fname)
                save_cam_overlay(img_np, heat_np, out_path, alpha=overlay_alpha)

                # (옵션) 원시 heatmap도 저장
                np.save(os.path.join(cam_dir, f"cam_test_{idx_global:05d}_center{center_idx}_heat.npy"), heat_np)

                idx_global += 1

    print(f"[CAM] Done. Saved {idx_global} overlays.")    