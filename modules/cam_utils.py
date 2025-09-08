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

def save_cam_overlay(img_np: np.ndarray, heat_01: np.ndarray, out_path: str):
    """Save grayscale image overlaid with heatmap.
    img_np: (H,W) original npy image (before transform)
    heat_01: (H,W) float32 in [0,1]
    """
    base = img_np  # [0,1]
    plt.figure(figsize=(5,5))
    plt.imshow(base, cmap='gray')
    plt.imshow(heat_01, cmap='jet', alpha=0.35)
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
    
    