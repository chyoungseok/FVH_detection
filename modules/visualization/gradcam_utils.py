import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ======================================================
# 1️⃣ Grad-CAM Core
# ======================================================
class GradCAM:
    """
    Grad-CAM implementation for 2D CNNs (e.g., ResNet18/34)
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model (nn.Module): trained model
            target_layer (nn.Module): layer to hook (e.g., model.backbone.layer4[-1])
        """
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """register forward/backward hooks to capture activations & gradients"""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for a single input
        Args:
            input_tensor (torch.Tensor): [1, C, H, W]
        Returns:
            np.ndarray: normalized CAM heatmap (H, W)
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)  # forward

        # Binary classifier → use logit directly
        if target_class is None:
            # For binary, single output scalar
            if logits.ndim == 1:
                score = logits.mean()
            else:
                score = logits[:, 0].mean()
        else:
            score = logits[:, target_class].mean()

        score.backward()

        grads = self.gradients          # [1, C, H', W']
        activs = self.activations       # [1, C, H', W']

        weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP over spatial
        cam = (weights * activs).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ======================================================
# 2️⃣ Visualization Utility
# ======================================================
def overlay_cam_on_image(img: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    '''
    Overlay Grad-CAM heatmap on grayscale image

    Args:
        img (np.ndarray): (H, W), grayscale
        cam (np.ndarray): (H, W), normalized [0,1]
        alpha (float): blending factor

    Returns:
        np.ndarray: RGB overlay image [H, W, 3], float in [0,1]
    '''
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.0

    img_rgb = np.repeat(img[..., None], 3, axis=-1)
    img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)

    overlay = heatmap * alpha + img_rgb * (1 - alpha)
    overlay = np.clip(overlay, 0, 1)
    return overlay

# ======================================================
# Helper for batch processing & saving
# ======================================================
def generate_and_save_cam(model, gradcam, img_tensor, raw_img_np, sid, gt_label, pred_label, save_dir):
    '''
    Generate Grad-CAM for a single test image and save it

    Args:
        model (nn.Module)
        gradcam (GradCAM)
        img_tensor (torch.Tensor): [1, C, H, W]
        raw_img_np (np.ndarray): original grayscale image (H, W)
        sid (str): subject ID
        gt_label (int): ground truth label (0 or 1)
        pred_label (int): predicted label (0 or 1)
        save_dir (str): output directory
    '''
    
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits).item()

    # Re-enable grad for GradCAM
    img_tensor.requires_grad_(True)
    cam = gradcam.generate(img_tensor)

    # Overlay 생성
    overlay = overlay_cam_on_image(raw_img_np, cam)

    # ✅ 90도 회전 (시계 방향으로)
    overlay_rot = np.rot90(overlay, k=1)  # k=1: counter-clockwise, k=-1: clockwise

    # 파일 이름 구성
    fname = f"{sid}_GT{gt_label}_Pred{pred_label}_Prob{prob:.2f}.png"
    save_path = os.path.join(save_dir, fname)

    # 저장
    plt.imsave(save_path, overlay_rot)
    return save_path
