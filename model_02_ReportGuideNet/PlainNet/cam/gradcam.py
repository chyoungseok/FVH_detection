import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # hook 등록
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, class_score):
        # Grad-CAM 계산
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP over H, W
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.relu(cam)

        # Normalize 0~1
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam

def overlay_cam_on_image(img, cam):
    """img: numpy (H,W), cam: torch (1,H,W)"""
    cam = cam.squeeze().cpu().numpy()
    cam = cv2.resize(cam, img.shape[::-1])  # (W,H)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.4 * heatmap + 0.6 * np.stack([img*255]*3, axis=-1)
    return overlay.astype(np.uint8)