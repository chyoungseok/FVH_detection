import torch
import torch.nn.functional as F

class GradCAM:
    """
    Minimal Grad-CAM for PlainNet (uses the last conv block of ResNet).
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.target_module = self.model.backbone.layer4[-1].conv2
        self.hook_handlers = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inputs, output):
            self.activations = output.detach()
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handlers.append(self.target_module.register_forward_hook(fwd_hook))
        self.hook_handlers.append(self.target_module.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self.hook_handlers:
            h.remove()

    @torch.no_grad()
    def _normalize(self, x):
        x_min, x_max = x.min(), x.max()
        if (x_max - x_min) < 1e-6:
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    def __call__(self, x: torch.Tensor):
        """
        x: [1, 1, H, W]
        returns heatmap: [H, W] in [0,1]
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[0]
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [C,1,1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1,1,H',W']
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        heatmap = self._normalize(cam.squeeze(0).squeeze(0))
        return heatmap.cpu()
