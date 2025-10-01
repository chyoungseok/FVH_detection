import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import vit_b_16, VisionTransformer

class ViTBinaryClassifier(nn.Module):
    def __init__(self, pretrained: bool = True, image_size: int = 672):
        super().__init__()

        if pretrained:
            # ✅ ImageNet pretrained 모델 (224 고정)
            self.vit = vit_b_16(weights="IMAGENET1K_V1")
            self.image_size = 224
            self.input_proj = nn.Conv2d(1, 3, kernel_size=1)
            self.use_input_proj = True
        else:
            # ✅ scratch ViT (image_size 자유롭게 지정 가능)
            self.vit = VisionTransformer(
                image_size=image_size,
                patch_size=16,
                num_layers=12,
                num_heads=12,
                hidden_dim=768,
                mlp_dim=3072,
                num_classes=1000
            )
            self.image_size = image_size
            old_conv = self.vit.conv_proj
            self.vit.conv_proj = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            nn.init.kaiming_normal_(self.vit.conv_proj.weight, mode="fan_out", nonlinearity="relu")
            self.use_input_proj = False

        # ✅ 분류 head 수정
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 입력 resize (pretrained 모델일 때만 224로 강제)
        if self.use_input_proj:
            x = self.input_proj(x)  # (B,1,H,W) → (B,3,H,W)
            if x.shape[2] != self.image_size or x.shape[3] != self.image_size:
                x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        logits = self.vit(x)
        return logits.squeeze(1)
