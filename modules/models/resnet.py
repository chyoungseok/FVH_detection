import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv
from torchvision import models


def _replace_first_conv_to_1ch(m: nn.Module) -> None:
    """ResNet 첫 conv 레이어를 1채널 입력으로 교체"""
    old = m.conv1
    new_conv = nn.Conv2d(
        1, old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False
    )
    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
    m.conv1 = new_conv


class resnet(nn.Module):
    """
    Simple CNN classifier for 2D FLAIR slices.
    backbone: resnet18/resnet34 (2D)
    head: GAP + Linear(1) -> use BCEWithLogitsLoss
    """
    def __init__(self, backbone: str = "resnet18", in_channels: int = 1, pretrained: bool = False) -> None:
        super().__init__()
        if backbone == "resnet18":
            m = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 512
        elif backbone == "resnet34":
            m = tv.resnet34(weights=tv.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 512
        else:
            raise ValueError("backbone must be 'resnet18' or 'resnet34'")

        if in_channels == 1:
            _replace_first_conv_to_1ch(m)

        m.fc = nn.Identity()
        self.backbone = m
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)          # [B, 512]
        logits = self.head(feats)         # [B, 1]
        return logits.squeeze(1)          # [B]