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


class AttentionPooling(nn.Module):
    """학습 가능한 attention pooling"""
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)

    def forward(self, x):  # x: (B, N, D)
        weights = F.softmax(self.attn(x), dim=1)  # (B, N, 1)
        pooled = (weights * x).sum(dim=1)         # (B, D)
        return pooled


class resnet_transformer(nn.Module):
    """
    ResNet feature extractor + Transformer encoder + classifier
    """
    def __init__(self, backbone="resnet18", in_channels=1, pretrained=False,
                 transformer_cfg=None, num_classes=1):
        super().__init__()
        transformer_cfg = transformer_cfg or {}
        embed_dim = transformer_cfg.get("d_model", 512)
        num_heads = transformer_cfg.get("nhead", 8)
        num_layers = transformer_cfg.get("num_layers", 2)
        pooling = transformer_cfg.get("pooling", "mean")

        # --- ResNet backbone ---
        resnet = getattr(models, backbone)(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if (pretrained and backbone == "resnet18") else None
        )
        if in_channels == 1:
            _replace_first_conv_to_1ch(resnet)

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # (B, C, H, W)
        feat_dim = list(resnet.children())[-1].in_features  # 512 for resnet18/34

        # --- Project to transformer embed dim ---
        self.proj = nn.Conv2d(feat_dim, embed_dim, kernel_size=1)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=transformer_cfg.get("dim_feedforward", 2048),
            dropout=transformer_cfg.get("dropout", 0.1),
            batch_first=True  # 중요: (B, N, D) 포맷
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- CLS token ---
        self.pooling = pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        # --- Attention pooling ---
        self.attn_pool = AttentionPooling(embed_dim) if pooling == "attention" else None

        # --- Classifier ---
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)              # (B, C, H, W)
        feat = self.proj(feat)                        # (B, D, H, W)
        B, D, H, W = feat.shape
        feat = feat.flatten(2).transpose(1, 2)        # (B, N, D)

        # CLS token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            feat = torch.cat([cls_tokens, feat], dim=1)

        feat = self.transformer(feat)  # (B, N, D)

        # Pooling
        if self.pooling == "mean":
            pooled = feat.mean(dim=1)
        elif self.pooling == "max":
            pooled = feat.max(dim=1).values
        elif self.pooling == "cls":
            pooled = feat[:, 0]
        elif self.pooling == "attention":
            pooled = self.attn_pool(feat)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.fc(pooled).squeeze(1)  # [B]
