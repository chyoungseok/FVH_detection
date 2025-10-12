# vit_gru_pure.py
import torch
import torch.nn as nn
from einops import rearrange

# ---------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, patch_size=8):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                           # [B, D, H', W']
        x = rearrange(x, 'b d h w -> b (h w) d')   # [B, N, D]
        x = self.norm(x)
        return x


# ---------------------------------------------------------
# Transformer Encoder (MLP 제거 버전)
# ---------------------------------------------------------
class MiniViTBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Only Attention + Residual (no MLP)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout(attn_out)
        return x


class MiniViTEncoder(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, num_layers=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MiniViTBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return self.norm(x)


# ---------------------------------------------------------
# GRU + Classifier Head
# ---------------------------------------------------------
class ViTGRU(nn.Module):
    def __init__(self,
                 in_channels=1,
                 img_size=672,
                 patch_size=8,
                 embed_dim=64,
                 num_heads=4,
                 num_layers=8,
                 gru_hidden=1024,
                 bidirectional=False,
                 dropout=0.3,
                 num_classes=1):
        super().__init__()

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2, embed_dim))
        self.transformer = MiniViTEncoder(embed_dim, num_heads, num_layers, dropout)
        self.gru = nn.GRU(embed_dim, gru_hidden, num_layers=1, batch_first=True, bidirectional=bidirectional)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * (2 if bidirectional else 1), num_classes)
        )

    def forward(self, x):
        # x: [B, 1, 672, 672]
        x = self.patch_embed(x) + self.pos_embed        # [B, N, D]
        x = self.transformer(x)                         # [B, N, D]

        out, h = self.gru(x)
        if self.gru.bidirectional:
            h = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            h = h[-1]

        logits = self.fc(h)                             # [B, 1]
        return logits.squeeze(1)
