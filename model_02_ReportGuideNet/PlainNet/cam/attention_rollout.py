import torch
import torch.nn as nn
import types
import numpy as np
import cv2
from einops import rearrange
from swin_transformer_pytorch.swin_transformer import WindowAttention


# --- monkey-patch forward ---
def forward_with_attn(self, x):
    if self.shifted:
        x = self.cyclic_shift(x)

    b, n_h, n_w, _, h = *x.shape, self.heads
    qkv = self.to_qkv(x).chunk(3, dim=-1)
    nw_h = n_h // self.window_size
    nw_w = n_w // self.window_size

    q, k, v = map(
        lambda t: rearrange(
            t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
            h=h, w_h=self.window_size, w_w=self.window_size
        ), qkv
    )

    dots = torch.einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

    if self.relative_pos_embedding:
        dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
    else:
        dots += self.pos_embedding

    if self.shifted:
        dots[:, :, -nw_w:] += self.upper_lower_mask
        dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

    attn = dots.softmax(dim=-1)

    # --- save attention map ---
    self.last_attn = attn.detach()   # (B, heads, num_windows, ws², ws²)

    out = torch.einsum('b h w i j, b h w j d -> b h w i d', attn, v)
    out = rearrange(out,
        'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w
    )
    out = self.to_out(out)

    if self.shifted:
        out = self.cyclic_back_shift(out)
    return out


class AttentionRollout:
    def __init__(self, model: nn.Module, head_fusion="mean"):
        self.model = model
        self.head_fusion = head_fusion

        # monkey-patch WindowAttention
        for name, module in self.model.named_modules():
            if isinstance(module, WindowAttention):
                module.forward = types.MethodType(forward_with_attn, module)

    def __call__(self, x: torch.Tensor):
        _ = self.model(x)  # forward pass

        attn_maps = []
        for name, module in self.model.named_modules():
            if isinstance(module, WindowAttention) and hasattr(module, "last_attn"):
                attn = module.last_attn   # (B, heads, num_windows, ws², ws²)

                # head 평균
                if self.head_fusion == "mean":
                    attn = attn.mean(dim=1)   # (B, num_windows, ws², ws²)
                elif self.head_fusion == "max":
                    attn = attn.max(dim=1)[0]

                attn_maps.append(attn)

        if len(attn_maps) == 0:
            raise RuntimeError("No attention maps collected.")

        # 마지막 레이어 attention만 사용
        attn = attn_maps[-1]  # (B, num_windows, ws², ws²)
        B, num_windows, N, _ = attn.shape
        ws = int(np.sqrt(N))

        # 윈도우별로 원래 위치에 배치 (예: nw_h x nw_w 윈도우)
        nw = int(np.sqrt(num_windows))   # 가정: 정사각형 배치
        stitched = []
        for b in range(B):
            tiles = []
            for i in range(nw):
                row_tiles = []
                for j in range(nw):
                    idx = i * nw + j
                    tile = attn[b, idx]  # (ws², ws²)
                    tile = tile.mean(dim=0).view(ws, ws)  # 토큰 평균
                    row_tiles.append(tile)
                tiles.append(torch.cat(row_tiles, dim=1))
            stitched_map = torch.cat(tiles, dim=0)  # (nw*ws, nw*ws)
            stitched.append(stitched_map.unsqueeze(0))

        stitched = torch.cat(stitched, dim=0)  # (B, H, W)
        stitched = stitched / stitched.max()
        return stitched
