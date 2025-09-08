# modules/ig_utils.py
# -*- coding: utf-8 -*-
"""
Integrated Gradients (IG) utilities for FHV slice models (PyTorch + Captum)

핵심 기능
- 1ch/3ch 자동 대응: 첫 Conv2d in_channels 자동 추론 → 입력 채널 자동 매칭(3→1 평균, 1→3 복제)
- 1-logit(sigmoid)/multi-logit(softmax) 자동 처리 + pos_class_idx 명시 옵션
- Baseline 다양화: zeros / constant(c) / blur(k) / multi-baseline 평균
- NoiseTunnel(smoothgrad / smoothgrad_sq) 옵션 지원
- Completeness(보존성) 수치 검증 및 CSV 로깅
- Occlusion drop(충실성) 검증 및 CSV 로깅 (상위 k% 마스킹 후 점수 하락 측정)
- 시각화 확장: viz_sign(positive/negative/both/absolute), display_mode(percentile/zscore)
- 원본 해상도 overlay 옵션 (입력 리사이즈 전/후 선택)
- per-sample / per-batch IG 실행 모드
- 단일 슬라이스/배치 API 제공 + 파일 저장

Author: YS helpers (refined)
"""

from __future__ import annotations
import os, csv, math
from typing import Any, Iterable, Optional, Tuple, Union, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel

import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn as tnn


# =============================================================================
# Filesystem & Logging
# =============================================================================
def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _csv_append(path: str, header: List[str], row: List[Any]):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(header)
        writer.writerow(row)


# =============================================================================
# Channel & Model utilities
# =============================================================================
def _infer_first_conv_in_channels(model: nn.Module, fallback: Optional[int] = None) -> int:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return m.in_channels
    return 3 if fallback is None else fallback


def _match_channels(x: torch.Tensor, expected_c: int) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"x must be (B,C,H,W), got {tuple(x.shape)}")
    B, C, H, W = x.shape
    if expected_c == C:
        return x
    if expected_c == 1 and C == 3:
        return x.mean(dim=1, keepdim=True)
    if expected_c == 3 and C == 1:
        return x.repeat(1, 3, 1, 1)
    if expected_c < C:
        return x[:, :expected_c]
    reps = (expected_c + C - 1) // C
    return x.repeat(1, reps, 1, 1)[:, :expected_c]


def _sanitize_denorm_mean_std(
    mean: Optional[Union[Tuple[float, ...], List[float]]],
    std: Optional[Union[Tuple[float, ...], List[float]]],
    expected_c: int
) -> Tuple[Optional[Tuple[float, ...]], Optional[Tuple[float, ...]]]:
    if mean is None or std is None:
        return None, None
    try:
        mean = tuple(float(v) for v in mean)
        std = tuple(float(v) for v in std)
    except Exception:
        return None, None
    if len(mean) == expected_c and len(std) == expected_c:
        return mean, std
    if expected_c == 1:
        m = (float(np.mean(mean)),)
        s = (float(np.mean(std)),)
        return m, s
    return None, None


# =============================================================================
# Resize helpers (for original resolution overlay)
# =============================================================================
def _resize_like_numpy(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """
    img: (H,W) or (H,W,3) or (C,H,W)
    out_hw: (H_out, W_out)
    간단히 PyTorch interpolate로 크기 변환(최근접/쌍선형).
    """
    H_out, W_out = out_hw
    if img.ndim == 2:
        t = torch.from_numpy(img)[None, None, ...].float()
        r = torch.nn.functional.interpolate(t, size=(H_out, W_out), mode="bilinear", align_corners=False)
        return r[0, 0].cpu().numpy()
    if img.ndim == 3 and img.shape[0] in (1, 3):  # (C,H,W)
        t = torch.from_numpy(img)[None, ...].float()
        r = torch.nn.functional.interpolate(t, size=(H_out, W_out), mode="bilinear", align_corners=False)
        return r[0].cpu().numpy()
    if img.ndim == 3 and img.shape[-1] in (1, 3):  # (H,W,3)
        t = torch.from_numpy(img).permute(2, 0, 1)[None, ...].float()
        r = torch.nn.functional.interpolate(t, size=(H_out, W_out), mode="bilinear", align_corners=False)
        return r[0].permute(1, 2, 0).cpu().numpy()
    raise ValueError(f"Unsupported shape for resize: {img.shape}")


# =============================================================================
# Visualization helpers
# =============================================================================
def _to_numpy_img_for_display(
    x: torch.Tensor,
    mean: Optional[Tuple[float, ...]],
    std: Optional[Tuple[float, ...]],
    display_mode: str = "percentile",
    z_k: float = 3.0
) -> List[np.ndarray]:
    """
    (B,C,H,W) → list of (H,W) gray images.
    display_mode: 'percentile' (1..99) | 'zscore' (μ±kσ)
    """
    x = x.detach().cpu().float()
    B, C, H, W = x.shape

    if mean is not None and std is not None and len(mean) == C and len(std) == C:
        mean_t = torch.tensor(mean, dtype=x.dtype).view(1, C, 1, 1)
        std_t = torch.tensor(std, dtype=x.dtype).view(1, C, 1, 1)
        x = x * std_t + mean_t

    gray = x.mean(dim=1)  # (B,H,W)

    imgs: List[np.ndarray] = []
    for i in range(B):
        g = gray[i].numpy()
        if display_mode == "zscore":
            mu, sigma = float(g.mean()), float(g.std() + 1e-8)
            g = np.clip((g - (mu - z_k * sigma)) / (2 * z_k * sigma + 1e-8), 0, 1)
        else:
            p1, p99 = np.percentile(g, 1), np.percentile(g, 99)
            g = np.clip((g - p1) / (p99 - p1 + 1e-8), 0, 1)
        imgs.append(g)
    return imgs


def _viz_sign_map(attr_np: np.ndarray, viz_sign: str = "positive") -> np.ndarray:
    """
    attr_np: (C,H,W) or (H,W)
    viz_sign: 'positive' | 'negative' | 'both' | 'absolute'
      - 'both'은 시각화 단계에서 bwr 적용 권장 (여기선 절대값 normalize에 맡김)
    """
    if attr_np.ndim == 3:
        A = attr_np.sum(axis=0)  # (H,W)
    else:
        A = attr_np

    if viz_sign == "negative":
        A = -np.minimum(A, 0.0)
    elif viz_sign == "absolute":
        A = np.abs(A)
    elif viz_sign == "both":
        # 반환은 그대로 두고, 컬러맵을 bwr로 쓰는 것을 권장
        return A
    else:  # 'positive'
        A = np.maximum(A, 0.0)
    return A


def _normalize_01(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m, M = float(A.min()), float(A.max())
    return (A - m) / (M - m + eps) if M > m else np.zeros_like(A)


def _overlay_heatmap(
    gray_hw: np.ndarray,
    heat_hw: np.ndarray,
    alpha: float = 0.5,
    cmap: str = "jet"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    gray_hw: (H,W) 0..1
    heat_hw: (H,W) 0..1 (양/음 혼합 가능) → cmap에 따라 표현
    """
    cmap_fn = cm.get_cmap(cmap)
    heat_rgb = cmap_fn(heat_hw)[..., :3]
    gray_rgb = np.stack([gray_hw] * 3, axis=-1)
    out = (1 - alpha) * gray_rgb + alpha * heat_rgb
    return gray_rgb, heat_rgb, out


# =============================================================================
# Forward wrapper for 1-logit (sigmoid) heads
# =============================================================================
class _ForwardForSigmoid(nn.Module):
    """
    Single-logit head:
      mode='pos' -> +logit
      mode='neg' -> -logit
    """
    def __init__(self, model: nn.Module, mode: str = 'pos'):
        super().__init__()
        self.model = model
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        if logits.ndim == 2 and logits.shape[1] == 1:
            s = logits[:, 0]
        elif logits.ndim == 1:
            s = logits
        else:
            raise RuntimeError("This wrapper is only for 1-logit models.")
        return s if self.mode == 'pos' else -s


# =============================================================================
# Predict helper
# =============================================================================
@torch.no_grad()
def _predict_class_and_score(model: nn.Module, x: torch.Tensor, pos_class_idx: int = 1) -> Tuple[int, float]:
    y = model(x)
    if y.ndim == 2 and y.shape[1] > 1:
        p = F.softmax(y, dim=1)
        pred = int(p.argmax(dim=1).item())
        score = float(p[0, pos_class_idx].item())  # report positive class prob
        return pred, score
    prob = torch.sigmoid(y).flatten()[0]
    pred = 1 if float(prob) >= 0.5 else 0
    return pred, float(prob)


# =============================================================================
# Baselines
# =============================================================================
def _make_baseline(x: torch.Tensor, mode: str = "zeros", const_val: float = 0.0, blur_kernel: int = 7) -> torch.Tensor:
    """
    x: (B,C,H,W)
    mode: 'zeros' | 'constant' | 'blur'
    """
    if mode == "zeros":
        return torch.zeros_like(x)

    if mode == "constant":
        base = torch.zeros_like(x) + const_val
        return base

    if mode == "blur":
        pad = blur_kernel // 2
        avg = tnn.AvgPool2d(kernel_size=blur_kernel, stride=1, padding=pad)
        with torch.no_grad():
            return avg(x.detach())

    raise ValueError(f"Unknown baseline mode: {mode}")


# =============================================================================
# Core IG attribution (with optional NoiseTunnel)
# =============================================================================
def _compute_ig(
    model: nn.Module,
    x: torch.Tensor,                  # (B,C,H,W)
    baseline: torch.Tensor,           # (B,C,H,W)
    *,
    is_sigmoid_head: bool,
    ig_steps: int,
    ig_method: str = "riemann_trapezoid",
    ig_target_mode: str = "pos",      # 'pos' or 'neg'
    pos_class_idx: int = 1,
    use_noise_tunnel: bool = False,
    nt_type: str = "smoothgrad_sq",
    nt_samples: int = 8,
    stdevs: float = 0.02,
) -> Tuple[torch.Tensor, Any]:
    """
    Returns:
      attributions: (B,C,H,W)
      delta: Captum's convergence delta(s)
    """
    if is_sigmoid_head:
        forward_fn = _ForwardForSigmoid(model, mode=('pos' if ig_target_mode == 'pos' else 'neg'))
        ig = IntegratedGradients(forward_fn)
        target_for_multilogit = None
    else:
        ig = IntegratedGradients(model)
        target_for_multilogit = pos_class_idx if ig_target_mode == 'pos' else (1 - pos_class_idx)

    if use_noise_tunnel:
        ig = NoiseTunnel(ig)

    torch.set_grad_enabled(True)
    if use_noise_tunnel:
        attributions = ig.attribute(
            inputs=x,
            baselines=baseline,
            target=(None if is_sigmoid_head else target_for_multilogit),
            n_steps=ig_steps,
            nt_type=nt_type,
            nt_samples=nt_samples,
            stdevs=stdevs,
            method=ig_method,
        )
        delta = 0.0  # NoiseTunnel은 delta를 반환하지 않음
    else:
        attributions, delta = ig.attribute(
            inputs=x,
            baselines=baseline,
            target=(None if is_sigmoid_head else target_for_multilogit),
            n_steps=ig_steps,
            method=ig_method,
            return_convergence_delta=True
        )
    torch.set_grad_enabled(False)
    return attributions, delta


# =============================================================================
# Completeness check & Occlusion
# =============================================================================
@torch.no_grad()
def _completeness_error(
    model: nn.Module,
    x: torch.Tensor,                 # (1,C,H,W)
    baseline: torch.Tensor,          # (1,C,H,W)
    attrs: torch.Tensor,             # (1,C,H,W)
    *,
    is_sigmoid_head: bool,
    pos_class_idx: int
) -> Tuple[float, float, float]:
    """
    Returns: (ig_sum, fx_diff, abs_err)
    """
    y_x = model(x)
    y_b = model(baseline)
    if not is_sigmoid_head:
        y_x = F.softmax(y_x, dim=1)[0, pos_class_idx]
        y_b = F.softmax(y_b, dim=1)[0, pos_class_idx]
    else:
        y_x = torch.sigmoid(y_x).flatten()[0]
        y_b = torch.sigmoid(y_b).flatten()[0]
    ig_sum = float(attrs.sum().item())
    fx_diff = float((y_x - y_b).item())
    abs_err = abs(ig_sum - fx_diff)
    return ig_sum, fx_diff, abs_err


@torch.no_grad()
def _occlusion_drop(
    model: nn.Module,
    x: torch.Tensor,                 # (1,C,H,W)
    attrs: torch.Tensor,             # (1,C,H,W)
    pos_class_idx: int,
    k_ratio: float = 0.02,
    mode: str = "zero",              # 'zero' | 'blur'
) -> Tuple[float, float, float]:
    """
    Returns: (score_before, score_after, drop) for positive class.
    """
    a = attrs.detach().abs().sum(dim=1)[0]  # (H,W)
    H, W = a.shape
    th = np.percentile(a.cpu().numpy(), 100 * (1 - k_ratio))
    mask = (a >= th).float()[None, None, ...]  # (1,1,H,W)

    if mode == "zero":
        x_masked = x * (1 - mask)
    else:
        blur = tnn.AvgPool2d(7, stride=1, padding=3)
        xb = blur(x)
        x_masked = x * (1 - mask) + xb * mask

    y0 = model(x)
    y1 = model(x_masked)

    if y0.ndim == 2 and y0.shape[1] > 1:
        p0 = F.softmax(y0, dim=1)[0, pos_class_idx].item()
        p1 = F.softmax(y1, dim=1)[0, pos_class_idx].item()
    else:
        p0 = torch.sigmoid(y0).flatten()[0].item()
        p1 = torch.sigmoid(y1).flatten()[0].item()

    return p0, p1, (p0 - p1)


# =============================================================================
# Batch unpack helper
# =============================================================================
def _extract_images_from_batch(batch: Any) -> torch.Tensor:
    if isinstance(batch, (list, tuple)):
        return batch[0]
    if isinstance(batch, dict):
        for k in ("images", "image", "x", "inputs"):
            if k in batch:
                return batch[k]
        for v in batch.values():
            if torch.is_tensor(v):
                return v
        raise ValueError("No tensor found in batch dict.")
    if torch.is_tensor(batch):
        return batch
    raise ValueError(f"Unsupported batch type: {type(batch)}")


# =============================================================================
# Public APIs
# =============================================================================
def ig_analysis(
    out_dir: str,
    model: nn.Module,
    dl_test: Iterable,
    device: Union[str, torch.device] = "cuda",
    *,
    export_ig: bool = True,
    ig_dir: Optional[str] = None,
    ig_steps: int = 128,
    ig_method: str = "riemann_trapezoid",
    ig_alpha: float = 0.5,
    ig_cmap: str = "jet",
    ig_target: str = "pos",                # 'pos' or 'neg'
    pos_class_idx: int = 1,                # positive class index for multi-logit
    viz_sign: str = "positive",            # 'positive'|'negative'|'both'|'absolute'
    ig_positive_only: Optional[bool] = None,  # deprecated; use viz_sign
    mean_for_denorm: Optional[Union[Tuple[float, ...], List[float]]] = None,
    std_for_denorm: Optional[Union[Tuple[float, ...], List[float]]] = None,
    display_mode: str = "percentile",      # 'percentile'|'zscore'
    z_k: float = 3.0,
    save_raw_attr: bool = False,
    # baseline options
    baseline_mode: str = "zeros",          # 'zeros'|'constant'|'blur'|'multi'
    const_val: float = 0.0,
    blur_kernel: int = 7,
    multi_baselines: int = 0,              # if >0 and baseline_mode != 'multi', still average (zeros + noise)
    multi_noise_std: float = 0.0,
    # NoiseTunnel
    use_noise_tunnel: bool = False,
    nt_type: str = "smoothgrad_sq",
    nt_samples: int = 8,
    stdevs: float = 0.02,
    # original resolution overlay
    resize_to: str = "input",              # 'input'|'original'
    orig_hw: Optional[Tuple[int, int]] = None,
    # evaluation logs
    log_completeness_csv: Optional[str] = None,
    completeness_tol: float = 1e-2,
    run_occlusion_eval: bool = False,
    occl_k: float = 0.02,
    occl_mode: str = "zero",               # 'zero'|'blur'
) -> List[str]:
    """
    Run IG over dl_test (per-sample loop) and save overlay PNGs.
    Returns list of saved PNG paths.
    """
    if not export_ig:
        return []

    device = torch.device(device)
    model = model.to(device).eval()

    ig_save_dir = ig_dir or os.path.join(out_dir, "ig_test")
    _ensure_dir(ig_save_dir)

    expected_c = _infer_first_conv_in_channels(model)
    mean_for_denorm, std_for_denorm = _sanitize_denorm_mean_std(mean_for_denorm, std_for_denorm, expected_c)

    # Head probe
    with torch.no_grad():
        first_batch = next(iter(dl_test))
        sample = _extract_images_from_batch(first_batch).to(device)
        sample = _match_channels(sample, expected_c)
        head_out = model(sample[:1])
        is_sigmoid_head = (head_out.ndim == 1) or (head_out.ndim == 2 and head_out.shape[1] == 1)

    saved_paths: List[str] = []
    idx_global = 0

    # CSV header (completeness/occlusion)
    if log_completeness_csv:
        _ensure_dir(os.path.dirname(log_completeness_csv))
        _csv_append(
            log_completeness_csv,
            header=["index", "ig_sum", "F(x)-F(b)", "abs_err", "delta", "warn_tol", "occl_before", "occl_after", "occl_drop"],
            row=[],
        )  # header only; first row empty prevents duplicate header

    for batch in dl_test:
        images = _extract_images_from_batch(batch).to(device)
        images = _match_channels(images, expected_c)

        B = images.shape[0]
        for b in range(B):
            x = images[b:b+1]  # (1,C,H,W)

            # --- baseline(s)
            if baseline_mode == "multi" or multi_baselines > 0:
                M = max(1, multi_baselines)
                attrs_accum = None
                delta_accum = 0.0
                for m in range(M):
                    base = _make_baseline(x, mode=("zeros" if baseline_mode == "multi" else baseline_mode),
                                          const_val=const_val, blur_kernel=blur_kernel)
                    if multi_noise_std > 0.0:
                        base = base + torch.randn_like(base) * multi_noise_std
                    attributions, delta = _compute_ig(
                        model, x, base,
                        is_sigmoid_head=is_sigmoid_head,
                        ig_steps=ig_steps,
                        ig_method=ig_method,
                        ig_target_mode=ig_target,
                        pos_class_idx=pos_class_idx,
                        use_noise_tunnel=use_noise_tunnel,
                        nt_type=nt_type, nt_samples=nt_samples, stdevs=stdevs
                    )
                    attrs_accum = attributions if attrs_accum is None else attrs_accum + attributions
                    try:
                        delta_accum = delta_accum + float(delta.detach().cpu().item() if torch.is_tensor(delta) else float(delta))
                    except Exception:
                        pass
                attributions = attrs_accum / float(M)
                delta_val = delta_accum / float(M) if not use_noise_tunnel else 0.0
                baseline = _make_baseline(x, mode=("zeros" if baseline_mode == "multi" else baseline_mode),
                                          const_val=const_val, blur_kernel=blur_kernel)
            else:
                baseline = _make_baseline(x, mode=baseline_mode, const_val=const_val, blur_kernel=blur_kernel)
                attributions, delta = _compute_ig(
                    model, x, baseline,
                    is_sigmoid_head=is_sigmoid_head,
                    ig_steps=ig_steps,
                    ig_method=ig_method,
                    ig_target_mode=ig_target,
                    pos_class_idx=pos_class_idx,
                    use_noise_tunnel=use_noise_tunnel,
                    nt_type=nt_type, nt_samples=nt_samples, stdevs=stdevs
                )
                delta_val = float(delta.detach().cpu().item() if torch.is_tensor(delta) else float(delta) if not use_noise_tunnel else 0.0)

            # numpy & viz map
            attr_np = attributions.squeeze(0).detach().cpu().numpy()  # (C,H,W)
            A = _viz_sign_map(attr_np, viz_sign=viz_sign)             # (H,W) possibly signed
            if viz_sign == "both":
                # both의 경우, [-, +]를 0..1로 매핑하려면 미리 -1..1 → 0..1 스케일
                A01 = 0.5 + 0.5 * (A / (np.abs(A).max() + 1e-8))
                cmap_use = "bwr"
            else:
                A01 = _normalize_01(A)
                cmap_use = ig_cmap

            # background gray
            gray_list = _to_numpy_img_for_display(x, mean_for_denorm, std_for_denorm,
                                                  display_mode=display_mode, z_k=z_k)
            gray = gray_list[0]

            # (선택) 원본 해상도로 리사이즈
            heat_for_overlay = A01
            gray_for_overlay = gray
            if resize_to == "original" and orig_hw is not None:
                heat_for_overlay = _resize_like_numpy(heat_for_overlay, orig_hw)
                gray_for_overlay = _resize_like_numpy(gray_for_overlay, orig_hw)

            gray_rgb, heat_rgb, overlay_rgb = _overlay_heatmap(gray_for_overlay, heat_for_overlay, alpha=ig_alpha, cmap=cmap_use)

            pred, pos_score = _predict_class_and_score(model, x, pos_class_idx=pos_class_idx)

            # save fig
            fig = plt.figure(figsize=(12, 4), dpi=140)
            ax1 = plt.subplot(1, 3, 1); ax1.imshow(gray_for_overlay if gray_for_overlay.ndim==2 else gray_rgb); ax1.set_title("Input (gray)"); ax1.axis("off")
            ax2 = plt.subplot(1, 3, 2); ax2.imshow(heat_rgb); ax2.set_title("IG Heatmap"); ax2.axis("off")
            ax3 = plt.subplot(1, 3, 3); ax3.imshow(overlay_rgb); ax3.set_title(f"Overlay (pred={pred}, pos_score={pos_score:.3f}, δ={delta_val:.2e})"); ax3.axis("off")
            plt.tight_layout(pad=1.0)

            png_path = os.path.join(ig_save_dir, f"ig_{idx_global:06d}.png")
            fig.savefig(png_path, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(png_path)

            if save_raw_attr:
                np.save(os.path.join(ig_save_dir, f"ig_attr_{idx_global:06d}.npy"), attr_np)

            # completeness & occlusion logs
            ig_sum, fx_diff, abs_err = _completeness_error(
                model, x, baseline, attributions,
                is_sigmoid_head=is_sigmoid_head, pos_class_idx=pos_class_idx
            )
            oc0 = oc1 = drop = math.nan
            if run_occlusion_eval:
                oc0, oc1, drop = _occlusion_drop(
                    model, x, attributions, pos_class_idx=pos_class_idx, k_ratio=occl_k, mode=occl_mode
                )

            if log_completeness_csv:
                _csv_append(
                    log_completeness_csv,
                    header=["index", "ig_sum", "F(x)-F(b)", "abs_err", "delta", "warn_tol", "occl_before", "occl_after", "occl_drop"],
                    row=[idx_global, ig_sum, fx_diff, abs_err, delta_val, (abs_err > completeness_tol), oc0, oc1, drop],
                )

            idx_global += 1

    return saved_paths


# -----------------------------------------------------------------------------
# Single-slice API
# -----------------------------------------------------------------------------
def _prepare_input_from_slice(
    slice_2d: Union[np.ndarray, torch.Tensor],
    expected_c: int,
    device: Union[str, torch.device] = "cuda",
) -> torch.Tensor:
    if isinstance(slice_2d, np.ndarray):
        t = torch.from_numpy(slice_2d)
    else:
        t = slice_2d
    if t.ndim == 2:       # (H,W)
        t = t.unsqueeze(0)
    if t.ndim == 3:       # (C,H,W)
        t = t.unsqueeze(0)
    if t.ndim != 4:
        raise ValueError(f"slice must be (H,W) or (C,H,W), got shape {tuple(t.shape)}")
    t = t.float().to(device)
    t = _match_channels(t, expected_c)
    return t


def ig_on_single_slice(
    model: nn.Module,
    slice_2d: Union[np.ndarray, torch.Tensor],
    device: Union[str, torch.device] = "cuda",
    *,
    ig_steps: int = 128,
    ig_method: str = "riemann_trapezoid",
    ig_alpha: float = 0.5,
    ig_cmap: str = "jet",
    ig_target: str = "pos",
    pos_class_idx: int = 1,
    viz_sign: str = "positive",
    mean_for_denorm: Optional[Union[Tuple[float, ...], List[float]]] = None,
    std_for_denorm: Optional[Union[Tuple[float, ...], List[float]]] = None,
    display_mode: str = "percentile",
    z_k: float = 3.0,
    baseline_mode: str = "zeros",
    const_val: float = 0.0,
    blur_kernel: int = 7,
    multi_baselines: int = 0,
    multi_noise_std: float = 0.0,
    use_noise_tunnel: bool = False,
    nt_type: str = "smoothgrad_sq",
    nt_samples: int = 8,
    stdevs: float = 0.02,
    resize_to: str = "input",
    orig_hw: Optional[Tuple[int, int]] = None,
    save_png_path: Optional[str] = None,
    return_figure: bool = False,
) -> Dict[str, Any]:
    device = torch.device(device)
    model = model.to(device).eval()
    expected_c = _infer_first_conv_in_channels(model)
    mean_for_denorm, std_for_denorm = _sanitize_denorm_mean_std(mean_for_denorm, std_for_denorm, expected_c)

    x = _prepare_input_from_slice(slice_2d, expected_c, device)  # (1,C,H,W)

    # head probe
    with torch.no_grad():
        out = model(x)
        is_sigmoid_head = (out.ndim == 1) or (out.ndim == 2 and out.shape[1] == 1)

    # baselines & IG
    if multi_baselines > 0 or baseline_mode == "multi":
        M = max(1, multi_baselines)
        attrs_accum = None
        for m in range(M):
            base = _make_baseline(x, mode=("zeros" if baseline_mode == "multi" else baseline_mode),
                                  const_val=const_val, blur_kernel=blur_kernel)
            if multi_noise_std > 0.0:
                base = base + torch.randn_like(base) * multi_noise_std
            attrs, _ = _compute_ig(
                model, x, base,
                is_sigmoid_head=is_sigmoid_head,
                ig_steps=ig_steps,
                ig_method=ig_method,
                ig_target_mode=ig_target,
                pos_class_idx=pos_class_idx,
                use_noise_tunnel=use_noise_tunnel,
                nt_type=nt_type, nt_samples=nt_samples, stdevs=stdevs
            )
            attrs_accum = attrs if attrs_accum is None else attrs_accum + attrs
        attributions = attrs_accum / float(M)
        baseline = _make_baseline(x, mode=("zeros" if baseline_mode == "multi" else baseline_mode),
                                  const_val=const_val, blur_kernel=blur_kernel)
        delta_val = 0.0
    else:
        baseline = _make_baseline(x, mode=baseline_mode, const_val=const_val, blur_kernel=blur_kernel)
        attributions, delta = _compute_ig(
            model, x, baseline,
            is_sigmoid_head=is_sigmoid_head,
            ig_steps=ig_steps,
            ig_method=ig_method,
            ig_target_mode=ig_target,
            pos_class_idx=pos_class_idx,
            use_noise_tunnel=use_noise_tunnel,
            nt_type=nt_type, nt_samples=nt_samples, stdevs=stdevs
        )
        delta_val = float(delta.detach().cpu().item() if torch.is_tensor(delta) else float(delta) if not use_noise_tunnel else 0.0)

    # numpy & visualization
    attr_np = attributions.squeeze(0).detach().cpu().numpy()   # (C,H,W)
    A = _viz_sign_map(attr_np, viz_sign=viz_sign)
    if viz_sign == "both":
        A01 = 0.5 + 0.5 * (A / (np.abs(A).max() + 1e-8))
        cmap_use = "bwr"
    else:
        A01 = _normalize_01(A)
        cmap_use = ig_cmap

    gray_list = _to_numpy_img_for_display(x, mean_for_denorm, std_for_denorm,
                                          display_mode=display_mode, z_k=z_k)
    gray = gray_list[0]

    heat_for_overlay = A01
    gray_for_overlay = gray
    if resize_to == "original" and orig_hw is not None:
        heat_for_overlay = _resize_like_numpy(heat_for_overlay, orig_hw)
        gray_for_overlay = _resize_like_numpy(gray_for_overlay, orig_hw)

    gray_rgb, heat_rgb, overlay_rgb = _overlay_heatmap(gray_for_overlay, heat_for_overlay, alpha=ig_alpha, cmap=cmap_use)
    pred, pos_score = _predict_class_and_score(model, x, pos_class_idx=pos_class_idx)

    # completeness/occlusion (single)
    ig_sum, fx_diff, abs_err = _completeness_error(model, x, baseline, attributions,
                                                   is_sigmoid_head=is_sigmoid_head, pos_class_idx=pos_class_idx)

    oc0, oc1, drop = _occlusion_drop(model, x, attributions, pos_class_idx=pos_class_idx, k_ratio=0.02, mode="zero")

    # save figure
    fig = None
    if save_png_path is not None or return_figure:
        fig = plt.figure(figsize=(12, 4), dpi=140)
        ax1 = plt.subplot(1, 3, 1); ax1.imshow(gray_for_overlay if gray_for_overlay.ndim==2 else gray_rgb); ax1.set_title("Input (gray)"); ax1.axis("off")
        ax2 = plt.subplot(1, 3, 2); ax2.imshow(heat_rgb); ax2.set_title("IG Heatmap"); ax2.axis("off")
        ax3 = plt.subplot(1, 3, 3); ax3.imshow(overlay_rgb); ax3.set_title(f"Overlay (pred={pred}, pos_score={pos_score:.3f}, δ={delta_val:.2e})"); ax3.axis("off")
        plt.tight_layout(pad=1.0)

        if save_png_path is not None:
            _ensure_dir(os.path.dirname(save_png_path))
            fig.savefig(save_png_path, bbox_inches="tight")
            if not return_figure:
                plt.close(fig)

    return {
        "attr": attr_np,
        "heat01": A01 if viz_sign != "both" else None,  # both는 signed → heat01 제공 안함
        "gray_rgb": gray_rgb,
        "heat_rgb": heat_rgb,
        "overlay_rgb": overlay_rgb,
        "pred": pred,
        "pos_score": pos_score,
        "delta": delta_val,
        "ig_sum": ig_sum,
        "fx_diff": fx_diff,
        "completeness_abs_err": abs_err,
        "occl_before": oc0,
        "occl_after": oc1,
        "occl_drop": drop,
        "figure": fig,
    }


# -----------------------------------------------------------------------------
# Batch API (run IG on full batch once)
# -----------------------------------------------------------------------------
def ig_on_batch(
    model: nn.Module,
    batch_images: torch.Tensor,                 # (B,C,H,W)
    device: Union[str, torch.device] = "cuda",
    *,
    out_dir: Optional[str] = None,              # if set, saves to out_dir/ig_batch
    fname_prefix: str = "igb_",
    fname_index = 0,
    ig_steps: int = 128,
    ig_method: str = "riemann_trapezoid",
    ig_alpha: float = 0.5,
    ig_cmap: str = "jet",
    ig_target: str = "pos",
    pos_class_idx: int = 1,
    viz_sign: str = "positive",
    mean_for_denorm: Optional[Union[Tuple[float, ...], List[float]]] = None,
    std_for_denorm: Optional[Union[Tuple[float, ...], List[float]]] = None,
    display_mode: str = "percentile",
    z_k: float = 3.0,
    baseline_mode: str = "zeros",
    const_val: float = 0.0,
    blur_kernel: int = 7,
    use_noise_tunnel: bool = False,
    nt_type: str = "smoothgrad_sq",
    nt_samples: int = 8,
    stdevs: float = 0.02,
    return_figures: bool = False,
    save_raw_attr: bool = False,
) -> List[Dict[str, Any]]:
    device = torch.device(device)
    model = model.to(device).eval()

    expected_c = _infer_first_conv_in_channels(model)
    mean_for_denorm, std_for_denorm = _sanitize_denorm_mean_std(mean_for_denorm, std_for_denorm, expected_c)

    if not torch.is_tensor(batch_images) or batch_images.ndim != 4:
        raise ValueError("batch_images must be (B,C,H,W) torch.Tensor")
    x = _match_channels(batch_images.to(device).float(), expected_c)
    B = x.shape[0]

    with torch.no_grad():
        head_out = model(x[:1])
        is_sigmoid_head = (head_out.ndim == 1) or (head_out.ndim == 2 and head_out.shape[1] == 1)

    baseline = _make_baseline(x, mode=baseline_mode, const_val=const_val, blur_kernel=blur_kernel)

    attributions, delta = _compute_ig(
        model, x, baseline,
        is_sigmoid_head=is_sigmoid_head,
        ig_steps=ig_steps,
        ig_method=ig_method,
        ig_target_mode=ig_target,
        pos_class_idx=pos_class_idx,
        use_noise_tunnel=use_noise_tunnel,
        nt_type=nt_type, nt_samples=nt_samples, stdevs=stdevs
    )
    delta_list: List[float]
    if isinstance(delta, torch.Tensor):
        delta_list = [float(d) for d in delta.detach().cpu().flatten()]
    else:
        delta_list = [float(delta) if not use_noise_tunnel else 0.0] * B

    save_dir = None
    if out_dir is not None:
        save_dir = os.path.join(out_dir, "ig_batch")
        _ensure_dir(save_dir)

    attr_np_all = attributions.detach().cpu().numpy()  # (B,C,H,W)
    gray_all = _to_numpy_img_for_display(x, mean_for_denorm, std_for_denorm, display_mode=display_mode, z_k=z_k)

    results: List[Dict[str, Any]] = []

    # predictions (vectorized)
    with torch.no_grad():
        y = model(x)
        if not is_sigmoid_head:
            p = F.softmax(y, dim=1)
            pos_scores = p[:, pos_class_idx]
            preds = p.argmax(dim=1)
        else:
            pos_scores = torch.sigmoid(y).flatten()
            preds = (pos_scores >= 0.5).long()
    pos_scores = pos_scores.detach().cpu().numpy().tolist()
    preds = preds.detach().cpu().numpy().tolist()

    for i in range(B):
        attr_np = attr_np_all[i]
        A = _viz_sign_map(attr_np, viz_sign=viz_sign)
        if viz_sign == "both":
            A01 = 0.5 + 0.5 * (A / (np.abs(A).max() + 1e-8))
            cmap_use = "bwr"
        else:
            A01 = _normalize_01(A)
            cmap_use = ig_cmap

        gray = gray_all[i]
        gray_rgb, heat_rgb, overlay_rgb = _overlay_heatmap(gray, A01, alpha=ig_alpha, cmap=cmap_use)

        png_path = None
        fig = None
        if save_dir is not None or return_figures:
            fig = plt.figure(figsize=(12, 4), dpi=140)
            ax1 = plt.subplot(1, 3, 1); ax1.imshow(gray); ax1.set_title("Input (gray)"); ax1.axis("off")
            ax2 = plt.subplot(1, 3, 2); ax2.imshow(heat_rgb); ax2.set_title("IG Heatmap"); ax2.axis("off")
            ax3 = plt.subplot(1, 3, 3); ax3.imshow(overlay_rgb)
            ax3.set_title(f"Overlay (pred={preds[i]}, pos_score={pos_scores[i]:.3f}, δ={delta_list[i]:.2e})"); ax3.axis("off")
            plt.tight_layout(pad=1.0)

            if save_dir is not None:
                png_path = os.path.join(save_dir, f"{fname_prefix}{fname_index:06d}.png")
                fig.savefig(png_path, bbox_inches="tight")
            if not return_figures:
                plt.close(fig)

        if save_raw_attr and save_dir is not None:
            np.save(os.path.join(save_dir, f"{fname_prefix}attr_{i:06d}.npy"), attr_np)

        results.append({
            "attr": attr_np,
            "heat01": A01 if viz_sign != "both" else None,
            "gray_rgb": gray_rgb,
            "heat_rgb": heat_rgb,
            "overlay_rgb": overlay_rgb,
            "pred": int(preds[i]),
            "pos_score": float(pos_scores[i]),
            "delta": float(delta_list[i]),
            "png_path": png_path,
            "figure": fig if return_figures else None,
        })

    return results


# -----------------------------------------------------------------------------
# Batch maker (helper)
# -----------------------------------------------------------------------------
def make_batch_from_slices(
    slices_2d: List[Union[np.ndarray, torch.Tensor]],
    expected_c: Optional[int] = None,
    device: Union[str, torch.device] = "cuda",
) -> torch.Tensor:
    if len(slices_2d) == 0:
        raise ValueError("slices_2d must be a non-empty list.")
    xs = []
    H0 = W0 = None
    for sl in slices_2d:
        t = torch.from_numpy(sl) if isinstance(sl, np.ndarray) else sl
        if t.ndim == 2:
            t = t.unsqueeze(0)  # (1,H,W)
        if t.ndim != 3:
            raise ValueError(f"each slice must be (H,W) or (C,H,W), got {tuple(t.shape)}")
        if H0 is None:
            H0, W0 = t.shape[1], t.shape[2]
        if t.shape[1] != H0 or t.shape[2] != W0:
            raise ValueError("all slices must share same H,W")
        xs.append(t.float())
    x = torch.stack(xs, dim=0).to(device)  # (B,C,H,W)
    if expected_c is not None:
        x = _match_channels(x, expected_c)
    return x