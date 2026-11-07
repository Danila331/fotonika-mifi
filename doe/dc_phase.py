"""Experimental double-constraint reconstruction-phase regularization.

This is a research pilot for masked phase regularization in the reconstruction
plane, not a claim of theoretical optimality or an official jury metric.
"""
import torch
import torch.nn.functional as F


def signal_mask(target01: torch.Tensor, threshold: float = 0.05, dilation: int = 2) -> torch.Tensor:
    """Binary signal support from per-image max-normalized targets (B,D,H,W)."""
    if target01.ndim != 4:
        raise ValueError("target01 must have shape (B,D,H,W)")
    peak = target01.flatten(-2).amax(-1)[..., None, None]
    normalized = target01 / peak.clamp_min(1e-12)
    mask = (normalized > threshold).to(target01.dtype)
    if dilation > 0:
        b, d, h, w = mask.shape
        mask = F.max_pool2d(mask.reshape(b * d, 1, h, w), 2 * dilation + 1,
                            stride=1, padding=dilation).reshape(b, d, h, w)
    return mask


def reconstruction_phase_loss(field: torch.Tensor, target01: torch.Tensor,
                              mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean |Laplacian(mask * angle(field))| per sample, over valid 3x3 regions."""
    if field.ndim != 4 or field.shape != target01.shape:
        raise ValueError("field and target01 must have matching (B,D,H,W) shapes")
    if mask is None:
        mask = signal_mask(target01)
    if mask.shape != field.shape:
        raise ValueError("mask must have the same shape as field")
    b, d, h, w = field.shape
    phase = torch.angle(field)
    x = (phase * mask).reshape(b * d, 1, h, w)
    kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                          dtype=x.dtype, device=x.device).reshape(1, 1, 3, 3)
    laplacian = F.conv2d(x, kernel).abs().mean((-2, -1)).reshape(b, d, 1)
    return laplacian.mean(1)
