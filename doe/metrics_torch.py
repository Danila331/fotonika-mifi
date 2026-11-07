"""Differentiable versions of the metrics for training losses and phase optimisation.

All functions take (..., H, W) tensors and return (...) tensors. Images are expected on a common
scale; data_range sets the SSIM constants (use 1.0 for images in [0, 1]).
"""

import torch
import torch.nn.functional as F


def pcc(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = a.flatten(-2)
    b = b.flatten(-2)
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / torch.sqrt((a * a).sum(-1) * (b * b).sum(-1) + eps)


def ssim_global(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    a = a.flatten(-2)
    b = b.flatten(-2)
    ma, mb = a.mean(-1), b.mean(-1)
    va = ((a - ma[..., None]) ** 2).mean(-1)
    vb = ((b - mb[..., None]) ** 2).mean(-1)
    cov = ((a - ma[..., None]) * (b - mb[..., None])).mean(-1)
    return (2 * ma * mb + c1) * (2 * cov + c2) / ((ma**2 + mb**2 + c1) * (va + vb + c2))


def _gauss_kernel(size: int = 11, sigma: float = 1.5, device=None, dtype=None) -> torch.Tensor:
    x = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim_local(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0, size: int = 11,
               sigma: float = 1.5) -> torch.Tensor:
    """Windowed (Wang) SSIM, valid region only, mean over the map."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    shape = a.shape[:-2]
    a = a.reshape(-1, 1, *a.shape[-2:])
    b = b.reshape(-1, 1, *b.shape[-2:])
    w = _gauss_kernel(size, sigma, a.device, a.dtype)
    mu_a, mu_b = F.conv2d(a, w), F.conv2d(b, w)
    saa = F.conv2d(a * a, w) - mu_a**2
    sbb = F.conv2d(b * b, w) - mu_b**2
    sab = F.conv2d(a * b, w) - mu_a * mu_b
    m = (2 * mu_a * mu_b + c1) * (2 * sab + c2) / ((mu_a**2 + mu_b**2 + c1) * (saa + sbb + c2))
    return m.mean((-3, -2, -1)).reshape(shape)


def normalize_peak(i: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Matches the default 'peak' export policy: I / max(I), per image."""
    return i / (i.flatten(-2).amax(-1)[..., None, None] + eps)


def amplitude(field: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """|K| of a complex field with a finite gradient at zero."""
    return torch.sqrt(field.real**2 + field.imag**2 + eps)


def normalize_max(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Expert rule E2: per image, divide by the maximum so it becomes 1. Same as normalize_peak."""
    return normalize_peak(x, eps)
