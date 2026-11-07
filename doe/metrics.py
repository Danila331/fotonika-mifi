"""Evaluation metrics (slide 3) and the reference evaluator that works only from the saved phase PNG.

Expert answers (BACKLOG.md, E1/E2): images are compared as AMPLITUDES |K|, and every image (target
and reconstruction) is normalised so that its maximum is 1 before the metrics. That is the default:
quantity="amplitude", policy="peak". Intensity |K|² is kept only for comparison with older results.

Brightness policy of the saved reconstruction PNG (applied to |K| or |K|²):
  peak   - X / max(X) * 255           (default, matches the expert answer)
  minmax - (X - min) / (max - min) * 255
  fit    - least-squares scale to the target, clipped to [0, 255]
Metrics are then computed on both images divided by their own max (range [0, 1], SSIM L = 1).
SSIM is reported three ways because the slide formula is global (no window) while common libraries
are windowed: ssim_global (slide), ssim_skimage (skimage defaults, 7x7), ssim_gauss (Wang 11x11 σ=1.5).
"""

import numpy as np
from skimage.metrics import structural_similarity

from .config import DEPTHS
from .io import u8_to_phase
from .physics import intensity_np

def to_uint8(intensity: np.ndarray, policy: str = "peak", target: np.ndarray | None = None) -> np.ndarray:
    i = np.asarray(intensity, dtype=np.float64)
    if policy == "peak":
        m = i.max()
        out = i / m * 255.0 if m > 0 else np.zeros_like(i)
    elif policy == "minmax":
        lo, hi = i.min(), i.max()
        out = (i - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(i)
    elif policy == "fit":
        if target is None:
            raise ValueError("fit policy needs target")
        t = np.asarray(target, dtype=np.float64)
        d = (i * i).sum()
        out = i * ((i * t).sum() / d) if d > 0 else np.zeros_like(i)
    else:
        raise ValueError(f"unknown policy {policy}")
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def pcc(a, b):
    """Pearson correlation; None if either image is constant (undefined)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else None


def ssim_global(a, b, data_range: float = 255.0) -> float:
    """Exactly the slide formula: global means, variances and covariance, c = (k·L)²."""
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    return float((2 * ma * mb + c1) * (2 * cov + c2) / ((ma**2 + mb**2 + c1) * (va + vb + c2)))


def ssim_skimage(a, b, data_range: float = 255.0) -> float:
    return float(structural_similarity(np.asarray(a, np.float64), np.asarray(b, np.float64), data_range=data_range))


def ssim_gauss(a, b, data_range: float = 255.0) -> float:
    return float(structural_similarity(np.asarray(a, np.float64), np.asarray(b, np.float64), data_range=data_range,
                                       gaussian_weights=True, sigma=1.5, use_sample_covariance=False))


def normalize_max(x) -> np.ndarray:
    """Expert rule E2: divide by the image maximum so it becomes 1 (all-zero image stays zero)."""
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    return x / m if m > 0 else x


def compare(target_u8, recon_u8) -> dict:
    """Both images are normalised to max = 1 (E2), metrics with L = 1."""
    t, r = normalize_max(target_u8), normalize_max(recon_u8)
    return {
        "pcc": pcc(t, r),
        "ssim_global": ssim_global(t, r, 1.0),
        "ssim_skimage": ssim_skimage(t, r, 1.0),
        "ssim_gauss": ssim_gauss(t, r, 1.0),
    }


def reconstruct_u8(phase_u8: np.ndarray, depths=DEPTHS, policy: str = "peak", targets=None,
                   quantity: str = "amplitude") -> np.ndarray:
    """Saved phase PNG (uint8) -> (D,N,N) uint8 reconstructions of |K| (default) or |K|²."""
    inten = intensity_np(u8_to_phase(phase_u8), depths)
    if quantity == "amplitude":
        x = np.sqrt(inten)
    elif quantity == "intensity":
        x = inten
    else:
        raise ValueError(f"unknown quantity {quantity}")
    return np.stack([to_uint8(x[i], policy, None if targets is None else targets[i])
                     for i in range(len(depths))])


def evaluate(phase_u8: np.ndarray, targets_u8: np.ndarray, policy: str = "peak", depths=DEPTHS,
             quantity: str = "amplitude") -> dict:
    """Full report from the 8-bit phase only. targets_u8: (D,N,N) uint8."""
    recon = reconstruct_u8(phase_u8, depths, policy, targets_u8, quantity)
    planes = [dict(depth_mm=round(z * 1e3, 3), **compare(targets_u8[i], recon[i])) for i, z in enumerate(depths)]
    summary = {}
    for k in ("pcc", "ssim_global", "ssim_skimage", "ssim_gauss"):
        vals = [p[k] for p in planes if p[k] is not None]
        summary[k] = {"mean": float(np.mean(vals)) if vals else None,
                      "worst": float(np.min(vals)) if vals else None,
                      "valid": len(vals)}
    return {"quantity": quantity, "policy": policy, "planes": planes, "summary": summary, "recon": recon}


def crosstalk(targets_u8, recon_u8) -> dict:
    """Depth separation. M[i, j] = PCC(target i, reconstruction in plane j), both max-normalised.
    selectivity = mean(diag) - mean(off-diag); own_best = planes whose reconstruction matches its own target
    better than any other target. A DOE that shows the same blend in every plane has selectivity ~0."""
    d = len(targets_u8)
    m = np.full((d, d), np.nan)
    for i in range(d):
        for j in range(d):
            v = pcc(normalize_max(targets_u8[i]), normalize_max(recon_u8[j]))
            m[i, j] = np.nan if v is None else v
    diag = np.nanmean(np.diag(m)) if np.isfinite(np.diag(m)).any() else np.nan
    off = m[~np.eye(d, dtype=bool)]
    offm = np.nanmean(off) if np.isfinite(off).any() else np.nan
    own = int(sum(np.nanargmax(np.nan_to_num(m[:, j], nan=-9)) == j for j in range(d)))
    return {"matrix": m.tolist(), "diag": float(diag), "off": float(offm),
            "selectivity": float(diag - offm), "own_best": own}
