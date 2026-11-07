"""Training / refinement losses mirroring the jury metrics (ML agent).

Expert rules (E1/E2): the jury compares AMPLITUDES |K| (not intensities), and every image — target and
reconstruction — is divided by its own maximum before PCC / SSIM (range [0, 1], SSIM L = 1). We mirror that:
  rec = |K| / max|K|  (or a soft max, see `norm`),   tgt = img / max(img)   (all-zero target stays zero)
per-plane loss  l_d = w_pcc (1 - PCC) + alpha (1 - SSIM_local) + beta (1 - SSIM_global)
total           L   = mean_d l_d + gamma * max_d l_d + xtalk * C      (extra weight on the worst plane)
C — crosstalk penalty: how much plane j still contains the OTHER targets i != j, measured against the part of
target i that is not explained by target j (so similar or identical targets are not penalised for being similar).
Without it the network converges to the same blend of all four images in every plane (good per-plane PCC,
almost no depth separation).
PCC is undefined for a constant target (the evaluator skips it), so that term is masked out there.

`quantize_ste` implements straight-through 8-bit phase quantisation for quantisation-aware training.
"""

import math
from dataclasses import dataclass

import torch

from .metrics_torch import amplitude, normalize_peak, pcc, ssim_global, ssim_local

TWO_PI = 2 * math.pi


def quantize_phase(phase: torch.Tensor) -> torch.Tensor:
    """Exactly doe.io.phase_to_u8 followed by u8_to_phase (for phases already in [-π, π])."""
    q = torch.round((phase + math.pi) / TWO_PI * 255.0).clamp(0, 255)
    return q / 255.0 * TWO_PI - math.pi


def quantize_ste(phase: torch.Tensor) -> torch.Tensor:
    """Forward: 8-bit quantised phase; backward: identity."""
    return phase + (quantize_phase(phase) - phase).detach()


@dataclass
class LossConfig:
    alpha: float = 0.2      # windowed (Gaussian 11x11) SSIM weight
    beta: float = 0.1       # global (slide formula) SSIM weight
    gamma: float = 0.3      # worst-plane weight
    quant: bool = False     # straight-through 8-bit quantisation of the phase
    norm: str = "soft"      # "soft": X/softmax-weighted peak (smoother gradients); "peak": X/max(X) exactly
    soft_beta: float = 50.0
    w_pcc: float = 1.0
    quantity: str = "amplitude"  # "amplitude" (expert rule E1) or legacy "intensity"
    xtalk: float = 0.0      # crosstalk penalty weight (depth separation)


def soft_peak_normalize(x: torch.Tensor, beta: float = 50.0) -> torch.Tensor:
    """X / smooth-max(X): softmax(beta*X/max)-weighted mean of X (as doe.baselines.soft_peak), so the
    brightness gradient is spread over all hot pixels instead of the single brightest one."""
    flat = x.flatten(-2)
    w = torch.softmax(beta * flat / (flat.detach().amax(-1, keepdim=True) + 1e-12), -1)
    return x / ((w * flat).sum(-1)[..., None, None] + 1e-12)


def normalize_target(target01: torch.Tensor) -> torch.Tensor:
    """img / max(img) per image (E2); an all-zero image stays zero."""
    m = target01.flatten(-2).amax(-1)[..., None, None]
    return torch.where(m > 0, target01 / m.clamp_min(1e-12), target01)


def recon_quantity(field: torch.Tensor, quantity: str = "amplitude") -> torch.Tensor:
    if quantity == "amplitude":
        return amplitude(field)
    if quantity == "intensity":
        return field.real**2 + field.imag**2
    raise ValueError(quantity)


def plane_metrics(x: torch.Tensor, target01: torch.Tensor, norm: str = "soft", soft_beta: float = 50.0) -> dict:
    """x: (B,D,N,N) raw |K| (or |K|^2); target01: (B,D,N,N) in [0,1]. Returns per-plane metric tensors."""
    rec = normalize_peak(x) if norm == "peak" else soft_peak_normalize(x, soft_beta)
    t = normalize_target(target01)
    valid = t.flatten(-2).std(-1) > 1e-6
    return {
        "rec": rec,
        "tgt": t,
        "pcc": pcc(rec, t),
        "pcc_valid": valid,
        "ssim_local": ssim_local(rec, t),
        "ssim_global": ssim_global(rec, t),
    }


def crosstalk_penalty(rec: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(B,D,N,N) normalised recon / targets -> (B,) mean over i != j of relu(PCC(rec_j, t_i ⊥ t_j)).

    t_i ⊥ t_j is the centred target i with its projection on centred target j removed: the part of image i
    that should NOT appear in plane j. Pairs where that part is ~0 (identical / proportional targets) are
    masked out. Everything is computed from Gram matrices, without materialising the D×D residual images.
    """
    r = rec.flatten(-2)
    t = tgt.flatten(-2)
    r = r - r.mean(-1, keepdim=True)
    t = t - t.mean(-1, keepdim=True)
    g = t @ t.transpose(-1, -2)                    # (B,D,D)  G[i,j] = <t_i, t_j>
    c = r @ t.transpose(-1, -2)                    # (B,D,D)  C[j,i] = <r_j, t_i>
    rr = (r * r).sum(-1)                           # (B,D)    |r_j|^2
    gd = torch.diagonal(g, dim1=-2, dim2=-1)       # (B,D)    |t_j|^2
    coef = g / (gd[..., None, :] + eps)            # coef[i,j] = G[i,j] / G[j,j]
    ct = c.transpose(-1, -2)                       # ct[i,j] = <r_j, t_i>
    cjj = torch.diagonal(c, dim1=-2, dim2=-1)      # <r_j, t_j>
    num = ct - coef * cjj[..., None, :]            # <r_j, t_i ⊥ t_j>
    perp2 = gd[..., :, None] - g**2 / (gd[..., None, :] + eps)   # |t_i ⊥ t_j|^2
    corr = num / torch.sqrt(rr[..., None, :] * perp2.clamp_min(0) + eps)
    d = rec.shape[-3]
    off = ~torch.eye(d, dtype=torch.bool, device=rec.device)
    ok = off & (perp2 > 1e-3 * gd[..., :, None]) & (gd[..., None, :] > eps) & (gd[..., :, None] > eps)
    pen = torch.where(ok, torch.relu(corr), torch.zeros_like(corr))
    return pen.sum((-2, -1)) / ok.sum((-2, -1)).clamp_min(1)


def plane_loss(x: torch.Tensor, target01: torch.Tensor, cfg: LossConfig) -> tuple[torch.Tensor, dict]:
    m = plane_metrics(x, target01, cfg.norm, cfg.soft_beta)
    lpcc = torch.where(m["pcc_valid"], 1 - m["pcc"], torch.zeros_like(m["pcc"]))
    lp = cfg.w_pcc * lpcc + cfg.alpha * (1 - m["ssim_local"]) + cfg.beta * (1 - m["ssim_global"])  # (B,D)
    per_sample = lp.mean(-1) + cfg.gamma * lp.amax(-1)
    if cfg.xtalk:
        xt = crosstalk_penalty(m["rec"], m["tgt"])
        m["xtalk"] = xt
        per_sample = per_sample + cfg.xtalk * xt
    return per_sample, m


def per_sample_loss(phase, target01, fresnel, cfg: LossConfig | None = None):
    """Per-sample loss (B,) and per-plane metrics; used by batched refinement (samples are independent)."""
    cfg = cfg or LossConfig()
    if cfg.quant:
        phase = quantize_ste(phase)
    return plane_loss(recon_quantity(fresnel(phase), cfg.quantity), target01, cfg)


def jury_loss(phase: torch.Tensor, target01: torch.Tensor, fresnel, cfg: LossConfig | None = None):
    """phase (B,N,N) -> scalar loss (mean over batch) and detached per-plane metrics."""
    per_sample, m = per_sample_loss(phase, target01, fresnel, cfg)
    return per_sample.mean(), {k: v.detach() for k, v in m.items() if k not in ("rec", "tgt")}
