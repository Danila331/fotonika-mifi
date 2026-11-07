"""Classic iterative baselines: multi-plane Gerchberg-Saxton and direct Adam phase optimisation.

Everything is batched internally (B,4,N,N) and runs through doe.physics.Fresnel, so the same code works
on CPU and MPS/CUDA. Public single-hologram API (CONTRACTS.md):

  gs(targets_u8 (4,N,N), iters, init="backprop"|"random", seed=0)          -> phase (N,N) float64
  adam(targets_u8 (4,N,N), steps, init_phase=None, lr=..., device="cpu")    -> phase (N,N) float64
Batched:
  gs_batch(targets_u8 (B,4,N,N), ...)   adam_batch(targets_u8 (B,4,N,N), ...) -> (B,N,N) float64
Reusable training loss for the ML agent:
  jury_loss(intensity (B,4,N,N), targets01 (B,4,N,N)) -> scalar

Expert rules (BACKLOG.md E1/E2): the jury compares AMPLITUDES |K| with the target image, both normalised
to max = 1. So by default (QUANTITY = "amplitude") the target pixel value IS the desired amplitude: GS
imposes |K| ∝ img, and the losses compare sqrt(I)/max with img/max(img). quantity="intensity" reproduces
the old |K|² ∝ img convention for comparison.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
import torch

from .config import DEPTHS
from .metrics_torch import normalize_peak, pcc as tpcc, ssim_global as tssim_global
from .physics import Fresnel

QUANTITY = "amplitude"  # what the target image prescribes: |K| ("amplitude", expert rule E1) or |K|²
_PROP_CACHE: dict = {}


def _prop(device) -> Fresnel:
    key = str(device)
    if key not in _PROP_CACHE:
        _PROP_CACHE[key] = Fresnel(DEPTHS).to(device)
    return _PROP_CACHE[key]


def _as_batch(targets_u8) -> tuple[np.ndarray, bool]:
    t = np.asarray(targets_u8)
    if t.ndim == 3:
        return t[None], True
    if t.ndim != 4:
        raise ValueError(f"targets must be (4,N,N) or (B,4,N,N), got {t.shape}")
    return t, False


def _targets01(targets_u8, device) -> torch.Tensor:
    t, _ = _as_batch(targets_u8)
    return torch.as_tensor(t.astype(np.float32) / 255.0, device=device)


def _target_amp(t01: torch.Tensor, quantity: str | None = None) -> torch.Tensor:
    """Desired |K| (up to scale) for targets in [0,1]."""
    q = quantity or QUANTITY
    if q == "amplitude":
        return t01
    if q == "intensity":
        return torch.sqrt(t01)
    raise ValueError(f"unknown quantity {q}")


# ----------------------------------------------------------------------------------------------- inits

def backprop_init(targets_u8, seed: int = 0, device="cpu", quantity: str | None = None) -> torch.Tensor:
    """Superposition of back-propagated target amplitudes with random image-plane phase -> (B,N,N) phase."""
    t01 = _targets01(targets_u8, device)
    prop = _prop(device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    rnd = (torch.rand(t01.shape, generator=g) * 2 * math.pi - math.pi).to(device)
    amp = _target_amp(t01, quantity)
    amp = amp / (amp.pow(2).sum((-2, -1), keepdim=True).sqrt() + 1e-12)  # equal energy per plane
    field = torch.polar(amp, rnd)
    doe = prop.backward_field(field).sum(1)
    return torch.angle(doe)


def smooth_init(targets_u8, alpha: float = 1.0, device="cpu", quantity: str | None = None) -> torch.Tensor:
    """Back-propagated superposition with a smooth quadratic image-plane phase (low-speckle holography):
    image phase π·alpha·r²/N (pixel units) spreads every target over the DOE aperture without speckle."""
    t01 = _targets01(targets_u8, device)
    n = t01.shape[-1]
    x = torch.arange(n, device=device, dtype=torch.float32) - n // 2
    quad = math.pi * alpha * (x[:, None] ** 2 + x[None, :] ** 2) / n
    amp = _target_amp(t01, quantity)
    amp = amp / (amp.pow(2).sum((-2, -1), keepdim=True).sqrt() + 1e-12)
    doe = _prop(device).backward_field(torch.polar(amp, quad.expand_as(amp))).sum(1)
    return torch.angle(doe)


def random_init(targets_u8, seed: int = 0, device="cpu") -> torch.Tensor:
    t, _ = _as_batch(targets_u8)
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.rand((t.shape[0], t.shape[-2], t.shape[-1]), generator=g) * 2 * math.pi - math.pi).to(device)


def _init(targets_u8, init, seed, device) -> torch.Tensor:
    if isinstance(init, str):
        if init == "backprop":
            return backprop_init(targets_u8, seed, device)
        if init == "random":
            return random_init(targets_u8, seed, device)
        if init.startswith("smooth"):
            return smooth_init(targets_u8, float(init[6:] or 1.0), device)
        if init == "gs":
            return torch.as_tensor(gs_batch(targets_u8, 50, init="backprop", seed=seed), dtype=torch.float32,
                                   device=device)
        raise ValueError(f"unknown init {init}")
    p = torch.as_tensor(np.asarray(init), dtype=torch.float32, device=device)
    return p[None] if p.ndim == 2 else p


# --------------------------------------------------------------------------------------- Gerchberg-Saxton

def _amp_constraint(field: torch.Tensor, amp_t: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Replace |field| by the target amplitude scaled to the field energy, keep the phase."""
    e_field = field.real.pow(2).add(field.imag.pow(2)).sum((-2, -1), keepdim=True)
    e_t = amp_t.pow(2).sum((-2, -1), keepdim=True) + 1e-12
    a = amp_t * torch.sqrt(e_field / e_t)
    if weights is not None:
        a = a * weights
    return torch.polar(a, torch.angle(field))


@torch.no_grad()
def gs_batch(targets_u8, iters: int, init="backprop", seed: int = 0, variant: str = "global",
             device="cpu", record_at=None, weighted: bool = False, timings: dict | None = None,
             quantity: str | None = None) -> np.ndarray | tuple:
    """Multi-plane Gerchberg-Saxton on a batch. Returns phases (B,N,N) float64.

    variant="global":     propagate to all planes, impose amplitudes, back-propagate, average DOE fields,
                          keep the phase (one iteration = 4 forward + 4 backward FFTs).
    variant="sequential": cycle the planes one by one (one iteration = one full cycle over 4 planes).
    weighted=True:        adaptive per-plane weights (weighted GS): planes whose PCC lags get a larger
                          share in the DOE average (global variant only).
    record_at:            iterable of iteration counts; returns (final, {it: phases}) instead.
    timings:              optional dict filled with {it: wall seconds since start} at recorded iterations.
    quantity:             "amplitude" (default, |K| ∝ img) or "intensity" (|K|² ∝ img).
    """
    t_start = time.perf_counter()
    tb, _ = _as_batch(targets_u8)
    prop = _prop(device)
    amp_t = _target_amp(_targets01(tb, device), quantity)
    phase = _init(tb, init, seed, device) if init != "backprop" else backprop_init(tb, seed, device, quantity)
    rec = sorted(set(int(r) for r in record_at)) if record_at else []
    snaps = {}
    if 0 in rec:
        snaps[0] = phase.cpu().double().numpy()
    chirp = prop.chirp
    D = amp_t.shape[1]
    w = torch.ones(amp_t.shape[:2] + (1, 1), device=device)
    for it in range(1, iters + 1):
        if variant == "global":
            f = prop(phase)
            if weighted:
                q = tpcc(f.abs(), amp_t).clamp(min=1e-3)[..., None, None]
                w = w * (q.mean(1, keepdim=True) / q) ** 0.5
                w = w / w.mean(1, keepdim=True)
            doe = prop.backward_field(_amp_constraint(f, amp_t, w if weighted else None)).sum(1)
            phase = torch.angle(doe)
        elif variant == "sequential":
            for d in range(D):
                h = torch.polar(torch.ones_like(phase), phase)
                f = prop.fftc(h * chirp[d])
                g = _amp_constraint(f, amp_t[:, d])
                phase = torch.angle(prop.ifftc(g) * torch.conj(chirp[d]))
        else:
            raise ValueError(f"unknown GS variant {variant}")
        if it in rec:
            if timings is not None:
                timings[it] = time.perf_counter() - t_start
            snaps[it] = phase.cpu().double().numpy()
    out = phase.cpu().double().numpy()
    return (out, snaps) if record_at else out


GS_DEFAULT_VARIANT = "global"  # set from results/baselines (see summary.md)


def gs(targets_u8, iters: int = 200, init: str = "backprop", seed: int = 0, variant: str | None = None,
       device="cpu", weighted: bool = False, quantity: str | None = None) -> np.ndarray:
    """Single hologram multi-plane GS. targets_u8 (4,N,N) -> phase (N,N) float64."""
    return gs_batch(targets_u8, iters, init, seed, variant or GS_DEFAULT_VARIANT, device,
                    weighted=weighted, quantity=quantity)[0]


# ------------------------------------------------------------------------------------------------ losses

def ssim_local_fft(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0, size: int = 11,
                   sigma: float = 1.5) -> torch.Tensor:
    """Same as metrics_torch.ssim_local (Gaussian 11x11, valid region, mean) but ~6x faster on CPU:
    the blur is a circular FFT convolution, which equals the linear one on the valid region."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    h, w = a.shape[-2:]
    r = size // 2
    key = (h, w, size, sigma, a.dtype, str(a.device))
    k = _KCACHE.get(key)
    if k is None:
        g = torch.arange(size, dtype=a.dtype) - r
        g = torch.exp(-(g**2) / (2 * sigma**2))
        g = g / g.sum()
        kk = torch.zeros(h, w, dtype=a.dtype)
        kk[:size, :size] = g[:, None] * g[None, :]
        k = _KCACHE[key] = torch.fft.rfft2(torch.roll(kk, (-r, -r), (0, 1))).to(a.device)
    st = torch.stack([a, b, a * a, b * b, a * b], 0)
    st = torch.fft.irfft2(torch.fft.rfft2(st) * k, s=(h, w))[..., r:h - r, r:w - r]
    ma, mb, saa, sbb, sab = st.unbind(0)
    saa = saa - ma**2
    sbb = sbb - mb**2
    sab = sab - ma * mb
    m = (2 * ma * mb + c1) * (2 * sab + c2) / ((ma**2 + mb**2 + c1) * (saa + sbb + c2))
    return m.mean((-2, -1))


_KCACHE: dict = {}


def soft_peak(i: torch.Tensor, beta: float = 50.0) -> torch.Tensor:
    """Smooth max: softmax(beta·I/max)-weighted mean of I, so the gradient is spread over all hot pixels."""
    flat = i.flatten(-2)
    w = torch.softmax(beta * flat / (flat.detach().amax(-1, keepdim=True) + 1e-12), -1)
    return (w * flat).sum(-1)[..., None, None] + 1e-12


def _worst_weighted(per_plane: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    """Softmax-weighted mean over planes (dim 1): pushes the worst plane up."""
    w = torch.softmax(per_plane.detach() / tau, dim=1)
    return (w * per_plane).sum(1)


def make_loss(w_pcc: float = 1.0, w_ssim_g: float = 0.0, w_ssim_l: float = 0.0, w_mse: float = 0.0,
              norm: str = "peak", worst: float = 0.0, tau: float = 0.05, beta: float = 50.0,
              quantity: str = "amplitude") -> Callable:
    """Composite loss: (intensity (B,4,N,N), targets01 (B,4,N,N) = img/255) -> scalar.

    quantity="amplitude" (jury rule): X = sqrt(I), T = t / max(t); "intensity" (old): X = I, T = t.
    per plane: w_pcc(1-PCC(X,T)) + w_ssim_g(1-SSIM_global(X̂,T)) + w_ssim_l(1-SSIM_gauss11(X̂,T)) + w_mse·MSE(X̂,T)
    X̂ = X normalised like the exporter: norm="peak" (X/max, exact), "soft" (X/softmax-peak, gradient spread
    over all hot pixels). worst∈[0,1] blends the plane mean with a soft worst-plane (softmax, temperature tau).
    """
    def loss(i, t):
        if quantity == "amplitude":
            x = torch.sqrt(i + 1e-8)
            t = t / (t.flatten(-2).amax(-1)[..., None, None] + 1e-6)
        else:
            x = i
        if norm == "peak":
            xp = normalize_peak(x)
        elif norm == "soft":
            xp = x / soft_peak(x, beta)
        else:
            raise ValueError(norm)
        per = torch.zeros(i.shape[:2], device=i.device, dtype=i.dtype)
        if w_pcc:
            per = per + w_pcc * (1 - tpcc(x, t))
        if w_ssim_g:
            per = per + w_ssim_g * (1 - tssim_global(xp, t))
        if w_ssim_l:
            per = per + w_ssim_l * (1 - ssim_local_fft(xp, t))
        if w_mse:
            per = per + w_mse * (xp - t).pow(2).mean((-2, -1))
        out = per.mean(1)
        if worst:
            out = (1 - worst) * out + worst * _worst_weighted(per, tau)
        return out.mean()
    return loss


def loss_amp_mse(i, t):
    """(a) amplitude MSE with the optimal per-plane scale, relative to the field energy (|K| ∝ img)."""
    a = torch.sqrt(i + 1e-8)
    s = (a * t).sum((-2, -1), keepdim=True) / (t.pow(2).sum((-2, -1), keepdim=True) + 1e-12)
    return ((a - s * t).pow(2).sum((-2, -1)) / (a.pow(2).sum((-2, -1)) + 1e-12)).mean()


LOSSES: dict[str, Callable] = {
    # amplitude (jury rule E1/E2)
    "amp_mse": loss_amp_mse,                                          # (a) scale-optimal amplitude MSE
    "pcc": make_loss(1, 0, 0),                                        # (b) 1-PCC on |K|
    "pcc_ssimg_peak": make_loss(1, 1, 0),                             # (c) + slide SSIM on |K|/max
    "pcc_ssimgl_peak": make_loss(1, 1, 1),                            # (c) + windowed SSIM
    "pcc_ssimgl_soft": make_loss(1, 1, 1, norm="soft"),               # (d) smooth max
    "mse_peak": make_loss(0, 0, 0, w_mse=10),                         # (d) MSE(|K|/max, img/max)
    "jury": make_loss(2, 0.5, 1, norm="soft", beta=20),               # (d) re-weighted: best J (winner)
    "jury_worst": make_loss(2, 0.5, 1, norm="soft", beta=20, worst=0.3),  # (d) + worst-plane weighting
    # old intensity-based winner (|K|² ∝ img), kept for the intensity-vs-amplitude comparison
    "jury_int": make_loss(2, 0.5, 1, norm="soft", quantity="intensity"),
}

# Winner of the amplitude loss study (results/baselines/summary.md, v2 val[:12], 300 Adam steps), best J:
#   2·(1-PCC(X,T)) + 0.5·(1-SSIM_global(X̂,T)) + 1·(1-SSIM_gauss11(X̂,T)),
#   X = sqrt(I), T = img/max(img), X̂ = X / soft-max(X) (softmax-weighted peak, beta=20).
# "jury_worst" trades ~0.01 J for a much better worst plane (PCC worst +0.09).
JURY_LOSS_NAME = "jury"


def jury_loss(intensity: torch.Tensor, targets01: torch.Tensor) -> torch.Tensor:
    """Best loss for the jury metrics: amplitude |K| = sqrt(intensity) vs img, both max-normalised.
    intensity (B,4,N,N) = |K|², targets01 (B,4,N,N) = img/255 (normalised to max 1 internally) -> scalar."""
    return LOSSES[JURY_LOSS_NAME](intensity, targets01)


# -------------------------------------------------------------------------------------------------- Adam

def adam_batch(targets_u8, steps: int, init_phase=None, lr: float = 0.15, device="cpu",
               loss: str | Callable | None = None, init: str = "backprop", seed: int = 0, record_at=None,
               cosine: bool = True, quant_aware_last: int = 0, timings: dict | None = None
               ) -> np.ndarray | tuple:
    """Direct phase optimisation by Adam through the differentiable Fresnel model.

    targets_u8 (B,4,N,N) or (4,N,N); init_phase (B,N,N)/(N,N) radians overrides `init`.
    quant_aware_last: optimise the last K steps through 8-bit phase quantisation (straight-through).
    Returns phases (B,N,N) float64 wrapped to [-π, π] (or (final, {it: phases}) if record_at).
    With record_at and cosine=True the schedule spans `steps`, so snapshot k is "k steps of a
    `steps`-long run", not an independent k-step run. timings: see gs_batch.
    """
    t_start = time.perf_counter()
    tb, _ = _as_batch(targets_u8)
    prop = _prop(device)
    t01 = _targets01(tb, device)
    loss_fn = LOSSES[loss or JURY_LOSS_NAME] if not callable(loss) else loss
    if init_phase is None and init == "backprop":
        p0 = backprop_init(tb, seed, device)
    else:
        p0 = _init(tb, init if init_phase is None else init_phase, seed, device)
    phase = torch.nn.Parameter(p0.clone().float())
    opt = torch.optim.Adam([phase], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps, 1), eta_min=lr * 0.05) if cosine else None
    rec = sorted(set(int(r) for r in record_at)) if record_at else []
    snaps = {}
    wrap = lambda p: torch.remainder(p + math.pi, 2 * math.pi) - math.pi
    if 0 in rec:
        snaps[0] = wrap(phase.detach()).cpu().double().numpy()
    step_q = 2 * math.pi / 255
    for it in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        p = phase
        if quant_aware_last and it > steps - quant_aware_last:
            pw = wrap(p)
            p = pw + (torch.round((pw + math.pi) / step_q) * step_q - math.pi - pw).detach()
        inten = prop.intensity(p)
        # sum over the batch so every hologram gets its own (scale-free) gradient regardless of B
        l = loss_fn(inten, t01) * tb.shape[0]
        l.backward()
        opt.step()
        if sched:
            sched.step()
        if it in rec:
            if timings is not None:
                timings[it] = time.perf_counter() - t_start
            snaps[it] = wrap(phase.detach()).cpu().double().numpy()
    out = wrap(phase.detach()).cpu().double().numpy()
    return (out, snaps) if record_at else out


def adam(targets_u8, steps: int = 500, init_phase=None, lr: float = 0.15, device="cpu", loss=None,
         init: str = "backprop", seed: int = 0) -> np.ndarray:
    """Single hologram Adam. targets_u8 (4,N,N) -> phase (N,N) float64."""
    return adam_batch(targets_u8, steps, init_phase, lr, device, loss, init, seed)[0]
