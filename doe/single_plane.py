"""Per-depth kinoforms followed by complex superposition (RV/FZP construction)."""
import math

import numpy as np
import torch

from .physics import Fresnel


@torch.no_grad()
def single_plane_gs_components(targets_u8, iters=20, device="cpu", seed=0):
    """Optimize an independent phase-only kinoform for every target plane.

    Each iteration performs one forward and one inverse Fresnel transform per
    plane, with amplitude replacement only against that plane's own target.
    """
    targets = np.asarray(targets_u8)
    if targets.ndim != 4 or targets.shape[1] != 4:
        raise ValueError("targets must have shape (B,4,H,W)")
    if iters < 0:
        raise ValueError("iters must be nonnegative")
    prop = Fresnel().to(device)
    amp = torch.as_tensor(targets.astype(np.float32) / 255.0, device=device)
    amp = amp / amp.flatten(-2).amax(-1)[..., None, None].clamp_min(1e-12)
    amp = amp / amp.pow(2).sum((-2, -1), keepdim=True).sqrt().clamp_min(1e-12)
    generator = torch.Generator().manual_seed(seed)
    random_phase = (torch.rand(amp.shape, generator=generator) * 2 * math.pi - math.pi).to(device)
    field = torch.polar(amp, random_phase)
    for _ in range(iters):
        phase = torch.angle(prop.backward_field(field))
        forward = prop.fftc(torch.polar(torch.ones_like(phase), phase) * prop.chirp)
        energy_field = forward.real.pow(2).add(forward.imag.pow(2)).sum((-2, -1), keepdim=True)
        energy_target = amp.pow(2).sum((-2, -1), keepdim=True).clamp_min(1e-12)
        constrained_amp = amp * torch.sqrt(energy_field / energy_target)
        field = torch.polar(constrained_amp, torch.angle(forward))
    return torch.angle(prop.backward_field(field)).cpu().double().numpy()


def single_plane_superposition(components):
    """Sum unit-modulus per-depth kinoforms and retain one phase-only DOE."""
    components = np.asarray(components)
    if components.ndim != 4:
        raise ValueError("components must have shape (B,D,H,W)")
    return np.angle(np.exp(1j * components).sum(axis=1))
