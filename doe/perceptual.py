"""Visual-quality refinement with multiscale amplitude and excess-TV losses."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

from .image_losses import excess_tv_loss, multiscale_l1_loss
from .io import phase_to_u8, u8_to_phase
from .losses import plane_loss, recon_quantity
from .quality import acceptable, exported_metrics, proxy_score


def refine_perceptual(net, targets, baseline, steps=300, ms_weight=0.3, tv_weight=0.1,
                      check_every=50):
    """Refine quantized-selected phases from a supplied baseline.

    Guardrails are identical to `refine_quality`; image losses affect candidates,
    never the independent evaluator or the saved reconstruction PNGs.
    """
    targets = np.asarray(targets)
    baseline = np.asarray(baseline)
    if targets.ndim != 4 or targets.shape[1:] != (4, 128, 128) or targets.dtype != np.uint8:
        raise ValueError("targets must be uint8 (B,4,128,128)")
    if baseline.shape != (len(targets), 128, 128) or not np.isfinite(baseline).all():
        raise ValueError("baseline must be finite (B,128,128)")
    if steps < 0 or check_every < 1 or ms_weight < 0 or tv_weight < 0:
        raise ValueError("steps >= 0, check_every >= 1 and nonnegative loss weights required")
    cfg = dataclasses.replace(net.loss_cfg, quant=False, alpha=1.0, beta=0.5, xtalk=0.25)
    x = net._to_tensor(targets)
    best = np.stack([u8_to_phase(phase_to_u8(p)) for p in baseline])
    refs = [exported_metrics(p, y) for p, y in zip(best, targets)]
    chosen = [dict(step=0, score=proxy_score(m), metrics=m) for m in refs]
    ph = torch.tensor(best, dtype=torch.float32, device=net.device, requires_grad=True)
    opt = torch.optim.Adam([ph], lr=0.03)
    for step in range(1, steps + 1):
        opt.param_groups[0]["lr"] = 0.03 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / max(steps, 1))))
        fields = net.model.fresnel(ph)
        loss, metrics = plane_loss(recon_quantity(fields), x, cfg)
        if ms_weight:
            loss = loss + ms_weight * multiscale_l1_loss(metrics["rec"], metrics["tgt"])
        if tv_weight:
            loss = loss + tv_weight * excess_tv_loss(metrics["rec"], metrics["tgt"])
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        opt.step()
        if step % check_every == 0 or step == steps:
            proposals = (torch.remainder(ph.detach() + math.pi, 2 * math.pi) - math.pi).cpu().numpy()
            for i, proposal in enumerate(proposals):
                candidate = exported_metrics(proposal, targets[i])
                score = proxy_score(candidate)
                if acceptable(candidate, refs[i]) and score > chosen[i]["score"]:
                    best[i] = u8_to_phase(phase_to_u8(proposal))
                    chosen[i] = dict(step=step, score=score, metrics=candidate)
    return best, chosen
