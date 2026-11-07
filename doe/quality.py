"""Optional quality-first refinement with selection on the actual 8-bit export.

The score is a documented engineering proxy, not an official jury formula.
Every candidate is reconstructed and scored after phase quantization; original
100-step refinement remains available if candidates violate metric guardrails.
"""
from dataclasses import dataclass, replace
import math

import numpy as np
import torch

from .io import phase_to_u8, u8_to_phase
from .losses import per_sample_loss
from .metrics import crosstalk, evaluate


@dataclass(frozen=True)
class QualityConfig:
    steps: int = 300
    lr: float = 0.03
    alpha: float = 1.0
    beta: float = 0.5
    xtalk: float = 0.25
    check_every: int = 50
    peak_final: bool = False

    def __post_init__(self):
        if self.steps < 0 or self.check_every < 1 or self.lr <= 0:
            raise ValueError('steps >= 0, check_every >= 1 and lr > 0 required')


def exported_metrics(phase, target):
    q = phase_to_u8(phase)
    rep = evaluate(q, target)
    xt = crosstalk(target, rep['recon'])
    vals = {k: rep['summary'][k]['mean'] for k in ('pcc', 'ssim_global', 'ssim_skimage', 'ssim_gauss')}
    vals['pcc_worst'] = rep['summary']['pcc']['worst']
    vals['selectivity'] = xt['selectivity']
    return vals


def proxy_score(metrics):
    weighted = [(0.4, metrics['pcc']), (0.3, metrics['ssim_global']), (0.3, metrics['ssim_gauss'])]
    valid = [(w, v) for w, v in weighted if v is not None and np.isfinite(v)]
    return sum(w * v for w, v in valid) / sum(w for w, _ in valid) if valid else -math.inf


def acceptable(candidate, baseline):
    # Four mean metrics cannot regress. Worst-plane correlation and selectivity
    # have small, explicitly disclosed tolerances to avoid brittle selection.
    for key, tolerance in [('pcc', 0), ('ssim_global', 0), ('ssim_skimage', 0),
                           ('ssim_gauss', 0), ('pcc_worst', 0.01), ('selectivity', 0.02)]:
        ref, val = baseline[key], candidate[key]
        if ref is None or not np.isfinite(ref):
            continue
        if val is None or not np.isfinite(val) or val < ref - tolerance:
            return False
    return True


def refine_quality(net, targets, config=None, baseline=None):
    """Return (B,N,N) phases and selection metadata; targets are (B,4,N,N) uint8.

    baseline, if supplied, must be the caller's original 100-step phases. This
    parameter permits caching in experiments without rerunning the baseline.
    """
    cfg = config or QualityConfig()
    t = np.asarray(targets)
    if t.ndim != 4 or t.shape[1:] != (4, 128, 128) or t.dtype != np.uint8 or not len(t):
        raise ValueError('expected nonempty uint8 (B,4,128,128) targets')
    if baseline is None:
        baseline = net.predict_refined_batch(t, steps=100, xtalk=0.25, alpha=1.0, beta=0.5)
    baseline = np.asarray(baseline)
    if baseline.shape != (len(t), 128, 128) or not np.isfinite(baseline).all():
        raise ValueError('baseline must be finite (B,128,128) phases')
    best = np.stack([u8_to_phase(phase_to_u8(p)) for p in baseline])
    refs = [exported_metrics(p, y) for p, y in zip(best, t)]
    chosen = [dict(step=0, score=proxy_score(m), metrics=m) for m in refs]
    x = net._to_tensor(t)
    ph = torch.tensor(baseline, dtype=torch.float32, device=net.device, requires_grad=True)
    loss_cfg = replace(net.loss_cfg, quant=False, alpha=cfg.alpha, beta=cfg.beta, xtalk=cfg.xtalk)
    opt = torch.optim.Adam([ph], lr=cfg.lr)
    for step in range(1, cfg.steps + 1):
        if cfg.peak_final and step > cfg.steps * 0.75:
            loss_cfg = replace(loss_cfg, norm='peak')
        # Anneal the step size; evaluate the wrapped phase with the actual codec.
        opt.param_groups[0]['lr'] = cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step-1) / max(cfg.steps, 1))))
        loss, _ = per_sample_loss(ph, x, net.model.fresnel, loss_cfg)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        opt.step()
        if step % cfg.check_every == 0 or step == cfg.steps:
            proposals = (torch.remainder(ph.detach() + math.pi, 2 * math.pi) - math.pi).cpu().numpy()
            for i, proposal in enumerate(proposals):
                m = exported_metrics(proposal, t[i])
                score = proxy_score(m)
                if acceptable(m, refs[i]) and score > chosen[i]['score']:
                    best[i] = u8_to_phase(phase_to_u8(proposal))
                    chosen[i] = dict(step=step, score=score, metrics=m)
    return best, chosen
