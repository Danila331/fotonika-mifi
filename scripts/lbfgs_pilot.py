"""Compare full L-BFGS teacher optimization against Adam quality mode."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from doe.image_losses import excess_tv_loss, multiscale_l1_loss
from doe.infer import NeuralDOE
from doe.io import phase_to_u8, save_gray, u8_to_phase
from doe.losses import plane_loss, recon_quantity
from doe.quality import QualityConfig, acceptable, exported_metrics, proxy_score, refine_quality
from scripts.dc_phase_pilot import load_demos, make_figure, save_result


def lbfgs_refine(net, targets, baseline, outer_steps, max_iter, perceptual=False):
    cfg = dataclasses.replace(net.loss_cfg, quant=False, alpha=1.0, beta=0.5, xtalk=0.25)
    x = net._to_tensor(targets)
    best = np.stack([u8_to_phase(phase_to_u8(p)) for p in baseline])
    refs = [exported_metrics(p, y) for p, y in zip(best, targets)]
    chosen = [dict(step=0, score=proxy_score(m), metrics=m) for m in refs]
    ph = torch.tensor(best, dtype=torch.float32, device=net.device, requires_grad=True)
    opt = torch.optim.LBFGS([ph], lr=0.1, history_size=10, max_iter=max_iter,
                            tolerance_grad=1e-7, tolerance_change=1e-9,
                            line_search_fn="strong_wolfe")
    for outer in range(1, outer_steps + 1):
        def closure():
            opt.zero_grad(set_to_none=True)
            fields = net.model.fresnel(ph)
            loss, metrics = plane_loss(recon_quantity(fields), x, cfg)
            if perceptual:
                loss = loss + 0.3 * multiscale_l1_loss(metrics["rec"], metrics["tgt"])
                loss = loss + 0.1 * excess_tv_loss(metrics["rec"], metrics["tgt"])
            loss.sum().backward()
            return loss.sum()
        opt.step(closure)
        proposals = (torch.remainder(ph.detach() + np.pi, 2 * np.pi) - np.pi).cpu().numpy()
        for i, proposal in enumerate(proposals):
            metrics = exported_metrics(proposal, targets[i])
            score = proxy_score(metrics)
            if acceptable(metrics, refs[i]) and score > chosen[i]["score"]:
                best[i] = u8_to_phase(phase_to_u8(proposal))
                chosen[i] = dict(step=outer * max_iter, score=score, metrics=metrics)
    return best, chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=15)
    parser.add_argument("--demos", default="rocket_smoke,shapes,halftone_photos,photo_text")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    names = args.demos.split(",")
    targets = load_demos(names)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = NeuralDOE.load(device=device)
    baseline = net.predict_refined_batch(targets, steps=100, xtalk=0.25, alpha=1.0, beta=0.5)
    quality, _ = refine_quality(net, targets, QualityConfig(steps=args.outer * args.max_iter), baseline=baseline)
    lbfgs, _ = lbfgs_refine(net, targets, baseline, args.outer, args.max_iter, perceptual=False)
    perceptual, _ = lbfgs_refine(net, targets, baseline, args.outer, args.max_iter, perceptual=True)
    methods = {"baseline100": baseline, "quality": quality, "lbfgs": lbfgs, "lbfgs_perceptual": perceptual}
    summary = {"protocol": "exploratory demos, not held-out test", "outer": args.outer,
               "max_iter": args.max_iter, "demos": names, "methods": {}}
    for method, phases in methods.items():
        summary["methods"][method] = {}
        for i, name in enumerate(names):
            summary["methods"][method][name] = save_result(args.out / name / method, phases[i], targets[i])
        keys = ("pcc", "ssim_global", "ssim_skimage", "ssim_gauss", "pcc_worst", "selectivity", "background_rmse")
        summary["methods"][method]["mean"] = {
            key: float(np.nanmean([summary["methods"][method][name][key] for name in names])) for key in keys}
    for i, name in enumerate(names):
        for plane in range(4):
            save_gray(args.out / name / f"target_{plane+1}.png", targets[i, plane])
        metrics = {m: summary["methods"][m][name] for m in methods}
        make_figure(args.out / name / "comparison.png", name, targets[i], list(methods), metrics)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({m: v["mean"] for m, v in summary["methods"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
