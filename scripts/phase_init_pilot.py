"""Compare random-phase-free and limited-random-phase initializations on demos."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from doe.baselines import backprop_init, smooth_init
from doe.infer import NeuralDOE
from doe.quality import QualityConfig, refine_quality
from scripts.dc_phase_pilot import dc_refine, load_demos, make_figure, save_result


def limited_random_init(targets, range_pi, device, seed=0):
    x = torch.as_tensor(targets.astype(np.float32) / 255.0, device=device)
    x = x / x.flatten(-2).amax(-1)[..., None, None].clamp_min(1e-12)
    x = x / x.pow(2).sum((-2, -1), keepdim=True).sqrt().clamp_min(1e-12)
    g = torch.Generator().manual_seed(seed)
    phase = (torch.rand(x.shape, generator=g) * 2 - 1).to(device) * (range_pi * math.pi)
    field = torch.polar(x, phase)
    return torch.angle(net.model.fresnel.backward_field(field).sum(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--demos", default="rocket_smoke,shapes,halftone_photos,photo_text")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    names = args.demos.split(",")
    targets = load_demos(names)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    global net
    net = NeuralDOE.load(device=device)
    neural = net.predict_refined_batch(targets, steps=100, xtalk=0.25, alpha=1.0, beta=0.5)
    methods = {"neural100": neural}
    methods["neural_quality"], _ = refine_quality(net, targets, QualityConfig(steps=args.steps), baseline=neural)
    initializers = {
        "random_full": backprop_init(targets, device=device).cpu().numpy(),
        "limited_0.5": limited_random_init(targets, 0.5, device).cpu().numpy(),
        "limited_0.25": limited_random_init(targets, 0.25, device).cpu().numpy(),
        "smooth_0.01": smooth_init(targets, alpha=0.01, device=device).cpu().numpy(),
        "smooth_0.1": smooth_init(targets, alpha=0.1, device=device).cpu().numpy(),
        "smooth_1": smooth_init(targets, alpha=1.0, device=device).cpu().numpy(),
    }
    for name, initial in initializers.items():
        methods[name], _ = dc_refine(net, targets, initial, args.steps, weight=0)
    summary = {"protocol": "exploratory demos, not held-out test", "steps": args.steps,
               "demos": names, "methods": {}}
    for method, phases in methods.items():
        summary["methods"][method] = {}
        for i, name in enumerate(names):
            summary["methods"][method][name] = save_result(args.out / name / method, phases[i], targets[i])
        keys = ("pcc", "ssim_global", "ssim_skimage", "ssim_gauss", "pcc_worst", "selectivity", "background_rmse")
        summary["methods"][method]["mean"] = {
            key: float(np.nanmean([summary["methods"][method][name][key] for name in names])) for key in keys}
    for i, name in enumerate(names):
        for plane in range(4):
            from doe.io import save_gray
            save_gray(args.out / name / f"target_{plane+1}.png", targets[i, plane])
        metrics = {m: summary["methods"][m][name] for m in methods}
        make_figure(args.out / name / "comparison.png", name, targets[i], list(methods), metrics)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({m: v["mean"] for m, v in summary["methods"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
