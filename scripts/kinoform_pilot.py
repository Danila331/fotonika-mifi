"""Pilot per-plane kinoform optimization and complex superposition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from doe.infer import NeuralDOE
from doe.io import save_gray
from doe.quality import QualityConfig, refine_quality
from doe.single_plane import single_plane_gs_components, single_plane_superposition
from scripts.dc_phase_pilot import dc_refine, load_demos, make_figure, save_result


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
    net = NeuralDOE.load(device=device)
    neural = net.predict_refined_batch(targets, steps=100, xtalk=0.25, alpha=1.0, beta=0.5)
    quality, _ = refine_quality(net, targets, QualityConfig(steps=args.steps), baseline=neural)
    methods = {"neural_quality": quality}
    initials = {}
    for iterations in (0, 5, 20, 50):
        components = single_plane_gs_components(targets, iters=iterations, device=device)
        initial = single_plane_superposition(components)
        initials[iterations] = initial
        methods[f"kino_{iterations}"] = initial
    for iterations in (5, 20, 50):
        methods[f"kino_{iterations}_refined"], _ = dc_refine(
            net, targets, initials[iterations], args.steps, weight=0, xtalk=0.25)
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
            save_gray(args.out / name / f"target_{plane+1}.png", targets[i, plane])
        metrics = {m: summary["methods"][m][name] for m in methods}
        make_figure(args.out / name / "comparison.png", name, targets[i], list(methods), metrics)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({m: v["mean"] for m, v in summary["methods"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
