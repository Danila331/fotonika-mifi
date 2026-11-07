"""Exploratory DC-phase pilot on named demos; not a held-out benchmark."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from doe.dc_phase import reconstruction_phase_loss, signal_mask
from doe.infer import NeuralDOE
from doe.io import load_gray, phase_to_u8, save_gray, u8_to_phase
from doe.losses import plane_loss, recon_quantity
from doe.metrics import crosstalk, evaluate
from doe.quality import QualityConfig, acceptable, exported_metrics, proxy_score, refine_quality

ROOT = Path(__file__).resolve().parent.parent


def load_demos(names):
    arrays = []
    for name in names:
        folder = ROOT / "data" / "demo" / name
        arrays.append(np.stack([load_gray(folder / f"target_{i}.png", strict=True) for i in range(1, 5)]))
    return np.stack(arrays)


def dc_refine(net, targets, baseline, steps, weight, check_every=50, xtalk=0.25):
    cfg = dataclasses.replace(net.loss_cfg, quant=False, alpha=1.0, beta=0.5, xtalk=xtalk)
    x = net._to_tensor(targets)
    mask = signal_mask(x, threshold=0.05, dilation=2)
    best = np.stack([u8_to_phase(phase_to_u8(p)) for p in baseline])
    refs = [exported_metrics(p, y) for p, y in zip(best, targets)]
    chosen = [dict(step=0, score=proxy_score(m), metrics=m) for m in refs]
    ph = torch.tensor(best, dtype=torch.float32, device=net.device, requires_grad=True)
    opt = torch.optim.Adam([ph], lr=0.03)
    for step in range(1, steps + 1):
        opt.param_groups[0]["lr"] = 0.03 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / max(steps, 1))))
        fields = net.model.fresnel(ph)
        loss, _ = plane_loss(recon_quantity(fields), x, cfg)
        if weight:
            loss = loss + weight * reconstruction_phase_loss(fields, x, mask)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        opt.step()
        if step % check_every == 0 or step == steps:
            proposals = (torch.remainder(ph.detach() + math.pi, 2 * math.pi) - math.pi).cpu().numpy()
            for i, proposal in enumerate(proposals):
                metrics = exported_metrics(proposal, targets[i])
                score = proxy_score(metrics)
                if acceptable(metrics, refs[i]) and score > chosen[i]["score"]:
                    best[i] = u8_to_phase(phase_to_u8(proposal))
                    chosen[i] = dict(step=step, score=score, metrics=metrics)
    return best, chosen


def background_rmse(phase, target):
    report = evaluate(phase_to_u8(phase), target)
    values = []
    for rec, tgt in zip(report["recon"], target):
        r = rec.astype(np.float64) / max(float(rec.max()), 1.0)
        t = tgt.astype(np.float64) / max(float(tgt.max()), 1.0)
        bg = t <= 0.05
        values.append(float(np.sqrt(np.mean(r[bg] ** 2))) if bg.any() else None)
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def save_result(folder, phase, target):
    folder.mkdir(parents=True, exist_ok=True)
    q = phase_to_u8(phase)
    save_gray(folder / "doe.png", q)
    report = evaluate(q, target)
    for i, rec in enumerate(report["recon"], 1):
        save_gray(folder / f"recon_{i}.png", rec)
    metrics = {k: report["summary"][k]["mean"] for k in ("pcc", "ssim_global", "ssim_skimage", "ssim_gauss")}
    metrics["pcc_worst"] = report["summary"]["pcc"]["worst"]
    metrics["selectivity"] = crosstalk(target, report["recon"])["selectivity"]
    metrics["background_rmse"] = background_rmse(phase, target)
    return metrics


def make_figure(path, name, target, methods, all_metrics):
    base = path.parent
    labels = {"baseline100": "База +100", "quality": "Текущий quality"}
    rows = [("Цели", None)] + [(labels.get(m, m), m) for m in methods]
    fig, axes = plt.subplots(len(rows), 4, figsize=(12, 2.65 * len(rows)))
    for row, (label, method) in enumerate(rows):
        images = target
        if method is not None:
            phase = np.asarray(Image.open(base / method / "doe.png"))
            images = evaluate(phase, target)["recon"]
        for col in range(4):
            ax = axes[row, col]
            ax.imshow(images[col], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(f"z = {[24.1, 25.3, 26.5, 27.7][col]} мм")
        m = all_metrics.get(method)
        extra = "" if m is None else f"\nPCC {m['pcc']:.3f} · SSIMg {m['ssim_gauss']:.3f}\nBG-RMSE {m['background_rmse']:.3f}"
        axes[row, 0].set_ylabel(label + extra, rotation=0, ha="right", va="center", fontsize=10, labelpad=22)
    fig.suptitle(f"{name}: сырые восстановления, единая шкала 0–255", fontsize=14)
    fig.tight_layout(rect=(0.08, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--weights", default="0.003,0.01,0.03")
    parser.add_argument("--demos", default="rocket_smoke,shapes,halftone_photos,photo_text")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    names = args.demos.split(",")
    targets = load_demos(names)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = NeuralDOE.load(device=device)
    baseline = net.predict_refined_batch(targets, steps=100, xtalk=0.25, alpha=1.0, beta=0.5)
    methods = {"baseline100": baseline}
    quality, _ = refine_quality(net, targets, QualityConfig(steps=args.steps), baseline=baseline)
    methods["quality"] = quality
    for weight in [float(v) for v in args.weights.split(",") if v]:
        methods[f"dc_{weight:g}"], _ = dc_refine(net, targets, baseline, args.steps, weight)
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
