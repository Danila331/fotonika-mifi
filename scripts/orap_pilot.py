"""Train only the four shared ORAP masks; exploratory validation, no test-set tuning."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from doe.data import QuadGenerator
from doe.infer import NeuralDOE
from doe.io import save_gray
from doe.losses import per_sample_loss
from doe.orap import make_shared_phase_trainable
from doe.train import official_metrics, predict_batch
from scripts.dc_phase_pilot import load_demos, make_figure, save_result


def proxy(metrics):
    return 0.4 * metrics["pcc_mean"] + 0.3 * metrics["ssim_global_mean"] + 0.3 * metrics["ssim_gauss_mean"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = NeuralDOE.load(device=device)
    model = net.model
    select_u8 = QuadGenerator("val", seed=88232).sample(16)
    confirm_u8 = QuadGenerator("val", seed=88233).sample(32)
    demo_names = ["rocket_smoke", "shapes", "halftone_photos", "photo_text"]
    demos_u8 = load_demos(demo_names)
    baseline_select = official_metrics(predict_batch(model, select_u8, device), select_u8)
    baseline_confirm = official_metrics(predict_batch(model, confirm_u8, device), confirm_u8)
    baseline_demos = model(net._to_tensor(demos_u8)).detach().cpu().numpy()
    baseline_refined = net.predict_refined_batch(confirm_u8, steps=100, xtalk=0.25, alpha=1.0, beta=0.5)
    baseline_refined_metrics = official_metrics(baseline_refined, confirm_u8)
    phase = make_shared_phase_trainable(model)
    opt = torch.optim.Adam([phase], lr=args.lr)
    generator = QuadGenerator("train", seed=88231)
    best = {"score": proxy(baseline_select), "phase": phase.detach().cpu().clone(),
            "step": 0, "selectivity": baseline_select["selectivity"]}
    history = {"baseline_select": baseline_select, "baseline_confirm": baseline_confirm,
               "baseline_refined_confirm": baseline_refined_metrics, "validation": []}
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / max(args.steps, 1))))
        opt.param_groups[0]["lr"] = lr
        targets = torch.tensor(generator.sample(args.batch), dtype=torch.float32, device=device) / 255.0
        phases = model(targets)
        loss, _ = per_sample_loss(phases, targets, model.fresnel, net.loss_cfg)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        opt.step()
        if step % args.val_every == 0 or step == args.steps:
            metrics = official_metrics(predict_batch(model, select_u8, device), select_u8)
            metrics["step"] = step
            metrics["proxy"] = proxy(metrics)
            history["validation"].append(metrics)
            if metrics["proxy"] > best["score"] and metrics["selectivity"] >= baseline_select["selectivity"] - 0.03:
                best = {"score": metrics["proxy"], "phase": phase.detach().cpu().clone(),
                        "step": step, "selectivity": metrics["selectivity"]}
            print(json.dumps({"step": step, "loss": float(loss.mean().detach()),
                              "proxy": metrics["proxy"], "pcc": metrics["pcc_mean"],
                              "ssim_gauss": metrics["ssim_gauss_mean"],
                              "selectivity": metrics["selectivity"]}), flush=True)
    phase.data.copy_(best["phase"].to(device))
    model.eval()
    checkpoint = {"state_dict": model.state_dict(), "model_cfg": model.cfg,
                  "loss_cfg": dataclasses.asdict(net.loss_cfg),
                  "orap": {"steps": args.steps, "batch": args.batch, "lr": args.lr,
                           "seed": 88231, "best_step": best["step"], "train_only": "rand_phase"}}
    torch.save(checkpoint, args.out / "best.pt")
    confirm = official_metrics(predict_batch(model, confirm_u8, device), confirm_u8)
    refined = net.predict_refined_batch(confirm_u8, steps=100, xtalk=0.25, alpha=1.0, beta=0.5)
    refined_metrics = official_metrics(refined, confirm_u8)
    orap_demos = model(net._to_tensor(demos_u8)).detach().cpu().numpy()
    orap_demo_refined = net.predict_refined_batch(demos_u8, steps=100, xtalk=0.25, alpha=1.0, beta=0.5)
    methods = {"before": baseline_demos, "orap": orap_demos, "orap_refined100": orap_demo_refined}
    demo_metrics = {}
    for method, phases in methods.items():
        demo_metrics[method] = {}
        for i, name in enumerate(demo_names):
            demo_metrics[method][name] = save_result(args.out / "demos" / name / method, phases[i], demos_u8[i])
    for i, name in enumerate(demo_names):
        for plane in range(4):
            save_gray(args.out / "demos" / name / f"target_{plane+1}.png", demos_u8[i, plane])
        metrics = {m: demo_metrics[m][name] for m in methods}
        make_figure(args.out / "demos" / name / "comparison.png", name, demos_u8[i], list(methods), metrics)
    summary = {"protocol": "ORAP-only exploratory training; confirm is new val seed, not external test",
               "best_step": best["step"], "elapsed_seconds": time.perf_counter() - start,
               "select_baseline": baseline_select, "confirm_baseline": baseline_confirm,
               "confirm_orap": confirm, "confirm_baseline_refined100": baseline_refined_metrics,
               "confirm_orap_refined100": refined_metrics, "demos": demo_metrics}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (args.out / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2))
    print("FINAL", json.dumps({"baseline": baseline_confirm, "orap": confirm,
                                "baseline_refined": baseline_refined_metrics,
                                "orap_refined": refined_metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
