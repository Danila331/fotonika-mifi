"""Train a compact student with phasor, physics, deep supervision and QAT."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from doe.baselines import gs_batch
from doe.data import QuadGenerator
from doe.distill import build_student, circular_phasor_loss, truncate_stages, validate_teacher_arrays
from doe.infer import NeuralDOE
from doe.losses import per_sample_loss
from doe.model import count_params
from doe.train import official_metrics, predict_batch


def proxy(metrics):
    return 0.4 * metrics["pcc_mean"] + 0.3 * metrics["ssim_global_mean"] + 0.3 * metrics["ssim_gauss_mean"]


def inference_checkpoint(model, loss_cfg, metadata):
    return {"state_dict": model.state_dict(), "model_cfg": model.cfg,
            "loss_cfg": dataclasses.asdict(loss_cfg), "student": metadata}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--phasor-weight", type=float, default=0.5)
    parser.add_argument("--xtalk", type=float, default=0.25)
    parser.add_argument("--init-stages", type=int, default=None)
    parser.add_argument("--reference-gs", type=int, default=None,
                        help="require better proxy than plain GS with this iteration count")
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=88410)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    targets = np.load(args.teacher_data / "targets.npy")
    teacher_phases = np.load(args.teacher_data / "phases.npy")
    validate_teacher_arrays(targets, teacher_phases)
    teacher = NeuralDOE.load(device=device)
    student = build_student().to(device)
    val_u8 = QuadGenerator("val", seed=args.seed + 1).sample(32)
    baseline_val = official_metrics(predict_batch(teacher.model, val_u8, device), val_u8)
    reference_val = None
    if args.reference_gs is not None:
        reference_val = official_metrics(gs_batch(val_u8, iters=args.reference_gs, device=device), val_u8)
    if args.init_stages is not None:
        student = NeuralDOE.load(device=device).model
        truncate_stages(student, args.init_stages)
    parameters = count_params(student)
    if args.init_stages is None and not 1_000_000 <= parameters <= 3_000_000:
        raise ValueError(f"unexpected student parameter count {parameters}")
    selectivity_floor = (reference_val or baseline_val)["selectivity"] - 0.02
    base_cfg = dataclasses.replace(teacher.loss_cfg, alpha=1.0, beta=0.5, xtalk=args.xtalk)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    best = {"score": proxy(reference_val) if reference_val is not None else -math.inf,
            "step": 0, "state": None, "metrics": None}
    history = {"parameters": parameters, "baseline_val": baseline_val, "reference_val": reference_val,
               "train": [], "val": []}
    order = rng.permutation(len(targets))
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        student.train()
        pos = ((step - 1) * args.batch) % len(targets)
        if pos == 0:
            order = rng.permutation(len(targets))
        indices = order[pos:pos + args.batch]
        x = torch.tensor(targets[indices], dtype=torch.float32, device=device) / 255.0
        teacher_phase = torch.tensor(teacher_phases[indices], dtype=torch.float32, device=device)
        cfg = dataclasses.replace(base_cfg, quant=step > args.steps / 2)
        phases = student(x, return_all=True)
        physics, _ = per_sample_loss(phases[-1], x, student.fresnel, cfg)
        phasor = circular_phasor_loss(phases[-1], teacher_phase)
        deep_terms = [per_sample_loss(phase, x, student.fresnel, cfg)[0].mean() for phase in phases[:-1]]
        deep = torch.stack(deep_terms).mean() if deep_terms else torch.zeros((), device=device)
        loss = (physics.mean() + args.phasor_weight * phasor.mean() + 0.2 * deep).mean()
        lr = args.lr * min(1.0, step / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / args.steps)))
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if step % 20 == 0:
            history["train"].append({"step": step, "loss": float(loss.detach()),
                                     "physics": float(physics.mean().detach()),
                                     "phasor": float(phasor.mean().detach()),
                                     "deep": float(deep.detach()), "lr": lr})
        if step % args.val_every == 0 or step == args.steps:
            metrics = official_metrics(predict_batch(student, val_u8, device), val_u8)
            metrics.update(step=step, proxy=proxy(metrics), elapsed_seconds=time.perf_counter() - started)
            history["val"].append(metrics)
            if metrics["proxy"] > best["score"] and metrics["selectivity"] >= selectivity_floor:
                best = {"score": metrics["proxy"], "step": step,
                        "state": {key: value.detach().cpu().clone() for key, value in student.state_dict().items()},
                        "metrics": metrics}
                torch.save(inference_checkpoint(student, base_cfg,
                                                {"parameters": parameters, "best_step": step,
                                                 "steps": args.steps, "seed": args.seed,
                                                 "phasor_weight": args.phasor_weight}), args.out / "best.pt")
            print(json.dumps({"step": step, "loss": history["train"][-1]["loss"],
                              "proxy": metrics["proxy"], "pcc": metrics["pcc_mean"],
                              "ssim_gauss": metrics["ssim_gauss_mean"],
                              "selectivity": metrics["selectivity"], "best_step": best["step"]}), flush=True)
            (args.out / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2))
    if best["state"] is None:
        raise RuntimeError("no checkpoint passed the validation proxy and selectivity gates")
    student.load_state_dict(best["state"])
    metadata = {"parameters": parameters, "best_step": best["step"], "steps": args.steps,
                "seed": args.seed, "phasor_weight": args.phasor_weight,
                "elapsed_seconds": time.perf_counter() - started}
    torch.save(inference_checkpoint(student, base_cfg, metadata), args.out / "best.pt")
    summary = {"parameters": parameters, "best_step": best["step"], "best_metrics": best["metrics"],
               "baseline_val": baseline_val, "reference_val": reference_val, "selectivity_floor": selectivity_floor,
               "elapsed_seconds": metadata["elapsed_seconds"], "qat_start": args.steps // 2 + 1,
               "phasor_weight": args.phasor_weight, "deep_weight": 0.2}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("STUDENT_READY", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
