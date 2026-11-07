"""Measure learned-stage truncation and bf16 inference without retraining."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from doe.baselines import gs
from doe.data import QuadGenerator
from doe.distill import truncate_stages
from doe.infer import NeuralDOE
from doe.model import count_params
from doe.train import official_metrics, predict_batch
from scripts.dc_phase_pilot import load_demos


def warm_ms(net, targets, repeats=20):
    for _ in range(3):
        net.predict(targets)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        net.predict(targets)
        samples.append((time.perf_counter() - start) * 1000)
    return float(np.median(samples))


def gs_warm_ms(target, repeats=8):
    for _ in range(2):
        gs(target, iters=200, device="mps")
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        gs(target, iters=200, device="mps")
        samples.append((time.perf_counter() - start) * 1000)
    return float(np.median(samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = "mps"
    holdout = QuadGenerator("val", seed=88411).sample(32)
    demo_names = sorted(path.name for path in (Path(__file__).resolve().parent.parent / "data" / "demo").iterdir()
                        if path.is_dir())
    demos = load_demos(demo_names)
    gs_times = [gs_warm_ms(target) for target in demos]
    results = {"protocol": "same 32 validation quadruples and all 12 demos; warm MPS median",
               "gs200_demo_ms": gs_times, "gs200_median_ms": float(np.median(gs_times)), "configs": []}
    for stages in (0, 2, 4, 6):
        for bf16 in (False, True):
            name = f"stages_{stages}_bf16_{int(bf16)}"
            try:
                net = NeuralDOE.load(device=device)
                truncate_stages(net.model, stages)
                net.model.cnn_autocast = bf16
                metrics = official_metrics(predict_batch(net.model, holdout, device), holdout)
                times = [warm_ms(net, target) for target in demos]
                record = {"name": name, "stages": stages, "bf16": bf16,
                          "parameters": count_params(net.model), "metrics": metrics,
                          "demo_ms": times, "median_ms": float(np.median(times)),
                          "speedup_vs_gs200": float(np.median(gs_times) / np.median(times))}
            except Exception as error:
                record = {"name": name, "stages": stages, "bf16": bf16, "error": str(error)}
            results["configs"].append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    (args.out / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
