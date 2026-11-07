"""Benchmark stage-0 inference optimizations with GS-50/100 on all demos."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from doe.baselines import gs_batch
from doe.infer import NeuralDOE
from scripts.pareto_docker import metrics, summarize

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("eager", "channels", "torchscript", "compile"), default="eager")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iters", default="0,50,100")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "stage0.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-model", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    demos = sorted(path for path in (ROOT / "data" / "demo").iterdir() if path.is_dir())
    cases = [(path.name, np.stack([np.asarray(Image.open(path / f"target_{i}.png")) for i in range(1, 5)]))
             for path in demos]
    net = NeuralDOE.load(args.checkpoint, device=args.device)
    model = net.model
    if args.variant == "channels":
        model = model.to(memory_format=torch.channels_last)
    if args.variant == "torchscript":
        example = net._to_tensor(cases[0][1][None])
        with torch.no_grad():
            traced = torch.jit.trace(model, example)
            traced = torch.jit.freeze(traced)
            traced = torch.jit.optimize_for_inference(traced)
            expected = model(example)
            actual = traced(example)
        error = float((expected - actual).abs().max())
        if error > 1e-5:
            raise RuntimeError(f"TorchScript output mismatch {error}")
        if args.save_model:
            args.save_model.parent.mkdir(parents=True, exist_ok=True)
            traced.save(str(args.save_model))
        model = traced
    if args.variant == "compile":
        model = torch.compile(model, backend="inductor", fullgraph=True, dynamic=False)

    def predict(targets):
        x = net._to_tensor(targets[None])
        if args.variant == "channels":
            x = x.to(memory_format=torch.channels_last)
        with torch.no_grad():
            return model(x).float().cpu().numpy()[0]

    predict(cases[0][1])
    predict(cases[0][1])
    result = {"variant": args.variant, "threads": args.threads, "device": args.device,
              "cases": len(cases), "modes": {}}
    for iterations in [int(value) for value in args.iters.split(",")]:
        rows = []
        for name, targets in cases:
            start = time.perf_counter()
            initial = predict(targets)
            network_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            phase = initial if iterations == 0 else gs_batch(targets, iters=iterations, init=initial,
                                                              device=args.device)[0]
            gs_ms = (time.perf_counter() - start) * 1000
            rows.append({"case": name, "network_ms": network_ms, "gs_ms": gs_ms,
                         "total_ms": network_ms + gs_ms, **metrics(phase, targets)})
        result["modes"][f"gs{iterations}"] = summarize(rows)
        print(args.variant, args.threads, f"gs{iterations}",
              json.dumps({k: v for k, v in result["modes"][f"gs{iterations}"].items() if k != "rows"}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
