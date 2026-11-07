"""Reference plain-GS quality/latency at the same iteration counts as the hybrid."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from doe.baselines import gs
from scripts.pareto_docker import metrics, summarize

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iters", default="20,50,75,100,125,150,200")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cases = []
    for path in sorted(p for p in (ROOT / "data" / "demo").iterdir() if p.is_dir()):
        targets = np.stack([np.asarray(Image.open(path / f"target_{i}.png")) for i in range(1, 5)])
        cases.append((path.name, targets))
    gs(cases[0][1], iters=200, device=args.device)
    result = {"device": args.device, "threads": args.threads, "modes": {}}
    for iterations in [int(v) for v in args.iters.split(",")]:
        rows = []
        for name, targets in cases:
            start = time.perf_counter()
            phase = gs(targets, iters=iterations, device=args.device)
            elapsed = (time.perf_counter() - start) * 1000
            rows.append({"case": name, "network_ms": 0.0, "gs_ms": elapsed,
                         "total_ms": elapsed, **metrics(phase, targets)})
        result["modes"][f"gs{iterations}"] = summarize(rows)
        print(f"gs{iterations}", json.dumps({k: v for k, v in result["modes"][f"gs{iterations}"].items() if k != "rows"}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
