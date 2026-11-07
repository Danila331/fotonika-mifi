"""Measure neural-stage + GS Pareto modes on all demos inside Docker or native CPU."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from doe.baselines import gs, gs_batch
from doe.infer import NeuralDOE
from doe.io import phase_to_u8
from doe.metrics import crosstalk, evaluate

ROOT = Path(__file__).resolve().parents[1]


def metrics(phase, targets):
    report = evaluate(phase_to_u8(phase), targets)
    values = {key: report["summary"][key]["mean"] for key in ("pcc", "ssim_global", "ssim_skimage", "ssim_gauss")}
    values["pcc_worst"] = report["summary"]["pcc"]["worst"]
    values["selectivity"] = crosstalk(targets, report["recon"])["selectivity"]
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    demos = sorted(path for path in (ROOT / "data" / "demo").iterdir() if path.is_dir())
    cases = [(path.name, np.stack([np.asarray(Image.open(path / f"target_{i}.png")) for i in range(1, 5)]))
             for path in demos]
    checkpoints = {0: ROOT / "checkpoints" / "stage0.pt", 2: ROOT / "checkpoints" / "stage2.pt",
                   4: ROOT / "checkpoints" / "stage4.pt", 6: ROOT / "checkpoints" / "best.pt"}
    iterations = {0: (0, 5, 20, 50, 100, 200), 2: (0, 5, 20, 50, 100, 200),
                  4: (0, 5, 20, 50, 100, 200), 6: (0, 20, 50)}
    results = {"device": args.device, "cases": len(cases), "protocol": "one warm pass per model, then one timed pass per case; metrics from saved-codec equivalent uint8 phase",
               "modes": {}}
    for stages, checkpoint in checkpoints.items():
        net = NeuralDOE.load(checkpoint, device=args.device)
        net.predict(cases[0][1])  # warm model and device kernels before the measured cases
        for count in iterations[stages]:
            name = f"stage{stages}_gs{count}"
            rows = []
            for case_name, targets in cases:
                start = time.perf_counter()
                initial = net.predict(targets)
                network_ms = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                phase = initial if count == 0 else gs_batch(targets, iters=count, init=initial, device=args.device)[0]
                gs_ms = (time.perf_counter() - start) * 1000
                rows.append({"case": case_name, "network_ms": network_ms, "gs_ms": gs_ms,
                             "total_ms": network_ms + gs_ms, **metrics(phase, targets)})
            results["modes"][name] = summarize(rows)
            print(name, json.dumps({k: v for k, v in results["modes"][name].items() if k != "rows"}, ensure_ascii=False), flush=True)
    rows = []
    gs(cases[0][1], iters=200, device=args.device)
    for case_name, targets in cases:
        start = time.perf_counter()
        phase = gs(targets, iters=200, device=args.device)
        elapsed = (time.perf_counter() - start) * 1000
        rows.append({"case": case_name, "network_ms": 0.0, "gs_ms": elapsed,
                     "total_ms": elapsed, **metrics(phase, targets)})
    results["modes"]["gs200"] = summarize(rows)
    results["gs200_median_ms"] = results["modes"]["gs200"]["median_ms"]
    for mode in results["modes"].values():
        mode["speedup_vs_gs200"] = results["gs200_median_ms"] / mode["median_ms"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("GS200", json.dumps(results["modes"]["gs200"], ensure_ascii=False), flush=True)


def summarize(rows):
    numeric = ("network_ms", "gs_ms", "total_ms", "pcc", "ssim_global", "ssim_skimage",
               "ssim_gauss", "pcc_worst", "selectivity")
    summary = {"rows": rows}
    for key in numeric:
        values = [row[key] for row in rows]
        summary["mean_" + key] = float(np.mean(values))
        summary["median_" + key] = float(np.median(values))
    # Short aliases used by the Pareto report.
    summary["median_ms"] = summary["median_total_ms"]
    for key in ("pcc", "ssim_global", "ssim_skimage", "ssim_gauss", "pcc_worst", "selectivity"):
        summary[key] = summary["mean_" + key]
    return summary


if __name__ == "__main__":
    main()
