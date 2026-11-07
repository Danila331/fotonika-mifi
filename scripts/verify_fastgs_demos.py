"""Verify the production fastgs path on every demo and render representative cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import synth

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    engine = synth.Synthesiser("nn_fastgs", steps=args.steps, device="cpu")
    demos = sorted(path for path in (ROOT / "data" / "demo").iterdir() if path.is_dir())
    rows = []
    for demo in demos:
        inputs = [str(demo / f"target_{i}.png") for i in range(1, 5)]
        result = synth.run(inputs, args.out / demo.name, synthesiser=engine, figure=False, quiet=True)
        row = {"case": demo.name, **{key: result["summary"][key]["mean"] for key in synth.METRICS},
               "selectivity": result["crosstalk"]["selectivity"],
               "synth_ms": result["timings_ms"]["synth_ms"], "total_ms": result["timings_ms"]["total_ms"]}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = {"method": "nn_fastgs", "steps": args.steps, "cases": len(rows), "load_ms": engine.load_ms,
               "rows": rows}
    for key in (*synth.METRICS, "selectivity", "synth_ms", "total_ms"):
        summary["mean_" + key] = float(np.mean([row[key] for row in rows]))
        summary["median_" + key] = float(np.median([row[key] for row in rows]))
    ordered = sorted(rows, key=lambda row: row["ssim_global"])
    summary["representatives"] = {"worst": ordered[0]["case"], "median": ordered[len(ordered)//2]["case"],
                                  "best": ordered[-1]["case"]}
    if not args.skip_figures:
        for label, case in summary["representatives"].items():
            demo = ROOT / "data" / "demo" / case
            inputs = [str(demo / f"target_{i}.png") for i in range(1, 5)]
            synth.run(inputs, args.out / f"{label}_{case}", synthesiser=engine, figure=True, quiet=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
