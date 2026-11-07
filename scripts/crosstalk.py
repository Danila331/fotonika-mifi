#!/usr/bin/env python
"""Селективность по глубине: КК Пирсона каждой цели с амплитудой восстановления на расстоянии z (22–30 мм)
по сохранённому PNG ДОЭ (официальный протокол: амплитуда |K|, оба изображения нормированы на максимум).

    python scripts/crosstalk.py --run OUTDIR            # synth.py output dir (doe.png + metrics.json)
    python scripts/crosstalk.py --doe doe.png --targets t1.png t2.png t3.png t4.png
Пишет crosstalk.png (кривые; каждая должна иметь максимум в своей плоскости) и crosstalk.json
(кривые + матрица 4×4 «КК(цель i, восстановление в плоскости j)»).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from doe.config import DEPTHS  # noqa: E402
from doe.visualize import METHOD_RU, crosstalk_plot, depth_scan, load_run  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--doe")
    ap.add_argument("--targets", nargs=4)
    ap.add_argument("--zmin", type=float, default=22.0)
    ap.add_argument("--zmax", type=float, default=30.0)
    ap.add_argument("--n", type=int, default=321)
    ap.add_argument("--policy", default="peak", choices=("peak", "minmax"))
    ap.add_argument("--out", default="results/delivery/crosstalk.png")
    a = ap.parse_args(argv)

    phase_u8, targets, label = load_run(a.run, a.doe, a.targets)
    z_mm = np.unique(np.concatenate([np.linspace(a.zmin, a.zmax, a.n), np.array(DEPTHS) * 1e3]))
    _, curves = depth_scan(phase_u8, targets, z_mm * 1e-3, a.policy)
    _, mat = depth_scan(phase_u8, targets, DEPTHS, a.policy)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    crosstalk_plot(z_mm, curves, out, title=f"Селективность по глубине{' · ' + METHOD_RU.get(label, label) if label else ''}")
    peaks = [float(z_mm[int(np.nanargmax(c))]) if np.isfinite(c).any() else None for c in curves]
    res = {"z_mm": z_mm.round(4).tolist(), "pcc": np.round(curves, 5).tolist(), "policy": a.policy,
           "quantity": "amplitude",
           "peak_z_mm": peaks, "design_z_mm": [z * 1e3 for z in DEPTHS],
           "matrix_target_vs_plane": np.round(mat, 5).tolist()}
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    print("КК(цель i, плоскость j):")
    for i, row in enumerate(mat):
        print(f"  цель {i + 1}: " + " ".join(f"{v:6.3f}" for v in row))
    print("максимум КК при z, мм:", peaks, "плоскости:", res["design_z_mm"])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
