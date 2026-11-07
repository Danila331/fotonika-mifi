#!/usr/bin/env python
"""GIF «проход фокуса»: один сохранённый ДОЭ, z от 22 до 30 мм; 4 изображения по очереди становятся резкими.
В каждом кадре — амплитуда |K| на расстоянии z, нормированная на максимум кадра.

    python scripts/focus_sweep.py --run OUTDIR [--frames 80] [--out results/delivery/focus_sweep.gif]
    python scripts/focus_sweep.py --doe doe.png --targets t1.png t2.png t3.png t4.png
Кадры в расчётных плоскостях всегда есть и показываются дольше. Справа — КК каждой цели с текущим
восстановлением как функция z, курсор на текущем z.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from doe.config import DEPTHS  # noqa: E402
from doe.visualize import depth_scan, focus_frame, load_run  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--doe")
    ap.add_argument("--targets", nargs=4)
    ap.add_argument("--zmin", type=float, default=22.0)
    ap.add_argument("--zmax", type=float, default=30.0)
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--ms", type=int, default=90, help="frame duration")
    ap.add_argument("--hold", type=int, default=900, help="duration of design-plane frames")
    ap.add_argument("--out", default="results/delivery/focus_sweep.gif")
    a = ap.parse_args(argv)

    phase_u8, targets, _ = load_run(a.run, a.doe, a.targets)
    design = np.array(DEPTHS) * 1e3
    z_mm = np.unique(np.concatenate([np.linspace(a.zmin, a.zmax, a.frames), design]))
    recon, _ = depth_scan(phase_u8, targets, z_mm * 1e-3)
    cz = np.linspace(a.zmin, a.zmax, 241)
    _, curves = depth_scan(phase_u8, targets, cz * 1e-3)

    frames, durations = [], []
    for z, r in zip(z_mm, recon):
        rgb = focus_frame(r, float(z), cz, curves)
        frames.append(Image.fromarray(rgb).quantize(colors=128, method=Image.Quantize.MEDIANCUT))
        durations.append(a.hold if np.min(np.abs(design - z)) < 1e-9 else a.ms)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    # still image of the 4 in-focus frames, handy for slides
    idx = [int(np.argmin(np.abs(z_mm - d))) for d in design]
    Image.fromarray(np.concatenate([np.pad(recon[i], 4, constant_values=255) for i in idx], axis=1)).save(
        out.with_name(out.stem + "_planes.png"))
    print(f"{len(frames)} кадров, z {a.zmin}–{a.zmax} мм -> {out} ({out.stat().st_size / 1e6:.1f} МБ)")


if __name__ == "__main__":
    main()
