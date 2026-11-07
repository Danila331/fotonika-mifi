#!/usr/bin/env python
"""Селективность по глубине для всех демо-наборов, для сети и для сети + N шагов уточнения.

Матрица 4×4 КК(цель i, амплитуда в плоскости j) берётся из metrics.json synth.py (doe.metrics.crosstalk по
сохранённым recon PNG); selectivity = mean(диагональ) − mean(вне диагонали).

    python scripts/selectivity.py [--refine 50] [--top 2]
Пишет results/delivery/selectivity.md и selectivity.json; последняя строка вывода — лучшие наборы по
основному режиму (сеть + N), через пробел, для scripts/make_delivery.sh.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/delivery/demo")
    ap.add_argument("--refine", type=int, default=50)
    ap.add_argument("--top", type=int, default=2)
    a = ap.parse_args(argv)
    modes = [("nn", "сеть"), (f"nn_refine{a.refine}", f"сеть + {a.refine}")]
    rows = []
    for q in sorted(p for p in (ROOT / a.dir).iterdir() if p.is_dir()):
        r = {"quad": q.name}
        for sub, _ in modes:
            mj = q / sub / "metrics.json"
            if mj.exists():
                r[sub] = json.loads(mj.read_text())["crosstalk"]
        mm = q / modes[0][0] / "metrics.json"
        if mm.exists():
            from doe.io import load_gray
            t = np.stack([load_gray(p) for p in json.loads(mm.read_text())["inputs"]]).reshape(4, -1).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                c = np.corrcoef(t)[~np.eye(4, dtype=bool)]
            r["target_similarity"] = float(np.nanmean(c))
        rows.append(r)
    main_mode = modes[1][0]
    rows.sort(key=lambda r: -r.get(main_mode, r.get("nn", {})).get("selectivity", -9))
    out = ROOT / "results/delivery"
    (out / "selectivity.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))

    def cell(r, sub):
        x = r.get(sub)
        return ("— | —" if x is None else f"{x['selectivity']:.3f} | {x['own_best']}/4")

    lines = ["# Селективность по глубине на демо-наборах", "",
             "ДОЭ — `checkpoints/best.pt`; для каждого набора строится матрица 4×4: КК исходного изображения i с "
             "восстановленной амплитудой в плоскости j (протокол экспертов: |K|, нормировка на максимум). "
             "**Селективность** = среднее по диагонали − среднее вне диагонали: насколько восстановление в своей "
             "плоскости похоже на «своё» изображение сильнее, чем на соседние. ДОЭ, который во всех плоскостях "
             "рисует одну и ту же смесь, даёт ≈ 0. «Своя лучшая» — в скольких из 4 плоскостей восстановление больше "
             "всего похоже именно на свою цель. «Сходство целей» — средний КК между самими исходными изображениями: "
             "у похожих целей (четыре ракеты, четыре цифры одного шрифта) КК вне диагонали высокий при любом ДОЭ.",
             "", f"| Набор | сеть: селективность | сеть: своя лучшая | сеть + {a.refine}: селективность | "
                 f"сеть + {a.refine}: своя лучшая | сходство целей |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        ts = r.get("target_similarity")
        lines.append(f"| {r['quad']} | {cell(r, 'nn')} | {cell(r, main_mode)} | "
                     f"{'—' if ts is None else f'{ts:.3f}'} |")
    for sub, ru in modes:
        v = [r[sub]["selectivity"] for r in rows if sub in r]
        o = [r[sub]["own_best"] for r in rows if sub in r]
        if v:
            lines.append(f"\n{ru}: средняя селективность {np.nanmean(v):.3f}, своя лучшая в среднем {np.mean(o):.2f}/4.")
    (out / "selectivity.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(" ".join(r["quad"] for r in rows[: a.top]))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
