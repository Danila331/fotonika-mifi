#!/usr/bin/env python
"""Таблица метрик по демо-наборам: нейросеть (1 проход) и нейросеть + уточнение, из metrics.json synth.py.

    python scripts/demo_table.py [--refine 100]  ->  results/delivery/demo/summary.md
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def f(v):
    return "н/д" if v is None else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine", type=int, default=50)
    ap.add_argument("--dir", default="results/delivery/demo")
    a = ap.parse_args()
    d = ROOT / a.dir
    modes = [("nn", "сеть"), (f"nn_refine{a.refine}", f"сеть + {a.refine}")]
    lines = ["Демо-наборы (`data/demo/`), протокол экспертов: амплитуда |K|, нормировка на максимум. "
             "КК — корреляция Пирсона, ИСС — формула слайда, ИССg — гауссово окно 11×11; "
             "«худш.» — худшая из 4 плоскостей; селективность = среднее диагонали − среднее вне диагонали матрицы "
             "КК(цель i, плоскость j), «своя лучшая» — в скольких плоскостях восстановление больше всего похоже на свою цель. Время — синтез одного ДОЭ на CPU; для «сеть» — повторный вызов, для «сеть + N» — первый (холодный) "
             "вызов в отдельном процессе, поэтому оно больше медианы из сравнения методов.\n",
             "| Набор | режим | КК ср. | КК худш. | ИСС ср. | ИСС худш. | ИССg ср. | селективность | своя лучшая | время, мс |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for q in sorted(p for p in d.iterdir() if p.is_dir()):
        for sub, ru in modes:
            mj = q / sub / "metrics.json"
            if not mj.exists():
                continue
            m = json.loads(mj.read_text())
            s, t = m["summary"], m["timings_ms"]
            ms = t.get("synth_warm_ms", t["synth_ms"])
            lines.append(f"| {q.name} | {ru} | {f(s['pcc']['mean'])} | {f(s['pcc']['worst'])} | "
                         f"{f(s['ssim_global']['mean'])} | {f(s['ssim_global']['worst'])} | "
                         f"{f(s['ssim_gauss']['mean'])} | {f(m['crosstalk']['selectivity'])} | "
                         f"{m['crosstalk']['own_best']}/4 | {ms:.0f} |")
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
