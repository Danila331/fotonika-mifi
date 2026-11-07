#!/usr/bin/env python
"""Method comparison on the same quads: targets / backprop / GS / Adam / NN / NN+refine.

    python scripts/compare_methods.py [--quads demo|val:K|test:K] [--methods gs adam nn nn_refine]
                                      [--device cpu] [--out results/delivery/methods]
Every number goes through synth.run (saved 8-bit doe.png -> doe.metrics.evaluate). Writes per quad
OUT/<quad>/<method>/ (synth outputs) and OUT/<quad>/methods.png; across quads OUT/summary.json,
OUT/summary.md and results/delivery/speed_quality.png. Methods whose code is not there yet are skipped.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from doe.io import load_gray, save_gray  # noqa: E402
from synth import Synthesiser, run  # noqa: E402

LABELS = {"backprop": "Обратное распр. (1 шаг)", "gs": "Гершберг–Сакстон", "adam": "Adam", "nn": "Нейросеть (1 проход)",
          "nn_refine": "Нейросеть + уточнение"}
METRICS = ("pcc", "ssim_global", "ssim_skimage", "ssim_gauss")


def get_quads(spec, out):
    """-> list of (name, [4 png paths])"""
    if spec == "demo":
        dirs = sorted(p for p in (ROOT / "data/demo").glob("*") if (p / "target_1.png").exists())
        if dirs:
            return [(d.name, [d / f"target_{i}.png" for i in range(1, 5)]) for d in dirs]
        print("демо-наборов ещё нет, беру test:3")
        spec = "test:3"
    split, k = spec.split(":")
    npy = ROOT / f"data/{split}.npy"
    if npy.exists():
        arr = np.load(npy)[: int(k)]
    else:
        from doe.data import QuadGenerator
        arr = QuadGenerator(split, seed=123).sample(int(k))
    quads = []
    for j, q in enumerate(arr):
        name = f"{split}_{j:03d}"
        paths = [out / name / f"target_{i + 1}.png" for i in range(4)]
        for p, t in zip(paths, q):
            save_gray(p, np.asarray(t, np.uint8))
        quads.append((name, paths))
    return quads


def label(m, steps):
    return LABELS[m] + (f" ({steps} шагов)" if m in ("gs", "adam", "nn_refine") else "")


def _agg(fn, rs, k, field):
    """Aggregate over quads, skipping undefined values (PCC of a constant image is None)."""
    v = [r["summary"][k][field] for r in rs if r["summary"][k][field] is not None]
    return float(fn(v)) if v else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quads", default="demo")
    ap.add_argument("--methods", nargs="+", default=["backprop", "gs", "adam", "nn", "nn_refine"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--checkpoint", help="NN checkpoint (default: checkpoints/best.pt)")
    ap.add_argument("--policy", default="peak")
    ap.add_argument("--out", default="results/delivery/methods")
    ap.add_argument("--speed-plot", default="results/delivery/speed_quality.png")
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--gs-steps", type=int, default=500)
    ap.add_argument("--adam-steps", type=int, default=1000, help="1000 = quality ceiling of the baselines study")
    ap.add_argument("--refine-steps", type=int, default=50)
    a = ap.parse_args(argv)
    steps = {"gs": a.gs_steps, "adam": a.adam_steps, "nn_refine": a.refine_steps}
    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)

    synths = {}
    for m in a.methods:
        try:
            synths[m] = Synthesiser(m, steps=steps.get(m), device=a.device, checkpoint=a.checkpoint)
        except (SystemExit, Exception) as e:   # method not delivered yet
            print(f"пропускаю {m}: {e}")
    quads = get_quads(a.quads, out)

    from doe.visualize import method_grid, speed_quality_plot

    per = {m: [] for m in synths}
    for name, paths in quads:
        targets = np.stack([load_gray(p) for p in paths])
        rows = []
        for m, syn in synths.items():
            d = out / name / m
            res = run(paths, d, policy=a.policy, device=a.device, synthesiser=syn, quiet=True, figure=False)
            per[m].append(res)
            recon = np.stack([load_gray(d / f"recon_{i}.png", strict=True) for i in range(1, 5)])
            t = res["timings_ms"].get("synth_warm_ms", res["timings_ms"]["synth_ms"])
            rows.append({"label": label(m, syn.steps), "recon": recon, "planes": res["planes"], "summary": res["summary"],
                         "time_ms": t, "crosstalk": res["crosstalk"]})
            print(f"{name:>14} {m:>10}: КК {res['summary']['pcc']['mean']:.4f} "
                  f"ИСС {res['summary']['ssim_global']['mean']:.4f} селект. {res['crosstalk']['selectivity']:.3f} "
                  f"({res['crosstalk']['own_best']}/4) {t:9.1f} мс")
        if not a.no_grid:
            method_grid(targets, rows, out / name / "methods.png",
                        title=f"{name}: КК / ИСС (формула слайда) по плоскостям · амплитуда, max = 1 · {a.device.upper()}")

    summary = {"quads": [q[0] for q in quads], "device": a.device, "policy": a.policy, "methods": {}}
    for m, rs in per.items():
        times = [r["timings_ms"].get("synth_warm_ms", r["timings_ms"]["synth_ms"]) for r in rs]
        summary["methods"][m] = {
            "label": label(m, synths[m].steps), "steps": synths[m].steps, "time_ms_median": float(np.median(times)),
            **{f"{k}_mean": _agg(np.mean, rs, k, "mean") for k in METRICS},
            **{f"{k}_worst": _agg(np.mean, rs, k, "worst") for k in METRICS},   # mean of per-quad worst plane
            **{f"{k}_min": _agg(np.min, rs, k, "worst") for k in METRICS},
            "selectivity_mean": float(np.nanmean([r["crosstalk"]["selectivity"] for r in rs])),
            "own_best_mean": float(np.mean([r["crosstalk"]["own_best"] for r in rs])),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    lines = [f"Наборов: {len(quads)} ({a.quads}); устройство {a.device}; сравниваются амплитуды |K|, каждое "
             f"изображение нормировано на максимум = 1 (протокол экспертов). КК — корреляция Пирсона, "
             f"ИСС — формула слайда; «худш.» — худшая из 4 плоскостей, усреднённая по наборам (как в results/ml и results/baselines). Время — медиана синтеза одного ДОЭ. "
             f"Селективность = среднее диагонали − среднее вне диагонали матрицы КК(цель i, плоскость j); «своя лучшая» — "
             f"в скольких из 4 плоскостей восстановление больше всего похоже на свою цель (среднее по наборам).\n",
             "| Метод | шаги | время, мс | КК ср. | КК худш. | ИСС ср. | ИСС худш. | ИСС (окно 7×7) ср. | ИСС (гаусс 11×11) ср. | селективность | своя лучшая |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, s in summary["methods"].items():
        lines.append(f"| {s['label']} | {s['steps']} | {s['time_ms_median']:.1f} | {s['pcc_mean']:.4f} | "
                     f"{s['pcc_worst']:.4f} | {s['ssim_global_mean']:.4f} | {s['ssim_global_worst']:.4f} | "
                     f"{s['ssim_skimage_mean']:.4f} | {s['ssim_gauss_mean']:.4f} | {s['selectivity_mean']:.3f} | "
                     f"{s['own_best_mean']:.2f}/4 |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if len(summary["methods"]) >= 2:
        speed_plot(summary, ROOT / a.speed_plot, a.device, f"{len(quads)} ({a.quads})")


SHORT = {"backprop": "обратн. распр.", "gs": "ГС-{}", "adam": "Adam-{}", "nn": "сеть", "nn_refine": "сеть + {}"}


def speed_plot(summary, sp, device, quads_desc):
    from doe.visualize import speed_quality_plot

    pts = [{"label": SHORT[m].format(s["steps"]), "time_ms": s["time_ms_median"], **{k: v for k, v in s.items()
            if k != "label"}} for m, s in summary["methods"].items()]
    sp.parent.mkdir(parents=True, exist_ok=True)
    dev = {"cpu": "CPU (Apple M3 Pro)", "mps": "GPU Apple M3 Pro (MPS)"}.get(device, device)
    note = (f"Время всех методов измерено на одном устройстве: {dev}, один ДОЭ за вызов, без батчей. "
            f"Наборов: {quads_desc}. ГС — Гершберг–Сакстон, число — итерации/шаги.")
    cd = ROOT / "results/final/compare_deep.json"
    tm = ROOT / "results/final/timing_mac.json"
    if device == "cpu" and cd.exists() and "timing_mps_ms" in json.loads(cd.read_text()):
        t = json.loads(cd.read_text())["timing_mps_ms"]
        note += (f"\nДля справки, на GPU (MPS) той же машины: сеть {t['глубокая сеть']:.0f} мс, "
                 f"сеть + 50 шагов {t['глубокая сеть + 50']:.0f} мс (results/final/compare_deep.json).")
    elif device == "cpu" and tm.exists():
        t = json.loads(tm.read_text())
        rs = summary["methods"].get("nn_refine", {}).get("steps", 50)
        key = f"nn_refine{rs}_mps_ms"
        note += (f"\nДля справки, на GPU (MPS) той же машины: сеть {t['nn_mps_ms']:.0f} мс"
                 + (f", сеть + {rs} шагов {t[key]:.0f} мс" if key in t else "")
                 + " (results/final/timing_mac.json).")
    speed_quality_plot(pts, sp, title=f"Скорость и качество синтеза ДОЭ · {dev}", note=note,
                       metrics=(("pcc_mean", "КК Пирсона, среднее"), ("ssim_global_mean", "ИСС (формула слайда)"),
                                ("ssim_gauss_mean", "ИСС оконный (гаусс 11×11)"),
                                ("selectivity_mean", "селективность по глубине")))
    print(f"-> {sp}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--replot":       # --replot SUMMARY.json OUT.png
        sm = json.loads(Path(sys.argv[2]).read_text())
        speed_plot(sm, Path(sys.argv[3]), sm["device"], f"{len(sm['quads'])}")
    else:
        main()
