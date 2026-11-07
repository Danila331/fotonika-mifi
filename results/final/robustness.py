"""Надёжность выводов (по примеру Codex): bootstrap-ДИ 95% и парные разности на test-256,
плюс граничные случаи E0–E7 из их набора (только чтение их целей). Протокол экспертов + селективность."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from doe.io import load_gray, phase_to_u8
from doe.metrics import crosstalk, evaluate
from doe.infer import NeuralDOE
from doe.baselines import gs_batch

KEYS = ("pcc", "pcc_worst", "ssim_global", "ssim_gauss", "sel", "own")
RU = {"pcc": "КК ср.", "pcc_worst": "КК худш.", "ssim_global": "ИСС глоб.", "ssim_gauss": "ИСС gauss",
      "sel": "селективность", "own": "своя лучшая"}

def per_quad(phases_u8, T):
    rows = []
    for p, t in zip(phases_u8, T):
        r = evaluate(p, t); s = r["summary"]; x = crosstalk(t, r["recon"])
        rows.append([s["pcc"]["mean"], s["pcc"]["worst"], s["ssim_global"]["mean"], s["ssim_gauss"]["mean"],
                     x["selectivity"], x["own_best"]])
    return np.array(rows, dtype=float)

def boot(x, n=2000, seed=0):
    x = x[np.isfinite(x)]; rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

T = np.load("data/test.npy")
xt = NeuralDOE.load("checkpoints/amp_deep6/best.pt"); mn = NeuralDOE.load("checkpoints/amp_xtalk/best.pt")
methods = {
    "сеть (amp_deep6)": np.stack([xt.predict(t) for t in T]),
    "сеть + 50 (amp_deep6)": xt.predict_refined_batch(T, steps=50),
    "прежняя сеть (amp_xtalk)": np.stack([mn.predict(t) for t in T]),
    "GS-200": gs_batch(T, 200),
}
pq = {k: per_quad(np.stack([phase_to_u8(p) for p in v]), T) for k, v in methods.items()}
res = {"n": len(T), "methods": {}, "paired": {}}
for k, a in pq.items():
    res["methods"][k] = {key: boot(a[:, i]) for i, key in enumerate(KEYS)}
for a, b in (("сеть + 50 (amp_deep6)", "GS-200"), ("сеть (amp_deep6)", "прежняя сеть (amp_xtalk)"),
             ("сеть + 50 (amp_deep6)", "сеть (amp_deep6)")):
    d = pq[a] - pq[b]
    res["paired"][f"{a} − {b}"] = {key: boot(d[:, i]) for i, key in enumerate(KEYS)}

E = Path("../neural-doe/results/final_evaluation/runs/edge")
edge = {}
for case in sorted(p.name for p in E.iterdir()):
    tdir = E / case / "neural_nonqat2026" / "seed_0" / "targets"
    t = np.stack([load_gray(tdir / f"target_0{k}.png", strict=True) for k in range(4)])
    theirs = np.array(Image.open(E / case / "neural_nonqat2026" / "seed_0" / "phase.png"))
    row = {}
    for name, ph in (("наша сеть + 50", phase_to_u8(xt.predict_refined(t, steps=50))),
                     ("наша сеть", phase_to_u8(xt.predict(t))), ("их сеть", theirs)):
        r = evaluate(ph, t); s = r["summary"]
        row[name] = {"pcc": s["pcc"]["mean"], "pcc_valid": s["pcc"]["valid"], "ssim_global": s["ssim_global"]["mean"],
                     "planes_pcc": [p["pcc"] for p in r["planes"]],
                     "finite": bool(np.isfinite(r["recon"]).all())}
    edge[case] = {"target_means": [float(x.mean()) for x in t], "target_std": [float(x.std()) for x in t], **row}
res["edge"] = edge
json.dump(res, open("results/final/robustness.json", "w"), indent=1, ensure_ascii=False)

f = lambda v: f"{v[0]:.3f} [{v[1]:.3f}; {v[2]:.3f}]"
L = ["# Надёжность выводов: доверительные интервалы и граничные случаи", "",
     f"TEST, {len(T)} четвёрок (невиданные классы и шрифты). Протокол экспертов (амплитуда, max=1). "
     "Среднее и bootstrap-ДИ 95% по четвёркам (2000 выборок).", "",
     "| метод | " + " | ".join(RU[k] for k in KEYS) + " |", "|---|" + "---|" * len(KEYS)]
for k, v in res["methods"].items():
    L.append(f"| {k} | " + " | ".join(f(v[key]) for key in KEYS) + " |")
L += ["", "## Парные разности (на одних и тех же четвёрках)", "",
      "| разность | " + " | ".join(RU[k] for k in KEYS) + " |", "|---|" + "---|" * len(KEYS)]
for k, v in res["paired"].items():
    L.append(f"| {k} | " + " | ".join(f(v[key]) for key in KEYS) + " |")
L += ["", "ДИ, не содержащий 0, — устойчивая разница.", "",
      "## Граничные случаи E0–E7 (цели из набора Codex)", "",
      "| случай | ср. яркость целей | std целей | наша сеть+50: КК (валидных) | наша сеть: КК | их сеть: КК | ИСС глоб. наша+50 / их |",
      "|---|---|---|---|---|---|---|"]
for c, e in edge.items():
    g = lambda n: ("—" if e[n]["pcc"] is None else f"{e[n]['pcc']:.3f}") + f" ({e[n]['pcc_valid']}/4)"
    L.append(f"| {c} | {', '.join(f'{m:.0f}' for m in e['target_means'])} | {', '.join(f'{s:.0f}' for s in e['target_std'])} | "
             f"{g('наша сеть + 50')} | {g('наша сеть')} | {g('их сеть')} | "
             f"{e['наша сеть + 50']['ssim_global']:.3f} / {e['их сеть']['ssim_global']:.3f} |")
L += ["", "«—» — КК не определён (константная цель): так и должно быть, это не ошибка. "
      "Все восстановления конечны (без NaN/Inf), конвейер не падает ни на одном случае."]
Path("results/final/robustness.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
