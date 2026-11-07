"""Режимы сдачи на TEST-256 (ДИ 95%): быстрый (сеть) и основной (сеть + 100 шагов, штраф 0,25, ИСС×2) против GS-200."""
import json, numpy as np
from doe.io import phase_to_u8
from doe.metrics import crosstalk, evaluate
from doe.infer import NeuralDOE
from doe.baselines import gs_batch
T = np.load("data/test.npy"); net = NeuralDOE.load("checkpoints/amp_deep6/best.pt")
def per_quad(ph):
    r_ = []
    for p, t in zip(ph, T):
        r = evaluate(phase_to_u8(p), t); s = r["summary"]; x = crosstalk(t, r["recon"])
        r_.append([s["ssim_global"]["mean"], s["ssim_skimage"]["mean"], s["ssim_gauss"]["mean"], s["pcc"]["mean"], s["pcc"]["worst"], x["selectivity"]])
    return np.array(r_, float)
def boot(x):
    m = np.random.default_rng(0).choice(x, (2000, len(x))).mean(1); return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
modes = {"быстрый: сеть, 1 проход": net.predict_batch(T),
         "основной: сеть + 100 шагов": net.predict_refined_batch(T, 100, xtalk=0.25, alpha=1.0, beta=0.5),
         "GS-200 (классика, для сравнения)": gs_batch(T, 200)}
cols = ["ИСС (слайд, глоб.)", "ИСС skimage 7×7", "ИСС gauss 11×11", "Pearson ср.", "Pearson худш. плоск.", "селективность"]
f = lambda v: f"{v[0]:.3f} [{v[1]:.3f}; {v[2]:.3f}]"
L = ["| режим | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
res = {}
for k, ph in modes.items():
    a = per_quad(ph); res[k] = [boot(a[:, i]) for i in range(len(cols))]
    L.append(f"| {k} | " + " | ".join(f(v) for v in res[k]) + " |")
open("results/final/final_mode_test.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
json.dump(res, open("results/final/final_mode_test.json", "w"), ensure_ascii=False, indent=1)
print("\n".join(L))
