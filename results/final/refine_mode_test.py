"""Новый режим уточнения (200 шагов, штраф 0,25) против прежнего (50 шагов, штраф 1) и GS-200 на test-256, ДИ 95%."""
import numpy as np
from doe.io import phase_to_u8
from doe.metrics import crosstalk, evaluate
from doe.infer import NeuralDOE
from doe.baselines import gs_batch
KEYS = ("pcc", "pcc_worst", "ssim_global", "ssim_gauss", "sel", "own")
T = np.load("data/test.npy"); net = NeuralDOE.load("checkpoints/amp_deep6/best.pt")
def per_quad(ph):
    rows = []
    for p, t in zip(ph, T):
        r = evaluate(phase_to_u8(p), t); s = r["summary"]; x = crosstalk(t, r["recon"])
        rows.append([s["pcc"]["mean"], s["pcc"]["worst"], s["ssim_global"]["mean"], s["ssim_gauss"]["mean"], x["selectivity"], x["own_best"]])
    return np.array(rows, float)
def boot(x):
    m = np.random.default_rng(0).choice(x, (2000, len(x))).mean(1); return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
modes = {"сеть + 50, штраф 1 (прежний)": net.predict_refined_batch(T, 50, xtalk=1.0),
         "сеть + 200, штраф 0,25 (новый)": net.predict_refined_batch(T, 200, xtalk=0.25),
         "сеть + 200, штраф 0,5": net.predict_refined_batch(T, 200, xtalk=0.5),
         "GS-200": gs_batch(T, 200)}
pq = {k: per_quad(v) for k, v in modes.items()}
f = lambda v: f"{v[0]:.3f} [{v[1]:.3f}; {v[2]:.3f}]"
L = ["| режим | КК ср. | КК худш. | ИСС глоб. | ИСС gauss | селективность | своя лучшая |", "|---|---|---|---|---|---|---|"]
L += [f"| {k} | " + " | ".join(f(boot(a[:, i])) for i in range(6)) + " |" for k, a in pq.items()]
open("results/final/refine_mode_test.md", "w", encoding="utf-8").write("# Режимы уточнения на TEST-256 (ДИ 95%)\n\n" + "\n".join(L) + "\n")
print("\n".join(L))
