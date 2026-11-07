"""Глубокая модель (amp_deep6, 6 стадий) против amp_xtalk и классики на test-256: среднее и bootstrap-ДИ 95%,
парные разности. Протокол экспертов + селективность. Время на MPS — отдельно, одна голограмма."""
import json, math, time
from pathlib import Path
import numpy as np, torch
from doe.io import phase_to_u8
from doe.metrics import crosstalk, evaluate
from doe.infer import NeuralDOE
from doe.baselines import gs_batch
from doe.losses import per_sample_loss, quantize_phase

KEYS = ("pcc", "pcc_worst", "ssim_global", "ssim_gauss", "sel", "own")
RU = {"pcc": "КК ср.", "pcc_worst": "КК худш.", "ssim_global": "ИСС глоб.", "ssim_gauss": "ИСС gauss", "sel": "селективность", "own": "своя лучшая"}
T = np.load("data/test.npy")
deep = NeuralDOE.load("checkpoints/amp_deep6/best.pt"); xt = NeuralDOE.load("checkpoints/amp_xtalk/best.pt")

def per_quad(ph):
    rows = []
    for p, t in zip(ph, T):
        r = evaluate(phase_to_u8(p), t); s = r["summary"]; x = crosstalk(t, r["recon"])
        rows.append([s["pcc"]["mean"], s["pcc"]["worst"], s["ssim_global"]["mean"], s["ssim_gauss"]["mean"], x["selectivity"], x["own_best"]])
    return np.array(rows, float)
def boot(x, n=2000):
    x = x[np.isfinite(x)]; m = np.random.default_rng(0).choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def refine_from(net, p0, steps=50):
    X = torch.tensor(T, dtype=torch.float32, device=net.device) / 255; fr = net.model.fresnel
    ph = torch.tensor(p0, dtype=torch.float32, device=net.device).requires_grad_(True); opt = torch.optim.Adam([ph], lr=0.05)
    for _ in range(steps):
        l, _ = per_sample_loss(ph, X, fr, net.loss_cfg); opt.zero_grad(); l.sum().backward(); opt.step()
    return (torch.remainder(ph.detach() + math.pi, 2 * math.pi) - math.pi).cpu().numpy().astype(np.float64)

g50 = gs_batch(T, 50)
methods = {
    "глубокая сеть (amp_deep6)": deep.predict_batch(T),
    "глубокая сеть + 50": deep.predict_refined_batch(T, steps=50),
    "сеть amp_xtalk": xt.predict_batch(T),
    "сеть amp_xtalk + 50": xt.predict_refined_batch(T, steps=50),
    "GS-50 + 50 (классика)": refine_from(xt, g50),
    "GS-200 (классика)": gs_batch(T, 200),
}
pq = {k: per_quad(v) for k, v in methods.items()}
res = {"n": len(T), "methods": {k: {key: boot(a[:, i]) for i, key in enumerate(KEYS)} for k, a in pq.items()}, "paired": {}}
for a, b in (("глубокая сеть (amp_deep6)", "сеть amp_xtalk"), ("глубокая сеть (amp_deep6)", "GS-50 + 50 (классика)"),
             ("глубокая сеть + 50", "GS-50 + 50 (классика)"), ("глубокая сеть (amp_deep6)", "GS-200 (классика)")):
    d = pq[a] - pq[b]; res["paired"][f"{a} − {b}"] = {key: boot(d[:, i]) for i, key in enumerate(KEYS)}
# timing, one hologram, MPS
t1 = T[0]; tm = {}
for name, fn in (("глубокая сеть", lambda: deep.predict(t1)), ("глубокая сеть + 50", lambda: deep.predict_refined(t1, 50)),
                 ("сеть amp_xtalk", lambda: xt.predict(t1))):
    for _ in range(3): fn()
    ts = []
    for _ in range(10):
        s = time.perf_counter(); fn(); ts.append(time.perf_counter() - s)
    tm[name] = round(1e3 * float(np.median(ts)), 1)
res["timing_mps_ms"] = tm
json.dump(res, open("results/final/compare_deep.json", "w"), indent=1, ensure_ascii=False)
f = lambda v: f"{v[0]:.3f} [{v[1]:.3f}; {v[2]:.3f}]"
L = ["# Глубокая модель (6 стадий) против прежней и классики — TEST-256, протокол экспертов", "",
     "| метод | " + " | ".join(RU[k] for k in KEYS) + " |", "|---|" + "---|" * len(KEYS)]
L += [f"| {k} | " + " | ".join(f(v[key]) for key in KEYS) + " |" for k, v in res["methods"].items()]
L += ["", "## Парные разности", "", "| разность | " + " | ".join(RU[k] for k in KEYS) + " |", "|---|" + "---|" * len(KEYS)]
L += [f"| {k} | " + " | ".join(f(v[key]) for key in KEYS) + " |" for k, v in res["paired"].items()]
L += ["", "Время одной голограммы, GPU Mac (MPS): " + ", ".join(f"{k} — {v} мс" for k, v in tm.items())]
Path("results/final/compare_deep.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))
