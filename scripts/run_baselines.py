"""Quality ceiling and speed of the classic iterative baselines (GS, Adam) under the jury rules
(E1: amplitudes |K|, E2: every image normalised to max = 1).

Every reported number is computed by doe.metrics.evaluate(phase_to_u8(phase), targets) (amplitude, peak),
i.e. from the 8-bit phase, except rows explicitly labelled "continuous" / "intensity".

  .venv/bin/python scripts/run_baselines.py --stage losses              # loss study (val[:12], 300 steps)
  .venv/bin/python scripts/run_baselines.py --tag v1 --n-val 16         # compact main table on a data set
  .venv/bin/python scripts/run_baselines.py --tag v2 --n-val 32         # same after data/v2.ready
  .venv/bin/python scripts/run_baselines.py --md-only                   # rebuild summary.md from summary.json
  add --quick for a smoke test (tiny sets and budgets, writes to results/baselines/quick/)

Outputs: results/baselines/{summary.json, summary.md (Russian), curves_<tag>.png, per_plane_<tag>.png,
examples_<tag>_<demo>.png, phases_<tag>.npz}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from doe import baselines as B  # noqa: E402
from doe.config import DEPTHS  # noqa: E402
from doe.io import load_gray, phase_to_u8  # noqa: E402
from doe.metrics import compare, evaluate, to_uint8  # noqa: E402
from doe.physics import intensity_np  # noqa: E402

OUT = ROOT / "results" / "baselines"
METRICS = ("pcc", "ssim_global", "ssim_skimage", "ssim_gauss")
CURVE_ITS = (1, 5, 10, 20, 50, 100, 200, 500, 1000)
J_DEF = "J = 0.5·PCC_mean + 0.5·mean(SSIM_global, SSIM_skimage, SSIM_gauss)"


# ------------------------------------------------------------------------------------------------ data

def load_sets(n_val: int) -> dict:
    sets = {}
    demo_dir = ROOT / "data" / "demo"
    if demo_dir.is_dir() and any(demo_dir.iterdir()):
        names = sorted(d.name for d in demo_dir.iterdir() if (d / "target_1.png").exists())
        sets["demo"] = (np.stack([np.stack([load_gray(demo_dir / n / f"target_{i}.png") for i in range(1, 5)])
                                  for n in names]), names)
    else:
        from doe.data import QuadGenerator
        t = QuadGenerator("test", 0).sample(16)
        sets["demo"] = (t, [f"test{i}" for i in range(len(t))])
    val_path = ROOT / "data" / "val.npy"
    if val_path.exists():
        v = np.load(val_path)[:n_val]
    else:
        from doe.data import QuadGenerator
        v = QuadGenerator("val", 0).sample(n_val)
    sets["val"] = (v, [f"val{i}" for i in range(len(v))])
    return sets


# ------------------------------------------------------------------------------------------ evaluation

def _recon_continuous(phase, targets, policy, quantity):
    x = intensity_np(phase)
    x = np.sqrt(x) if quantity == "amplitude" else x
    return np.stack([to_uint8(x[i], policy, targets[i]) for i in range(len(DEPTHS))])


def eval_phases(phases, targets, policy="peak", continuous=False, quantity="amplitude") -> dict:
    """Aggregate jury metrics over a set. Quantised path = doe.metrics.evaluate(phase_to_u8(phase))."""
    vals = {k: np.full((len(targets), len(DEPTHS)), np.nan) for k in METRICS}
    for q, (ph, t) in enumerate(zip(phases, targets)):
        if continuous:
            rec = _recon_continuous(ph, t, policy, quantity)
            planes = [compare(t[i], rec[i]) for i in range(len(DEPTHS))]
        else:
            planes = evaluate(phase_to_u8(ph), t, policy=policy, quantity=quantity)["planes"]
        for i, p in enumerate(planes):
            for k in METRICS:
                if p[k] is not None:
                    vals[k][q, i] = p[k]
    out = {}
    for k in METRICS:
        v = vals[k]
        out[k] = {"mean": float(np.nanmean(v)), "worst": float(np.nanmean(np.nanmin(v, 1))),
                  "min": float(np.nanmin(v)), "per_plane": [float(x) for x in np.nanmean(v, 0)]}
    out["J"] = float(0.5 * out["pcc"]["mean"] + 0.5 * np.mean([out[k]["mean"] for k in METRICS[1:]]))
    kinds = image_kinds(targets)
    out["by_kind"] = {}
    for kind, mask in (("halftone", kinds), ("simple", ~kinds)):
        if mask.any():
            d = {k: float(np.nanmean(vals[k][mask])) for k in METRICS}
            d["J"] = 0.5 * d["pcc"] + 0.5 * float(np.mean([d[k] for k in METRICS[1:]]))
            d["n_images"] = int(mask.sum())
            out["by_kind"][kind] = d
    return out


def image_kinds(targets) -> np.ndarray:
    """(Q,4) bool: True = halftone (rich grey levels), False = simple (>90 % of pixels near 0 or near max)."""
    t = np.asarray(targets, dtype=np.float64)
    m = t.max((-2, -1), keepdims=True) + 1e-9
    near = ((t <= 0.1 * m) | (t >= 0.9 * m)).mean((-2, -1))
    return near < 0.9


def short(e: dict) -> str:
    return (f"pcc {e['pcc']['mean']:.3f}/{e['pcc']['worst']:.3f} sg {e['ssim_global']['mean']:.3f} "
            f"sk {e['ssim_skimage']['mean']:.3f} sgs {e['ssim_gauss']['mean']:.3f} J {e['J']:.3f}")


def load_summary(path) -> dict:
    if path.exists():
        R = json.loads(path.read_text())
        if R.get("version") == 2:
            return R
    return {"version": 2, "rules": "amplitude |K|, every image normalised to max=1 (E1/E2)", "J": J_DEF,
            "loss_study": {}, "runs": {}}


# ---------------------------------------------------------------------------------------------- stages

LOSS_EXTRA = {
    "w1_0_1_soft": dict(w_pcc=1, w_ssim_g=0, w_ssim_l=1, norm="soft"),
    "w3_1_1_soft": dict(w_pcc=3, w_ssim_g=1, w_ssim_l=1, norm="soft"),
    "w2_05_1_peak": dict(w_pcc=2, w_ssim_g=0.5, w_ssim_l=1, norm="peak"),
    "w1_05_2_soft": dict(w_pcc=1, w_ssim_g=0.5, w_ssim_l=2, norm="soft"),
    "w2_05_1_soft_b20": dict(w_pcc=2, w_ssim_g=0.5, w_ssim_l=1, norm="soft", beta=20),
}


def stage_losses(R, args):
    sets = load_sets(12 if not args.quick else 2)
    T = sets["val"][0]
    steps = 300 if not args.quick else 10
    losses = dict(B.LOSSES)
    losses.update({k: B.make_loss(**v) for k, v in LOSS_EXTRA.items()})
    res = {}
    for name, fn in losses.items():
        t0 = time.perf_counter()
        ph = B.adam_batch(T, steps, loss=fn)
        dt = time.perf_counter() - t0
        res[name] = {"steps": steps, "sec_per_hologram": dt / len(T), "val": eval_phases(ph, T),
                     "spec": LOSS_EXTRA.get(name, "doe.baselines.LOSSES")}
        print(f"loss {name:20s} {short(res[name]['val'])} ({dt:.0f} s)", flush=True)
    R["loss_study"] = {"n_quads": len(T), "steps": steps, "lr": 0.15, "init": "backprop",
                       "jury_loss_now": B.JURY_LOSS_NAME, "results": res}


def stage_main(R, args):
    tag = args.tag
    its = CURVE_ITS if not args.quick else (1, 5, 20)
    gs_rows = (20, 100, 500) if not args.quick else (5, 20)
    adam_rows = (20, 100, 500) if not args.quick else (5,)
    n_adam = its[-1]
    sets = load_sets(args.n_val if not args.quick else 2)
    if args.quick:
        sets = {k: (v[0][:2], v[1][:2]) for k, v in sets.items()}
    allT = np.concatenate([sets["demo"][0], sets["val"][0]])
    nd = len(sets["demo"][0])
    split = {"demo": slice(0, nd), "val": slice(nd, len(allT))}
    print(f"[{tag}] demo {nd} quads, val {len(allT) - nd} quads", flush=True)
    B.adam_batch(allT[:2], 2)  # warm-up (FFT plans, kernel caches)

    run = {"config": {"n_demo": nd, "n_val": len(allT) - nd, "demo_names": sets["demo"][1], "device": "cpu",
                      "threads": torch.get_num_threads(), "jury_loss": B.JURY_LOSS_NAME, "lr": 0.15,
                      "iterations": list(its), "loadavg_start": os.getloadavg()[0]},
           "methods": {}, "curves": {}, "quantisation": {}, "policy": {}, "intensity_vs_amplitude": {},
           "timing_single": {}}
    finals = {}

    def record(name, phases, t_total, n_iter, note=""):
        finals[name] = phases
        e = {"iterations": n_iter, "sec_per_hologram_batched": t_total / len(allT), "note": note}
        for s, sl in split.items():
            e[s] = eval_phases(phases[sl], allT[sl])
        run["methods"][name] = e
        print(f"{name:22s} {t_total / len(allT) * 1e3:8.1f} ms/holo | val {short(e['val'])} | demo {short(e['demo'])}",
              flush=True)

    def curve(name, snaps, timings, ks):
        run["curves"][name] = {k: {"sec_per_hologram": timings[k] / len(allT),
                                   **{s: eval_phases(snaps[k][sl], allT[sl]) for s, sl in split.items()}}
                               for k in ks}
        print(f"curve {name}: " + " ".join(f"{k}:{run['curves'][name][k]['val']['J']:.3f}" for k in ks), flush=True)

    t0 = time.perf_counter()
    record("backprop_init", B.backprop_init(allT).double().numpy(), time.perf_counter() - t0, 0)

    # GS: deterministic, so snapshots of one run equal independent runs
    gs_its = tuple(k for k in its if k <= gs_rows[-1])
    for name, kw in (("gs", {}), ("gs_weighted", {"weighted": True})):
        tim = {}
        _, snaps = B.gs_batch(allT, gs_rows[-1], record_at=gs_its, timings=tim, **kw)
        curve(name, snaps, tim, gs_its)
        for k in (gs_rows if name == "gs" else gs_rows[-1:]):
            record(f"{name}_{k}", snaps[k], tim[k], k)

    # Adam with the jury loss: one long run for the curve (+ its final row), independent shorter runs
    tim = {}
    fin, snaps = B.adam_batch(allT, n_adam, record_at=its, timings=tim)
    curve("adam", snaps, tim, its)
    for k in adam_rows:
        t0 = time.perf_counter()
        record(f"adam_{k}", B.adam_batch(allT, k), time.perf_counter() - t0, k)
    record(f"adam_{n_adam}", fin, tim[n_adam], n_adam)

    # Adam initialised by GS
    k_gs, k_ad = gs_rows[1], adam_rows[-1]
    t0 = time.perf_counter()
    p_gs = B.gs_batch(allT, k_gs)
    ph = B.adam_batch(allT, k_ad, init_phase=p_gs)
    record(f"gs{k_gs}+adam{k_ad}", ph, time.perf_counter() - t0, k_gs + k_ad, note="GS init, jury loss")

    # intensity-trained Adam / GS (old convention) evaluated by the amplitude rules
    t0 = time.perf_counter()
    record(f"adam_int_{k_ad}", B.adam_batch(allT, k_ad, loss="jury_int"), time.perf_counter() - t0, k_ad,
           note="loss on |K|^2 (old intensity convention)")
    t0 = time.perf_counter()
    record(f"gs_int_{k_gs}", B.gs_batch(allT, k_gs, quantity="intensity"), time.perf_counter() - t0, k_gs,
           note="GS with |K|^2 ∝ img (old)")

    cand = {k: v for k, v in run["methods"].items() if k != "backprop_init" and "_int_" not in k}
    best = max(cand, key=lambda k: cand[k]["val"]["J"])
    run["best"] = best
    print("best by J:", best, flush=True)

    # same phases scored under the old intensity metric (for the record)
    for name in (f"adam_{k_ad}", f"adam_int_{k_ad}"):
        run["intensity_vs_amplitude"][name] = {
            "amplitude": {s: run["methods"][name][s] for s in split},
            "intensity": {s: eval_phases(finals[name][sl], allT[sl], quantity="intensity") for s, sl in split.items()}}

    for name in dict.fromkeys([best, f"gs_{k_gs}"]):
        run["quantisation"][name] = {"continuous": {s: eval_phases(finals[name][sl], allT[sl], continuous=True)
                                                    for s, sl in split.items()},
                                     "8bit": {s: run["methods"][name][s] for s in split}}
        run["policy"][name] = {pol: {s: eval_phases(finals[name][sl], allT[sl], policy=pol) for s, sl in split.items()}
                               for pol in ("peak", "minmax", "fit")}
        print(f"quant {name}: cont {short(run['quantisation'][name]['continuous']['val'])}", flush=True)

    if not args.quick:
        run["timing_single"] = single_timing(allT[:1])
    run["config"]["loadavg_end"] = os.getloadavg()[0]
    R["runs"][tag] = run
    np.savez_compressed(OUT / f"phases_{tag}.npz", targets=allT,
                        **{k: phase_to_u8(finals[k]) for k in dict.fromkeys([best, f"gs_{k_gs}"])})
    plots(run, tag, finals, allT, sets, k_gs)


def single_timing(t1) -> dict:
    """Wall time of single-hologram (B=1) runs on CPU, plus the machine load at measurement time."""
    B.adam_batch(t1, 2)
    out = {"loadavg_1min": os.getloadavg()[0], "threads": torch.get_num_threads()}
    for name, fn in (("gs_100", lambda: B.gs_batch(t1, 100)), ("gs_500", lambda: B.gs_batch(t1, 500)),
                     ("adam_100", lambda: B.adam_batch(t1, 100)), ("adam_500", lambda: B.adam_batch(t1, 500))):
        t0 = time.perf_counter()
        fn()
        out[name] = time.perf_counter() - t0
        print(f"single {name}: {out[name]:.2f} s", flush=True)
    return out


# ----------------------------------------------------------------------------------------------- plots

def plots(run, tag, finals, allT, sets, k_gs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    for name, c in run["curves"].items():
        ks = [int(k) for k in c]
        for ax, m in zip(axes, METRICS):
            ax.plot(ks, [c[k]["val"][m]["mean"] for k in c], marker="o", ms=3, label=name)
    for ax, m in zip(axes, ("PCC", "SSIM global (слайд)", "SSIM skimage 7×7", "SSIM gauss 11×11")):
        ax.set_xscale("log")
        ax.set_xlabel("итерации")
        ax.set_title(f"{m}, val, 8-бит фаза")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"curves_{tag}.png", dpi=110)
    plt.close(fig)

    gs_name = f"gs_{k_gs}"
    names = [n for n in dict.fromkeys([run["best"], "adam_100", gs_name, "gs_weighted_500"]) if n in run["methods"]]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    x = np.arange(len(DEPTHS))
    for ax, m in zip(axes, METRICS):
        for j, n in enumerate(names):
            ax.bar(x + j * 0.8 / len(names), run["methods"][n]["val"][m]["per_plane"], 0.8 / len(names), label=n)
        ax.set_xticks(x + 0.4 - 0.4 / len(names))
        ax.set_xticklabels([f"{z * 1e3:.1f}" for z in DEPTHS])
        ax.set_xlabel("глубина, мм")
        ax.set_title(m)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / f"per_plane_{tag}.png", dpi=110)
    plt.close(fig)

    demo_T, demo_names = sets["demo"]
    pref = ("digits_1234", "rocket", "halftone", "photo")
    pick = []
    for p in pref:
        pick += [i for i, n in enumerate(demo_names) if p in n and i not in pick]
    pick = (pick + [i for i in range(len(demo_names)) if i not in pick])[:2]
    for i in pick:
        rows = [("цель", demo_T[i])]
        for n in dict.fromkeys([gs_name, run["best"]]):
            rows.append((n, evaluate(phase_to_u8(finals[n][i]), demo_T[i])["recon"]))
        fig, axes = plt.subplots(len(rows), 5, figsize=(13, 2.7 * len(rows)))
        for r, (lab, imgs) in enumerate(rows):
            for p in range(4):
                ax = axes[r, p]
                ax.imshow(imgs[p], cmap="gray", vmin=0, vmax=255)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"z = {DEPTHS[p] * 1e3:.1f} мм")
                else:
                    m = compare(demo_T[i][p], imgs[p])
                    ax.set_title(f"PCC {m['pcc'] or 0:.2f}  SSIMg {m['ssim_gauss']:.2f}", fontsize=8)
            axes[r, 0].set_ylabel(lab, fontsize=8)
            if r > 0:
                axes[r, 4].imshow(phase_to_u8(finals[lab][i]), cmap="twilight")
                axes[r, 4].set_title("фаза (8 бит)", fontsize=8)
            axes[r, 4].axis("off")
        fig.suptitle(f"демо '{demo_names[i]}': цели и восстановления |K| (норм. на максимум)")
        fig.tight_layout()
        fig.savefig(OUT / f"examples_{tag}_{demo_names[i]}.png", dpi=105)
        plt.close(fig)


# ---------------------------------------------------------------------------------------------- report

HDR = ("| метод | PCC ср. | PCC худш. | SSIM global | SSIM skimage | SSIM gauss | J | время/голограмму |\n"
       "|---|---|---|---|---|---|---|---|")


def _row(name, e, sec=""):
    return (f"| {name} | {e['pcc']['mean']:.3f} | {e['pcc']['worst']:.3f} | {e['ssim_global']['mean']:.3f} | "
            f"{e['ssim_skimage']['mean']:.3f} | {e['ssim_gauss']['mean']:.3f} | {e['J']:.3f} | {sec} |")


def _sec(s):
    return f"{s * 1e3:.0f} мс" if s < 1 else f"{s:.1f} с"


def conclusions(R) -> list[str]:
    """Short auto-generated 'Выводы' block from the latest run."""
    if not R["runs"]:
        return []
    tag = list(R["runs"])[-1]
    run = R["runs"][tag]
    M = run["methods"]
    v = lambda n: M[n]["val"]
    best = run["best"]
    b = v(best)
    L = [f"## Выводы (набор `{tag}`)", "",
         f"- **Потолок (лучший метод `{best}`, val):** PCC {b['pcc']['mean']:.3f} (худшая плоскость в среднем "
         f"{b['pcc']['worst']:.3f}), SSIM global {b['ssim_global']['mean']:.3f}, SSIM skimage "
         f"{b['ssim_skimage']['mean']:.3f}, SSIM gauss {b['ssim_gauss']['mean']:.3f}; на демо PCC "
         f"{M[best]['demo']['pcc']['mean']:.3f}, SSIM gauss {M[best]['demo']['ssim_gauss']['mean']:.3f}."]
    if "gs_100" in M and "adam_100" in M:
        L.append(f"- **GS против Adam:** GS быстро выходит на плато (J: 20 ит. {v('gs_20')['J']:.3f}, 100 ит. "
                 f"{v('gs_100')['J']:.3f}, 500 ит. {v('gs_500')['J']:.3f}); Adam с `jury_loss` при 100 шагах уже "
                 f"{v('adam_100')['J']:.3f}, при 500 — {v('adam_500')['J']:.3f}. Старт с GS почти не помогает "
                 "(в пределах шума).")
    t = run.get("timing_single") or {}
    if t:
        L.append(f"- **Время одной голограммы (CPU, B=1, load avg {t['loadavg_1min']:.0f}):** GS 100 ит. "
                 f"{t['gs_100']:.2f} с, GS 500 ит. {t['gs_500']:.2f} с, Adam 100 шагов {t['adam_100']:.2f} с, "
                 f"Adam 500 шагов {t['adam_500']:.2f} с.")
    if R["loss_study"]:
        res = R["loss_study"]["results"]
        w = max(res, key=lambda k: res[k]["val"]["J"])
        L.append(f"- **Лосс:** лучший по J — `{w}` (J {res[w]['val']['J']:.3f}); в коде `jury_loss` = "
                 f"`{B.JURY_LOSS_NAME}`: 2·(1−PCC) + 0.5·(1−SSIM_global) + 1·(1−SSIM_gauss11) по амплитуде, "
                 "нормировка на мягкий максимум (beta=20). Чистый 1−PCC даёт максимум PCC, но низкий SSIM; "
                 "`jury_worst` поднимает худшую плоскость ценой ~0.01 J.")
    q = run["quantisation"].get(best)
    if q:
        L.append(f"- **8-битное квантование фазы:** ΔJ = {q['8bit']['val']['J'] - q['continuous']['val']['J']:+.4f} "
                 "— пренебрежимо.")
    iva = run["intensity_vs_amplitude"]
    ka = [k for k in iva if "_int_" not in k]
    ki = [k for k in iva if "_int_" in k]
    if ka and ki:
        L.append(f"- **Интенсивность или амплитуда:** Adam, обученный на старом интенсивностном лоссе, по правилам "
                 f"жюри даёт J {iva[ki[0]]['amplitude']['val']['J']:.3f} против "
                 f"{iva[ka[0]]['amplitude']['val']['J']:.3f} у амплитудного — лосс обязательно по |K|.")
    pp = b["pcc"]["per_plane"]
    L.append(f"- **По плоскостям ({best}, PCC):** " + ", ".join(f"{z * 1e3:.1f} мм {x:.3f}" for z, x in
                                                                  zip(DEPTHS, pp)) +
             " — средние плоскости хуже крайних (у них по два соседа и больше перекрёстных помех).")
    bk = b.get("by_kind", {})
    if "halftone" in bk and "simple" in bk:
        L.append(f"- **Полутоновые и простые изображения ({best}):** J {bk['halftone']['J']:.3f} против "
                 f"{bk['simple']['J']:.3f}; у бинарных картинок окно SSIM сильно штрафует спекл на тёмном фоне "
                 f"(SSIM gauss {bk['simple']['ssim_gauss']:.3f}).")
    L.append("")
    return L


def write_md(R):
    L = ["# Бейзлайны: потолок качества и скорость классических итерационных методов", "",
         "**Правила жюри (E1/E2):** сравниваются амплитуды |K|; цель и восстановление нормируются на свой максимум (=1). "
         "Все числа — `doe.metrics.evaluate(phase_to_u8(phase), targets)` (амплитуда, политика `peak`), т.е. по "
         "8-битной фазе, если не указано иное. «PCC худш.» — среднее по четвёркам от худшей плоскости. "
         f"{R['J']}. CPU (6 потоков, один процесс); «время/голограмму» в таблицах — время батча / число голограмм "
         "(машина была загружена другими задачами, это верхняя оценка; см. время для одной голограммы).",
         "", "Методы: `gs_K` — многоплоскостной Гершберг–Сэкстон (глобальный, усреднение 4 полей ДОЭ), K итераций; "
         "`gs_weighted` — с адаптивными весами плоскостей; `adam_K` — прямая оптимизация фазы Adam (lr 0.15, косинус) с "
         f"лоссом `jury_loss` (`{B.JURY_LOSS_NAME}`), K шагов, старт — обратное распространение суммы целей со "
         "случайной фазой; `gsA+adamB` — Adam со старта GS; `*_int_*` — старая конвенция |K|² ∝ изображение.", ""]
    L += conclusions(R)
    for tag, run in R["runs"].items():
        c = run["config"]
        L += [f"## Набор `{tag}`: демо {c['n_demo']} четвёрок ({', '.join(c['demo_names'])}), "
              f"val — первые {c['n_val']} четвёрок data/val.npy", "",
              f"Лучший по J на val: **{run['best']}**. Загрузка машины (load avg): {c['loadavg_start']:.0f} → "
              f"{c.get('loadavg_end', float('nan')):.0f}.", ""]
        for s in ("val", "demo"):
            L += [f"### Основная таблица — {s}", "", HDR]
            for k, v in run["methods"].items():
                L.append(_row(k, v[s], _sec(v["sec_per_hologram_batched"])))
            L.append("")
        L += ["### Качество от числа итераций (val; накопленное время на голограмму)", "",
              "| метод | итерации | PCC | SSIM global | SSIM gauss | J | время |", "|---|---|---|---|---|---|---|"]
        for name, cv in run["curves"].items():
            for k, e in cv.items():
                v = e["val"]
                L.append(f"| {name} | {k} | {v['pcc']['mean']:.3f} | {v['ssim_global']['mean']:.3f} | "
                         f"{v['ssim_gauss']['mean']:.3f} | {v['J']:.3f} | {_sec(e['sec_per_hologram'])} |")
        L += ["", "### По плоскостям (val, среднее по четвёркам)", "",
              "| метод | метрика | " + " | ".join(f"{z * 1e3:.1f} мм" for z in DEPTHS) + " |",
              "|---|---|" + "---|" * len(DEPTHS)]
        for name in dict.fromkeys([run["best"], "gs_100", "gs_weighted_500"]):
            if name in run["methods"]:
                for m in METRICS:
                    L.append(f"| {name} | {m} | " + " | ".join(
                        f"{x:.3f}" for x in run["methods"][name]["val"][m]["per_plane"]) + " |")
        L += ["", "### Полутоновые и простые изображения (val, по отдельным картинкам)", "",
              "Простое = >90 % пикселей около 0 или около максимума (глифы, фигуры); остальное — полутоновое.", "",
              "| метод | тип | число картинок | PCC | SSIM global | SSIM skimage | SSIM gauss | J |",
              "|---|---|---|---|---|---|---|---|"]
        for name in dict.fromkeys([run["best"], "adam_100", "gs_100", "gs_500"]):
            for kind, d in run["methods"].get(name, {}).get("val", {}).get("by_kind", {}).items():
                L.append(f"| {name} | {kind} | {d['n_images']} | {d['pcc']:.3f} | {d['ssim_global']:.3f} | "
                         f"{d['ssim_skimage']:.3f} | {d['ssim_gauss']:.3f} | {d['J']:.3f} |")
        L += ["", "### Интенсивность против амплитуды (val)", "",
              "Одни и те же фазы, оценённые по амплитуде (правило жюри) и по старой интенсивностной метрике.", "", HDR]
        for name, d in run["intensity_vs_amplitude"].items():
            L.append(_row(f"{name} — оценка по амплитуде", d["amplitude"]["val"]))
            L.append(_row(f"{name} — оценка по интенсивности", d["intensity"]["val"]))
        L += ["", "### Квантование фазы в 8 бит (val)", "", HDR]
        for name, d in run["quantisation"].items():
            L.append(_row(f"{name} непрерывная фаза", d["continuous"]["val"]))
            L.append(_row(f"{name} 8 бит", d["8bit"]["val"]))
        L += ["", "### Политика яркости сохранённого восстановления (val, 8 бит)", "", HDR]
        for name, d in run["policy"].items():
            for pol, e in d.items():
                L.append(_row(f"{name} [{pol}]", e["val"]))
        if run["timing_single"]:
            t = run["timing_single"]
            L += ["", f"### Время одной голограммы (B=1, CPU {t['threads']} потока, load avg {t['loadavg_1min']:.0f})",
                  ""] + [f"- {k}: {v:.2f} с" for k, v in t.items() if k not in ("threads", "loadavg_1min")]
        L.append("")
    if R["loss_study"]:
        ls = R["loss_study"]
        L += ["## Сравнение лоссов для Adam (амплитудные правила)", "",
              f"val — первые {ls['n_quads']} четвёрок, {ls['steps']} шагов, lr {ls['lr']}, старт `{ls['init']}`. "
              "Все лоссы принимают интенсивность и цель img/255; амплитудные считают X = sqrt(I), T = img/max(img). "
              "Веса в имени `wA_B_C` — A·(1-PCC(X,T)) + B·(1-SSIM_global(X̂,T)) + C·(1-SSIM_gauss11(X̂,T)); "
              "X̂ = X/max (peak) или X/soft-max (soft); `*_worst` — смешивание со «худшей плоскостью»; "
              "`jury_int` — старый интенсивностный победитель.", "", HDR]
        for k, v in sorted(ls["results"].items(), key=lambda kv: -kv[1]["val"]["J"]):
            L.append(_row(k, v["val"], _sec(v["sec_per_hologram"])))
        L.append("")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------ main

def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("main", "losses"), default="main")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--n-val", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--md-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.quick:
        OUT = OUT / "quick"
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "summary.json"
    R = load_summary(path)
    if not args.md_only:
        (stage_losses if args.stage == "losses" else stage_main)(R, args)
        path.write_text(json.dumps(R, indent=1))
    write_md(R)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
