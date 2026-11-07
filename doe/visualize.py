"""Figures for reports and the presentation (matplotlib, Agg backend; no torch). All labels are Russian.
КК = коэффициент корреляции Пирсона (PCC), ИСС = индекс структурного сходства (SSIM, формула слайда).

All functions take plain numpy arrays / dicts and write a PNG (or return a figure for the GIF).
Colours: neutral ink for text, one fixed categorical hue per image plane (never per rank).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import DEPTHS  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
METHOD_RU = {"nn_quality": "нейросеть + отбор ДОЭ", "nn": "нейросеть", "nn_refine": "нейросеть + уточнение", "gs": "Гершберг–Сакстон",
             "adam": "Adam", "backprop": "обратное распространение"}
PLANE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")   # plane 1..4, fixed order
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.grid": False, "font.size": 10, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def _img(ax, im, title=None, sub=None):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9, color=INK)
    if sub:
        ax.set_xlabel(sub, fontsize=8, color=INK_2)


def comparison_figure(targets, recons, planes, path, title=None, depths=DEPTHS):
    """Row of targets, row of reconstructions with per-plane PCC / SSIM labels."""
    d = len(targets)
    fig, axes = plt.subplots(2, d, figsize=(2.3 * d, 6.0))
    for i in range(d):
        _img(axes[0, i], targets[i], f"цель {i + 1} · z = {depths[i] * 1e3:.1f} мм")
        p = planes[i]
        _img(axes[1, i], recons[i], f"восстановление {i + 1} (амплитуда)",
             f"КК {_fmt(p['pcc'])} · ИСС {_fmt(p['ssim_global'])}\nИСС (окно 7×7) {_fmt(p['ssim_skimage'])}")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def method_grid(targets, rows, path, title=None, depths=DEPTHS):
    """rows: list of dicts {label, recon (D,N,N) uint8, planes, summary, time_ms}. First row = targets."""
    d = len(targets)
    n = len(rows) + 1
    fig, axes = plt.subplots(n, d + 1, figsize=(2.0 * d + 1.9, 2.15 * n + 0.3),
                             gridspec_kw={"width_ratios": [0.85] + [1] * d})
    axes = np.atleast_2d(axes)

    def label_cell(ax, lines):
        ax.axis("off")
        ax.text(0.0, 0.5, lines, transform=ax.transAxes, va="center", ha="left", fontsize=8.5, color=INK,
                linespacing=1.5)

    label_cell(axes[0, 0], "цели")
    for i in range(d):
        _img(axes[0, i + 1], targets[i], f"z = {depths[i] * 1e3:.1f} мм")
    for r, row in enumerate(rows, start=1):
        s = row["summary"]
        t = row.get("time_ms")
        t_txt = "" if t is None else (f"\n{t / 1000:.1f} с" if t >= 1000 else f"\n{t:.0f} мс")
        xt = row.get("crosstalk")
        x_txt = f"\nселект. {xt['selectivity']:.3f} ({xt['own_best']}/4)" if xt else ""
        label_cell(axes[r, 0], f"{row['label']}\nКК {_fmt(s['pcc']['mean'])}\n"
                               f"ИСС {_fmt(s['ssim_global']['mean'])}{x_txt}{t_txt}")
        for i in range(d):
            p = row["planes"][i]
            _img(axes[r, i + 1], row["recon"][i], None, f"{_fmt(p['pcc'], 2)} / {_fmt(p['ssim_global'], 2)}")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98 if title else 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def crosstalk_plot(z_mm, curves, path, depths=DEPTHS, ylabel="КК: цель i и амплитуда на расстоянии z", title=None):
    """curves: (D, Z) metric of target i vs reconstruction at z. Vertical rules at design planes."""
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for i, c in enumerate(curves):
        col = PLANE_COLORS[i % len(PLANE_COLORS)]
        ax.axvline(depths[i] * 1e3, color=col, linewidth=1, linestyle=(0, (3, 3)), alpha=0.7)
        ax.plot(z_mm, c, color=col, linewidth=2, label=f"цель {i + 1} ({depths[i] * 1e3:.1f} мм)")
        k = int(np.argmin(np.abs(np.asarray(z_mm) - depths[i] * 1e3)))   # value at its own design plane
        ax.plot(z_mm[k], c[k], "o", color=col, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{c[k]:.2f}", (z_mm[k], c[k]), textcoords="offset points", xytext=(14, 4), ha="left",
                    fontsize=8, color=INK_2)
    ax.set_xlabel("расстояние z, мм")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8, loc="lower right" if np.nanmin(curves) > 0.3 else "upper right")
    if title:
        ax.set_title(title, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _place_labels(fig, ax, items, fontsize=8.5):
    """Direct labels next to points; try a few offsets and keep the first that overlaps neither a placed
    label nor another marker (simple greedy placement, no extra dependencies)."""
    cands = [(9, -3, "left"), (9, 8, "left"), (9, -15, "left"), (-9, -3, "right"), (-9, 8, "right"),
             (-9, -15, "right"), (0, 12, "center"), (0, -20, "center")]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = [ax.transData.transform((x, y)) for x, y, _ in items]
    placed = []
    for (x, y, text), (px, py) in zip(items, pts):
        best = None
        for dx, dy, ha in cands:
            t = ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha,
                            fontsize=fontsize, color=INK)
            bb = t.get_window_extent(r).expanded(1.05, 1.15)
            hit = any(bb.overlaps(o) for o in placed) or any(
                bb.contains(qx, qy) or (abs(bb.x0 - qx) < 7 and bb.y0 - 7 < qy < bb.y1 + 7)
                for qx, qy in pts if (qx, qy) != (px, py))
            if not hit:
                best = t
                break
            t.remove()
        if best is None:
            dx, dy, ha = cands[0]
            best = ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha,
                               fontsize=fontsize, color=INK)
        placed.append(best.get_window_extent(r))


def speed_quality_plot(points, path, metrics=(("pcc_mean", "КК Пирсона, среднее по плоскостям"),
                                                    ("ssim_global_mean", "ИСС (формула слайда), среднее")),
                       title=None, note=None):
    """points: list of {label, time_ms, <metric keys>}. One panel per metric (small multiples, shared log-x),
    every point direct-labelled so identity is never colour-only."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.6 if note else 4.2), squeeze=False)
    ts = [p["time_ms"] for p in points]
    for ax, (key, lab) in zip(axes[0], metrics):
        ax.grid(color=GRID, linewidth=0.8)
        for k, p in enumerate(points):
            ax.plot(p["time_ms"], p[key], "o", color=SERIES[k % len(SERIES)], markersize=10,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(min(ts) / 3, max(ts) * 12)
        ax.set_xlabel("время синтеза одного ДОЭ, мс (лог.)")
        ax.set_ylabel(lab)
    if title:
        fig.suptitle(title, color=INK, fontsize=11)
    if note:
        fig.text(0.01, 0.01, note, fontsize=8, color=INK_2, va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0.09 if note else 0, 1, 1))
    for ax, (key, _) in zip(axes[0], metrics):       # after layout, so pixel positions are final
        _place_labels(fig, ax, [(p["time_ms"], p[key], p["label"]) for p in points])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def focus_frame(image_u8, z_mm, curve_z, curves, depths=DEPTHS, size_px=(640, 400)):
    """One GIF frame: reconstruction at z (left) + PCC-vs-z curves with a cursor (right). Returns RGB uint8."""
    w, h = size_px
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax_i = fig.add_axes([0.02, 0.10, 0.5, 0.80])
    _img(ax_i, image_u8, sub="амплитуда |K| / max |K|")
    near = int(np.argmin([abs(z_mm - d * 1e3) for d in depths]))
    in_focus = abs(z_mm - depths[near] * 1e3) < 0.08
    ax_i.set_title(f"z = {z_mm:5.2f} мм" + (f"   ◀ плоскость {near + 1}" if in_focus else ""),
                   fontsize=11, color=PLANE_COLORS[near] if in_focus else INK, loc="left")
    ax_c = fig.add_axes([0.60, 0.18, 0.37, 0.70])
    for i, c in enumerate(curves):
        ax_c.plot(curve_z, c, color=PLANE_COLORS[i], linewidth=1.6, label=f"цель {i + 1}")
    ax_c.axvline(z_mm, color=INK, linewidth=1)
    ax_c.set_xlabel("z, мм", fontsize=8)
    ax_c.set_ylabel("КК с целью", fontsize=8)
    ax_c.tick_params(labelsize=7)
    ax_c.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    ax_c.set_ylim(min(-0.05, float(np.nanmin(curves))), 1.15)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return rgb


def depth_scan(phase_u8, targets_u8, z_m, policy="peak", quantity="amplitude"):
    """Reconstruct the saved 8-bit phase at every z in z_m (amplitude |K| by default, max -> 255) and score
    each target against it with the official rule (both images divided by their max, КК on [0, 1]).

    Returns (recon (Z,N,N) uint8, pcc (D,Z)). Imports physics lazily so plotting alone needs no torch.
    """
    from .metrics import normalize_max, pcc, reconstruct_u8

    pol = "peak" if policy == "fit" else policy
    recon = reconstruct_u8(phase_u8, list(z_m), pol, None, quantity)
    tn = [normalize_max(t) for t in targets_u8]
    rn = [normalize_max(r) for r in recon]
    curves = np.array([[np.nan if (v := pcc(t, r)) is None else v for r in rn] for t in tn])
    return recon, curves


def load_run(run_dir=None, doe=None, targets=None):
    """(phase_u8, targets_u8, label) from a synth.py output dir or explicit paths."""
    import json
    from pathlib import Path

    from .io import load_gray

    if run_dir:
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "metrics.json").read_text())
        doe = run_dir / meta.get("doe", "doe.png")
        targets = targets or meta["inputs"]
        label = meta.get("method", "")
    else:
        label = ""
    return load_gray(doe, strict=True), np.stack([load_gray(p) for p in targets]), label
