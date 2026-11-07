#!/usr/bin/env python
"""Независимая проверка: PNG фазы ДОЭ + 4 PNG исходных изображений -> 4 восстановления + КК Пирсона и ИСС.

Файл самодостаточен: только numpy, Pillow и scikit-image, без torch и без остального кода проекта,
чтобы эксперт мог прочитать каждую строку и воспроизвести наши числа. Численно совпадает с
doe.metrics.evaluate бит-в-бит (это проверяет tests/test_cli.py).

    python evaluate.py doe.png t1.png t2.png t3.png t4.png [--policy peak|minmax|fit|all]
                       [--quantity amplitude|intensity] [--json out.json] [--save-recon DIR] [--check-recon DIR]

Физика (слайд 2 кейса), однократное БПФ Френеля, λ = 532 нм, пиксель 10 мкм, N = 128:
    K(x,y) = exp(jkz)/(jλz) · exp(jπ(x²+y²)/λz) · FFT[ H(u,v) · exp(jπ(u²+v²)/λz) ],  H = exp(jφ)
Множитель и выходной чирп имеют единичный модуль, поэтому |K| ∝ |FFT[H · exp(jπ(u²+v²)/λz)]|
(центрированное унитарное БПФ, u = (n - N/2)·dx). Кодек фазы (слайд 6): 0 -> -π, 255 -> +π, линейно.

Официальный протокол (ответы экспертов E1, E2):
  E1  сравниваются АМПЛИТУДЫ |K| (--quantity amplitude, по умолчанию; intensity |K|² — только для справки);
  E2  каждое изображение (и исходное, и восстановленное) нормируется на свой максимум = 1.
Восстановленный PNG = round(255·|K|/max|K|) (политика peak). Метрики считаются по изображениям в [0, 1]:
КК Пирсона и ИСС (SSIM) с L = 1 в трёх вариантах: ssim_global — формула слайда (глобальная, без окна),
ssim_skimage — skimage по умолчанию (окно 7×7), ssim_gauss — Wang и др. (гауссово окно 11×11, σ = 1.5).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

N = 128
DX = 10e-6
WAVELENGTH = 532e-9
DEPTHS = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)
POLICIES = ("peak", "minmax", "fit")
METRICS = ("pcc", "ssim_global", "ssim_skimage", "ssim_gauss")


# ---------------------------------------------------------------- I/O
def read_png(path, strict=False):
    img = Image.open(path)
    ok = img.mode == "L" and img.size == (N, N)
    if not ok:
        msg = f"{path}: mode={img.mode}, размер {img.size}; ожидается 8 бит, 1 канал (mode L), {N}×{N}"
        if strict:
            raise SystemExit(msg)
        print("ПРЕДУПРЕЖДЕНИЕ: " + msg + " (сконвертировано)", file=sys.stderr)
        # same conversion as doe.io.load_gray: 16-bit rescaled, alpha composited on black
        if img.mode in ("I;16", "I;16B", "I;16L", "I"):
            a = np.asarray(img, dtype=np.float64)
            a = a / 257.0 if a.max() > 255 else a
            img = Image.fromarray(np.clip(np.round(a), 0, 255).astype(np.uint8))
        elif img.mode != "L":
            if img.mode in ("RGBA", "LA", "P", "PA"):
                img = Image.alpha_composite(Image.new("RGBA", img.size, (0, 0, 0, 255)), img.convert("RGBA"))
            img = img.convert("L")
        if img.size != (N, N):
            img = img.resize((N, N), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8).copy()


def u8_to_phase(q):
    return np.asarray(q, dtype=np.float64) / 255.0 * 2 * np.pi - np.pi


# ---------------------------------------------------------------- physics
def chirp(z, n=N, dx=DX, wavelength=WAVELENGTH):
    u = (np.arange(n) - n // 2) * dx
    r2 = u[:, None] ** 2 + u[None, :] ** 2
    return np.exp(1j * np.pi / (wavelength * z) * r2)


def fftc(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def intensity(phase, depths=DEPTHS):
    h = np.exp(1j * np.asarray(phase, dtype=np.float64))
    return np.abs(np.stack([fftc(h * chirp(z, h.shape[-1])) for z in depths])) ** 2


# ---------------------------------------------------------------- metrics
def to_uint8(i, policy="peak", target=None):
    i = np.asarray(i, dtype=np.float64)
    if policy == "peak":
        m = i.max()
        out = i / m * 255.0 if m > 0 else np.zeros_like(i)
    elif policy == "minmax":
        lo, hi = i.min(), i.max()
        out = (i - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(i)
    elif policy == "fit":
        t = np.asarray(target, dtype=np.float64)
        d = (i * i).sum()
        out = i * ((i * t).sum() / d) if d > 0 else np.zeros_like(i)
    else:
        raise ValueError(policy)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else None


def ssim_global(a, b, data_range=255.0):
    """Формула слайда: глобальные средние, дисперсии и ковариация, c = (k·L)²."""
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    return float((2 * ma * mb + c1) * (2 * cov + c2) / ((ma**2 + mb**2 + c1) * (va + vb + c2)))


def ssim_skimage(a, b, data_range=255.0):
    return float(structural_similarity(np.asarray(a, np.float64), np.asarray(b, np.float64), data_range=data_range))


def ssim_gauss(a, b, data_range=255.0):
    return float(structural_similarity(np.asarray(a, np.float64), np.asarray(b, np.float64), data_range=data_range,
                                       gaussian_weights=True, sigma=1.5, use_sample_covariance=False))


def normalize_max(x):
    """E2: деление на максимум изображения (полностью чёрное изображение остаётся нулевым)."""
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    return x / m if m > 0 else x


def evaluate(phase_u8, targets_u8, policy="peak", quantity="amplitude"):
    inten = intensity(u8_to_phase(phase_u8))
    if quantity == "amplitude":
        x = np.sqrt(inten)
    elif quantity == "intensity":
        x = inten
    else:
        raise ValueError(quantity)
    recon = np.stack([to_uint8(x[i], policy, targets_u8[i]) for i in range(len(DEPTHS))])
    planes = []
    for i, z in enumerate(DEPTHS):
        t, r = normalize_max(targets_u8[i]), normalize_max(recon[i])
        planes.append({"depth_mm": round(z * 1e3, 3), "pcc": pcc(t, r), "ssim_global": ssim_global(t, r, 1.0),
                       "ssim_skimage": ssim_skimage(t, r, 1.0), "ssim_gauss": ssim_gauss(t, r, 1.0)})
    summary = {}
    for k in METRICS:
        vals = [p[k] for p in planes if p[k] is not None]
        summary[k] = {"mean": float(np.mean(vals)) if vals else None,
                      "worst": float(np.min(vals)) if vals else None, "valid": len(vals)}
    return {"quantity": quantity, "policy": policy, "planes": planes, "summary": summary}, recon


def crosstalk(targets_u8, recon_u8):
    """Разделение по глубине. M[i, j] = КК(цель i, восстановление в плоскости j), оба нормированы на max.
    selectivity = среднее диагонали − среднее вне диагонали; own_best — число плоскостей, где восстановление
    больше всего похоже именно на свою цель. ДОЭ, рисующий одну смесь во всех плоскостях, даёт ~0."""
    d = len(targets_u8)
    m = np.full((d, d), np.nan)
    for i in range(d):
        for j in range(d):
            v = pcc(normalize_max(targets_u8[i]), normalize_max(recon_u8[j]))
            m[i, j] = np.nan if v is None else v
    diag = np.nanmean(np.diag(m)) if np.isfinite(np.diag(m)).any() else np.nan
    off = m[~np.eye(d, dtype=bool)]
    offm = np.nanmean(off) if np.isfinite(off).any() else np.nan
    own = int(sum(np.nanargmax(np.nan_to_num(m[:, j], nan=-9)) == j for j in range(d)))
    return {"matrix": m.tolist(), "diag": float(diag), "off": float(offm),
            "selectivity": float(diag - offm), "own_best": own}


def _fmt(v):
    return "   н/д" if v is None else f"{v:6.4f}"


QUANTITY_RU = {"amplitude": "амплитуда |K|", "intensity": "интенсивность |K|²"}


def print_table(rep, out=sys.stdout):
    print(f"величина: {QUANTITY_RU[rep.get('quantity', 'amplitude')]} · нормировка PNG: {rep['policy']} · "
          f"перед метриками каждое изображение делится на свой максимум", file=out)
    print(f"{'пл.':>5} {'z, мм':>6} | {'КК':>6} {'ИСС':>6} {'ИССsk':>6} {'ИССg':>6}", file=out)
    print("-" * 46, file=out)
    for i, p in enumerate(rep["planes"]):
        print(f"{i + 1:>5} {p['depth_mm']:>6.1f} | " + " ".join(_fmt(p[k]) for k in METRICS), file=out)
    print("-" * 46, file=out)
    for agg, ru in (("mean", "среднее"), ("worst", "худшее")):
        print(f"{ru:>12} | " + " ".join(_fmt(rep["summary"][k][agg]) for k in METRICS), file=out)
    print("(КК — коэффициент корреляции Пирсона; ИСС — формула слайда, глобальная; "
          "ИССsk — skimage, окно 7×7; ИССg — гауссово окно 11×11)", file=out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("doe", help="PNG фазы ДОЭ, 128×128, 8 бит (0 = −π, 255 = +π)")
    ap.add_argument("targets", nargs=4, help="4 исходных PNG для плоскостей 24.1/25.3/26.5/27.7 мм, по порядку")
    ap.add_argument("--policy", default="peak", choices=POLICIES + ("all",),
                    help="перевод восстановления в 8 бит (по умолчанию peak = X/max·255)")
    ap.add_argument("--quantity", default="amplitude", choices=("amplitude", "intensity"),
                    help="что сравнивать: amplitude |K| (официально, E1) или intensity |K|² (для справки)")
    ap.add_argument("--json", help="куда записать JSON (по умолчанию evaluation.json рядом с ДОЭ)")
    ap.add_argument("--save-recon", metavar="DIR", help="записать recon_1..4.png (первая политика) в DIR")
    ap.add_argument("--check-recon", metavar="DIR",
                    help="проверить, что DIR/recon_1..4.png бит-в-бит совпадают с восстановлением из ДОЭ")
    ap.add_argument("--strict", action="store_true", help="отклонять PNG, не являющиеся 128×128 mode L")
    a = ap.parse_args(argv)

    phase_u8 = read_png(a.doe, a.strict)
    targets = np.stack([read_png(p, a.strict) for p in a.targets])
    policies = POLICIES if a.policy == "all" else (a.policy,)

    reports, recons = [], []
    for pol in policies:
        rep, recon = evaluate(phase_u8, targets, pol, a.quantity)
        rep["crosstalk"] = crosstalk(targets, recon)
        reports.append(rep)
        recons.append(recon)
        print_table(rep)
        xt = rep["crosstalk"]
        print("КК(цель i, плоскость j):")
        for i, row in enumerate(xt["matrix"]):
            print(f"  цель {i + 1}: " + " ".join(f"{v:6.3f}" for v in row))
        print(f"селективность по глубине {xt['selectivity']:.4f} (диаг. {xt['diag']:.4f} − вне диаг. "
              f"{xt['off']:.4f}); своя цель лучшая в {xt['own_best']}/4 плоскостях")
        print()

    result = {"doe": str(a.doe), "targets": [str(p) for p in a.targets],
              "depths_mm": [round(z * 1e3, 3) for z in DEPTHS], "quantity": a.quantity,
              "normalisation": "каждое изображение делится на свой максимум (E2), метрики в [0, 1], L = 1",
              "evaluations": reports}
    if a.save_recon:
        d = Path(a.save_recon)
        d.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(recons[0]):
            Image.fromarray(r).save(d / f"recon_{i + 1}.png")
    status = 0
    if a.check_recon:
        checks = []
        for i in range(len(DEPTHS)):
            got = read_png(Path(a.check_recon) / f"recon_{i + 1}.png", strict=True)
            checks.append(int(np.abs(got.astype(int) - recons[0][i].astype(int)).max()))
        result["check_recon_max_abs_diff"] = checks
        ok = all(c == 0 for c in checks)
        print(f"проверка recon_*.png ({a.quantity}, {policies[0]}): "
              + ("совпадают бит-в-бит" if ok else f"РАСХОЖДЕНИЕ, max|разность| = {checks}"))
        status = 0 if ok else 1
    out = Path(a.json) if a.json else Path(a.doe).with_name("evaluation.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"JSON -> {out}")
    return status


if __name__ == "__main__":
    sys.exit(main())
