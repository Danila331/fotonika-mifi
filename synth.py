#!/usr/bin/env python
"""Синтез фазового ДОЭ для 4 изображений и расчёт метрик кейса.

    python synth.py img1.png img2.png img3.png img4.png --out OUTDIR
                    [--method nn|nn_refine|nn_quality|gs|adam|backprop] [--steps K] [--policy peak|minmax|fit]
                    [--quantity amplitude|intensity] [--strict] [--device cpu|mps|cuda] [--seed S]

Изображение i восстанавливается на z = 24.1 / 25.3 / 26.5 / 27.7 мм. В OUTDIR записываются:
  doe.png             фаза: 128×128, 8 бит, 1 канал; 0 -> −π, 255 -> +π
  recon_1..4.png      восстановленные амплитуды |K| (128×128, 8 бит), посчитанные из ЗАПИСАННОГО doe.png
  metrics.json/.csv   КК Пирсона и ИСС (формула слайда + оконные варианты) по плоскостям, среднее/худшее, время
  comparison.png      исходные изображения и восстановления с метриками
Официальный протокол экспертов: сравниваются амплитуды |K| (E1), каждое изображение нормируется на
свой максимум = 1 (E2). Все числа считает doe.metrics.evaluate по перечитанному 8-битному doe.png,
поэтому `python evaluate.py OUTDIR/doe.png img1..4.png` воспроизводит их независимо.
"""

import argparse
import csv
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from doe.config import DEPTHS, N
from doe.io import load_gray, phase_to_u8, save_gray
from doe.metrics import crosstalk, evaluate

REFINE_ALPHA, REFINE_BETA = 1.0, 0.5  # вес оконного / глобального ИСС при уточнении (SSIM — основная метрика сдачи)
REFINE_XTALK = 0.25  # вес штрафа за помехи при уточнении (при обучении 1.0): чище картинка, разделение чуть слабее
METHODS = ("nn", "nn_fastgs", "nn_refine", "nn_quality", "nn_visual", "gs", "adam", "backprop")
DEFAULT_STEPS = {"nn_fastgs": 100, "nn_quality": 300, "nn_visual": 300, "nn_refine": 100, "gs": 200, "adam": 500, "backprop": 0, "nn": 0}
METRICS = ("pcc", "ssim_global", "ssim_skimage", "ssim_gauss")
METHOD_RU = {"nn_quality": "Нейросеть + отбор улучшенного ДОЭ", "nn_visual": "Нейросеть + визуальное уточнение",
             "nn_fastgs": "Нейросеть + быстрый GS", "nn": "Нейросеть", "nn_refine": "Нейросеть + уточнение", "gs": "Гершберг–Сакстон",
             "adam": "Adam", "backprop": "Обратное распространение"}


# ------------------------------------------------------------------ methods
def backprop_phase(targets_u8, seed=0):
    """Trivial non-iterative baseline (= GS iteration 0): back-propagate sqrt(target)·random phase
    from every plane to the DOE, sum the fields, keep the argument."""
    from doe.physics import backpropagate_np

    rng = np.random.default_rng(seed)
    field = np.zeros((N, N), complex)
    for t, z in zip(targets_u8, DEPTHS):
        amp = np.sqrt(np.asarray(t, np.float64) / 255.0)
        field += backpropagate_np(amp * np.exp(1j * rng.uniform(-np.pi, np.pi, amp.shape)), z)
    return np.angle(field)


class Synthesiser:
    """Loads the method once (model weights, imports) so repeated calls are timed fairly."""

    def __init__(self, method, steps=None, device=None, seed=0, checkpoint=None, refine_xtalk=REFINE_XTALK):
        if method not in METHODS:
            raise ValueError(f"unknown method {method}")
        self.method = method
        self.steps = DEFAULT_STEPS[method] if steps is None else steps
        self.refine_xtalk = refine_xtalk
        self.device = device
        self.seed = seed
        self.info = {}
        t0 = time.perf_counter()
        if method == "nn_fastgs" and checkpoint is None:
            checkpoint = Path(__file__).resolve().parent / "checkpoints" / "stage0-finetuned.pt"
        if method in ("nn", "nn_fastgs", "nn_refine", "nn_quality", "nn_visual"):
            try:
                from doe.infer import NeuralDOE
            except ImportError as e:
                raise SystemExit(f"метод {method} требует doe/infer.py (пока недоступен: {e}); "
                                 f"используйте --method gs/adam/backprop") from e
            self.model = NeuralDOE.load(path=checkpoint, device=device)
            meta = getattr(self.model, "meta", None) or {}
            ck = meta.get("path") or checkpoint
            if ck:
                self.info["checkpoint"] = str(Path(ck).resolve().relative_to(Path(__file__).resolve().parent)
                                              if Path(ck).resolve().is_relative_to(Path(__file__).resolve().parent)
                                              else ck)
            self.info["model_device"] = str(getattr(self.model, "device", device))
            if method == "nn_fastgs":
                import torch
                self.model.model = self.model.model.to(memory_format=torch.channels_last)
                self.info["inference_layout"] = "channels_last"
        elif method in ("gs", "adam"):
            try:
                from doe import baselines
            except ImportError as e:
                raise SystemExit(f"метод {method} требует doe/baselines.py (пока недоступен: {e}); "
                                 f"используйте --method backprop") from e
            self.baselines = baselines
        self.load_ms = (time.perf_counter() - t0) * 1e3

    def __call__(self, targets_u8):
        m = self.method
        if m == "nn":
            return self.model.predict(targets_u8)
        if m == "nn_fastgs":
            import torch
            from doe.baselines import gs_batch
            x = self.model._to_tensor(targets_u8[None]).to(memory_format=torch.channels_last)
            with torch.no_grad():
                initial = self.model.model(x).float().cpu().numpy()
            self.info["fastgs"] = {"teacher": "stage0_finetuned", "steps": self.steps,
                                   "note": "neural initialization plus multi-plane GS refinement"}
            return gs_batch(targets_u8[None], iters=self.steps, init=initial, device=self.model.device)[0]
        if m == "nn_quality":
            from doe.quality import QualityConfig, refine_quality
            cfg = QualityConfig(steps=self.steps, xtalk=self.refine_xtalk)
            phases, choices = refine_quality(self.model, targets_u8[None], cfg)
            from dataclasses import asdict
            self.info["quality"] = {"config": asdict(cfg), "baseline_steps": 100, "baseline_xtalk": 0.25, "selection": choices[0],
                                    "score_note": "engineering proxy, not official jury score"}
            return phases[0]
        if m == "nn_visual":
            from doe.perceptual import refine_perceptual
            baseline = self.model.predict_refined_batch(targets_u8[None], steps=100, xtalk=0.25,
                                                        alpha=REFINE_ALPHA, beta=REFINE_BETA)
            phases, choices = refine_perceptual(self.model, targets_u8[None], baseline, steps=self.steps)
            self.info["visual"] = {"baseline_steps": 100, "additional_steps": self.steps,
                                   "ms_l1_weight": 0.3, "excess_tv_weight": 0.1,
                                   "selection": choices[0], "score_note": "engineering proxy, not official jury score"}
            return phases[0]
        if m == "nn_refine":
            return self.model.predict_refined(targets_u8, steps=self.steps, xtalk=self.refine_xtalk,
                                                alpha=REFINE_ALPHA, beta=REFINE_BETA)
        kw = {"device": self.device} if self.device else {}
        if m == "gs":
            return self.baselines.gs(targets_u8, iters=self.steps, seed=self.seed, **kw)
        if m == "adam":
            return self.baselines.adam(targets_u8, steps=self.steps, seed=self.seed, **kw)
        return backprop_phase(targets_u8, self.seed)

    def timed(self, targets_u8):
        t0 = time.perf_counter()
        phase = self(targets_u8)
        _sync(self.device)
        ms = (time.perf_counter() - t0) * 1e3
        phase = np.asarray(phase.detach().cpu().numpy() if hasattr(phase, "detach") else phase, np.float64)
        phase = np.squeeze(phase)
        if phase.shape != (N, N) or not np.isfinite(phase).all():
            raise RuntimeError(f"{self.method} вернул некорректную фазу {phase.shape}")
        return phase, ms


def _sync(device):
    if device and str(device).startswith(("mps", "cuda")):
        import torch
        getattr(torch, str(device).split(":")[0]).synchronize()


# ------------------------------------------------------------------ output checks
def check_png(path, expect=None):
    """Re-open a written PNG and assert the case format: mode L (8-bit, 1 channel), 128x128."""
    with Image.open(path) as img:
        if img.format != "PNG" or img.mode != "L" or img.size != (N, N):
            raise RuntimeError(f"{path}: формат {img.format}, mode {img.mode}, размер {img.size} — не соответствует требованиям")
        a = np.asarray(img)
    if a.dtype != np.uint8:
        raise RuntimeError(f"{path}: dtype {a.dtype}")
    if expect is not None and not np.array_equal(a, expect):
        raise RuntimeError(f"{path}: содержимое не совпадает с записанным массивом")
    return a


def write_csv(path, rep):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plane", "depth_mm", *METRICS])
        for i, p in enumerate(rep["planes"]):
            w.writerow([i + 1, p["depth_mm"], *("" if p[k] is None else f"{p[k]:.6f}" for k in METRICS)])
        for agg in ("mean", "worst"):
            w.writerow([agg, "", *("" if rep["summary"][k][agg] is None else f"{rep['summary'][k][agg]:.6f}"
                                   for k in METRICS)])


def _versions():
    import importlib.metadata as md
    v = {"python": platform.python_version()}
    for p in ("numpy", "scikit-image", "pillow", "torch"):
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return v


# ------------------------------------------------------------------ pipeline
def run(inputs, out, method="nn", steps=None, policy="peak", strict=False, device=None, seed=0,
        checkpoint=None, figure=True, synthesiser=None, quiet=False, quantity="amplitude", refine_xtalk=REFINE_XTALK):
    """Full pipeline; returns the metrics dict (also written to out/metrics.json)."""
    effective_method = synthesiser.method if synthesiser is not None else method
    if effective_method in ("nn_fastgs", "nn_quality", "nn_visual") and (policy != "peak" or quantity != "amplitude"):
        raise ValueError(f"{effective_method} supports only amplitude with peak normalization")
    t_start = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        targets = np.stack([load_gray(p, strict=strict) for p in inputs])
    input_warnings = [str(w.message) for w in caught]
    for w in input_warnings:
        print(f"ПРЕДУПРЕЖДЕНИЕ (вход не по формату, сконвертирован): {w}", file=sys.stderr)

    syn = synthesiser or Synthesiser(method, steps, device, seed, checkpoint, refine_xtalk)
    phase, synth_ms = syn.timed(targets)
    timings = {"load_ms": syn.load_ms, "synth_ms": synth_ms}

    doe_path = out / "doe.png"
    save_gray(doe_path, phase_to_u8(phase))
    phase_u8 = check_png(doe_path)                          # everything below uses the saved file

    t0 = time.perf_counter()
    rep = evaluate(phase_u8, targets, policy=policy, quantity=quantity)
    timings["eval_ms"] = (time.perf_counter() - t0) * 1e3
    recon = rep.pop("recon")
    recon_paths = []
    for i, r in enumerate(recon):
        p = out / f"recon_{i + 1}.png"
        save_gray(p, r)
        check_png(p, expect=r)
        recon_paths.append(p.name)

    xt = crosstalk(targets, recon)       # depth separation, from the saved recon PNGs

    # the other variants, for the record only (official = amplitude + peak)
    variants = {}
    for q in ("amplitude", "intensity"):
        for pol in ("peak", "minmax", "fit"):
            variants[f"{q}/{pol}"] = (rep["summary"] if (q, pol) == (quantity, policy)
                                      else evaluate(phase_u8, targets, policy=pol, quantity=q)["summary"])

    timings["total_ms"] = (time.perf_counter() - t_start) * 1e3
    result = {
        "method": syn.method, "steps": syn.steps, "refine_xtalk": syn.refine_xtalk if syn.method in ("nn_refine", "nn_quality", "nn_visual") else None, "quantity": quantity, "policy": policy,
        "protocol": "E1: сравниваются амплитуды |K|; E2: исходное и восстановленное изображения делятся "
                    "на свой максимум (= 1), метрики в [0, 1], ИСС с L = 1; recon_i.png = round(255·X/max X)",
        "inputs": [str(p) for p in inputs], "input_warnings": input_warnings,
        "doe": doe_path.name, "recon": recon_paths,
        "depths_mm": [round(z * 1e3, 3) for z in DEPTHS],
        "planes": rep["planes"], "summary": rep["summary"],
        "crosstalk": {"selectivity": xt["selectivity"], "own_best": xt["own_best"], "diag": xt["diag"],
                      "off": xt["off"], "matrix_target_vs_plane": xt["matrix"],
                      "note": "КК(цель i, восстановление в плоскости j); селективность = среднее диаг. − "
                              "среднее вне диаг.; own_best — в скольких плоскостях своя цель похожа больше всех"},
        "summary_variants": variants,
        "timings_ms": {k: round(v, 3) for k, v in timings.items()},
        "device": device, **syn.info,
        "format": {"doe": "PNG L, 8 бит, 128×128, 0 = −π, 255 = +π", "recon": "PNG L, 8 бит, 128×128",
                   "checked": True},
        "versions": _versions(),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    write_csv(out / "metrics.csv", rep)
    if figure:
        from doe.visualize import comparison_figure
        s = rep["summary"]
        comparison_figure(targets, recon, rep["planes"], out / "comparison.png",
                          title=f"{METHOD_RU[syn.method]} · КК {s['pcc']['mean']:.3f} · "
                                f"ИСС {s['ssim_global']['mean']:.3f} · селективность {xt['selectivity']:.3f} "
                                f"({xt['own_best']}/4) · {synth_ms:.0f} мс")
    if not quiet:
        from evaluate import print_table
        print_table(rep)
        print(f"селективность по глубине {xt['selectivity']:.3f}, своя цель лучшая в {xt['own_best']}/4 плоскостях")
        print(f"метод {syn.method}: синтез {synth_ms:.1f} мс"
              + f", загрузка {syn.load_ms:.0f} мс, всего {timings['total_ms']:.0f} мс -> {out}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs=4, help="4 исходных PNG (128×128, 8 бит, 1 канал) для плоскостей "
                                            "24.1/25.3/26.5/27.7 мм, по порядку")
    ap.add_argument("--out", required=True, help="папка для результатов")
    ap.add_argument("--method", default="nn", choices=METHODS,
                    help="метод синтеза: nn — один проход сети (по умолчанию), "
                         "nn_fastgs — компактная сеть + 100 итераций GS (основной Docker-режим), "
                         "nn_refine — сеть + 100 шагов, nn_quality — дополнительные шаги с отбором, "
                         "nn_visual — multiscale/TV уточнение для визуального качества, "
                         "gs / adam / backprop — базовые методы")
    ap.add_argument("--steps", type=int, help="число итераций для gs / adam / nn_refine "
                                              f"(по умолчанию {DEFAULT_STEPS['gs']}/{DEFAULT_STEPS['adam']}/"
                                              f"{DEFAULT_STEPS['nn_refine']})")
    ap.add_argument("--refine-xtalk", type=float, default=REFINE_XTALK,
                    help="вес штрафа за помехи при уточнении nn_refine (обучение шло с 1.0; по умолчанию "
                         f"{REFINE_XTALK}: картинка чище, разделение по глубине чуть слабее)")
    ap.add_argument("--quantity", default="amplitude", choices=("amplitude", "intensity"),
                    help="что восстанавливается и сравнивается: amplitude |K| (официально) или intensity |K|²")
    ap.add_argument("--policy", default="peak", choices=("peak", "minmax", "fit"),
                    help="перевод восстановления в 8 бит (по умолчанию peak = X/max·255)")
    ap.add_argument("--strict", action="store_true", help="отклонять входы не 128×128 / не 8 бит / не 1 канал")
    ap.add_argument("--device", help="устройство torch для nn / adam (по умолчанию автоматически)")
    ap.add_argument("--checkpoint", help="веса сети (по умолчанию checkpoints/best.pt)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figure", action="store_true", help="не рисовать comparison.png")
    a = ap.parse_args(argv)
    try:
        run(a.inputs, a.out, a.method, a.steps, a.policy, a.strict, a.device, a.seed, a.checkpoint,
            figure=not a.no_figure, quantity=a.quantity, refine_xtalk=a.refine_xtalk)
    except ValueError as e:     # --strict format violations
        raise SystemExit(f"ошибка: {e}")


if __name__ == "__main__":
    main()
