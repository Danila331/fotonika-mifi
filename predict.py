"""Точка входа решения: 4 изображения одного комплекта → один фазовый ДОЭ (PNG 128×128, 8 бит, градации серого).

    python predict.py img1.png img2.png img3.png img4.png --out out        # быстрый нейросетевой режим
    python predict.py --in-dir in --out out                                # 4 PNG из папки, по алфавиту
    python predict.py --in-dir in --out out --refine                       # сеть + 100 шагов уточнения

Порядок изображений = порядок расстояний 24.1 / 25.3 / 26.5 / 27.7 мм. Кодировка фазы: пиксель 0 → −π, 255 → +π.
Результат: out/doe.png (ДОЭ), out/recon_1..4.png (восстановления Френелем, амплитуда |K|), out/metrics.json.
"""
import argparse
import os
import sys
from pathlib import Path

import synth


def default_method():
    """Docker defaults to fastgs; the native CLI keeps the original single-pass default."""
    return "nn_fastgs" if os.environ.get("DOE_DEFAULT_MODE") == "fastgs" else "nn"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="4 PNG (128×128, 8 бит, 1 канал) в порядке плоскостей")
    ap.add_argument("--in-dir", help="папка с ровно четырьмя PNG (берутся по алфавиту)")
    ap.add_argument("--out", default="out", help="папка результатов (по умолчанию out)")
    ap.add_argument("--fast", action="store_true", help="только сеть, один проход (режим по умолчанию)")
    ap.add_argument("--fastgs", action="store_true", help="компактная нейросеть + GS (по умолчанию 100 итераций; основной Docker-режим)")
    ap.add_argument("--refine", action="store_true", help="сеть + 100 шагов уточнения фазы")
    ap.add_argument("--quality", action="store_true", help="качество: исходные 100 шагов + отбор улучшенных ДОЭ по метрикам")
    ap.add_argument("--visual", action="store_true", help="визуальное качество: multiscale L1 + excess-TV после базовых 100 шагов")
    ap.add_argument("--checkpoint", help="путь к альтернативным весам модели")
    ap.add_argument("--steps", type=int, help="число GS-итераций для --fastgs (по умолчанию 100) или дополнительные шаги для других режимов")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"),
                    help="auto: выбрать доступный GPU (по умолчанию); в Docker используется CPU")
    ap.add_argument("--figure", action="store_true", help="дополнительно нарисовать comparison.png")
    a = ap.parse_args(argv)
    if sum((a.fast, a.fastgs, a.refine, a.quality, a.visual)) > 1:
        ap.error("--fast, --fastgs, --refine, --quality и --visual нельзя использовать вместе")
    if a.steps is not None and not (a.fastgs or a.refine or a.quality or a.visual):
        ap.error("--steps требует --fastgs, --refine, --quality или --visual")
    if a.in_dir:
        files = sorted(p for p in Path(a.in_dir).iterdir() if p.suffix.lower() == ".png")
        inputs = [str(p) for p in files]
    else:
        inputs = a.images
    if len(inputs) != 4:
        ap.error(f"нужно ровно 4 изображения, получено {len(inputs)}")
    if a.fastgs:
        method = "nn_fastgs"
    elif a.visual:
        method = "nn_visual"
    elif a.quality:
        method = "nn_quality"
    elif a.refine:
        method = "nn_refine"
    elif a.fast:
        method = "nn"
    else:
        method = default_method()
    synth.run(inputs, a.out, method, a.steps, "peak", False, None if a.device == "auto" else a.device,
              0, a.checkpoint, figure=a.figure)
    print(f"ДОЭ: {Path(a.out) / 'doe.png'}")


if __name__ == "__main__":
    sys.exit(main())
