"""Точка входа решения: 4 изображения одного комплекта → один фазовый ДОЭ (PNG 128×128, 8 бит, градации серого).

    python predict.py img1.png img2.png img3.png img4.png --out out        # быстрый нейросетевой режим
    python predict.py --in-dir in --out out                                # 4 PNG из папки, по алфавиту
    python predict.py --in-dir in --out out --refine                       # сеть + 100 шагов уточнения

Порядок изображений = порядок расстояний 24.1 / 25.3 / 26.5 / 27.7 мм. Кодировка фазы: пиксель 0 → −π, 255 → +π.
Результат: out/doe.png (ДОЭ), out/recon_1..4.png (восстановления Френелем, амплитуда |K|), out/metrics.json.
"""
import argparse
import sys
from pathlib import Path

import synth


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="4 PNG (128×128, 8 бит, 1 канал) в порядке плоскостей")
    ap.add_argument("--in-dir", help="папка с ровно четырьмя PNG (берутся по алфавиту)")
    ap.add_argument("--out", default="out", help="папка результатов (по умолчанию out)")
    ap.add_argument("--fast", action="store_true", help="только сеть, один проход (режим по умолчанию)")
    ap.add_argument("--refine", action="store_true", help="сеть + 100 шагов уточнения фазы")
    ap.add_argument("--quality", action="store_true", help="качество: исходные 100 шагов + отбор улучшенных ДОЭ по метрикам")
    ap.add_argument("--checkpoint", help="путь к альтернативным весам модели")
    ap.add_argument("--steps", type=int, help="шаги уточнения: обычно 100; с --quality дополнительные шаги после базовых 100 (по умолчанию 300)")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"),
                    help="auto: выбрать доступный GPU (по умолчанию); в Docker используется CPU")
    ap.add_argument("--figure", action="store_true", help="дополнительно нарисовать comparison.png")
    a = ap.parse_args(argv)
    if sum((a.fast, a.refine, a.quality)) > 1:
        ap.error("--fast, --refine и --quality нельзя использовать вместе")
    if a.steps is not None and not (a.refine or a.quality):
        ap.error("--steps требует --refine или --quality")
    if a.in_dir:
        files = sorted(p for p in Path(a.in_dir).iterdir() if p.suffix.lower() == ".png")
        inputs = [str(p) for p in files]
    else:
        inputs = a.images
    if len(inputs) != 4:
        ap.error(f"нужно ровно 4 изображения, получено {len(inputs)}")
    synth.run(inputs, a.out, "nn_quality" if a.quality else ("nn_refine" if a.refine else "nn"),
              a.steps, "peak", False, None if a.device == "auto" else a.device,
              0, a.checkpoint, figure=a.figure)
    print(f"ДОЭ: {Path(a.out) / 'doe.png'}")


if __name__ == "__main__":
    sys.exit(main())
