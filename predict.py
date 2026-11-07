"""Точка входа решения: 4 изображения одного комплекта → один фазовый ДОЭ (PNG 128×128, 8 бит, градации серого).

    python predict.py img1.png img2.png img3.png img4.png --out out        # основной режим
    python predict.py --in-dir in --out out                                # 4 PNG из папки, по алфавиту
    python predict.py --in-dir in --out out --fast                         # только нейросеть, один проход

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
    ap.add_argument("--fast", action="store_true", help="быстрый режим: только нейросеть, один проход, без уточнения")
    ap.add_argument("--steps", type=int, help="число шагов уточнения основного режима (по умолчанию 100)")
    ap.add_argument("--device", default="cpu", help="cpu (по умолчанию) или cuda / mps")
    ap.add_argument("--figure", action="store_true", help="дополнительно нарисовать comparison.png")
    a = ap.parse_args(argv)
    if a.in_dir:
        files = sorted(p for p in Path(a.in_dir).iterdir() if p.suffix.lower() == ".png")
        inputs = [str(p) for p in files]
    else:
        inputs = a.images
    if len(inputs) != 4:
        ap.error(f"нужно ровно 4 изображения, получено {len(inputs)}")
    synth.run(inputs, a.out, "nn" if a.fast else "nn_refine", None if a.fast else a.steps, "peak", False,
              a.device, 0, None, figure=a.figure)
    print(f"ДОЭ: {Path(a.out) / 'doe.png'}")


if __name__ == "__main__":
    sys.exit(main())
