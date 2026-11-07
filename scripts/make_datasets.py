"""Build the fixed evaluation sets, named demo quads and a training-data contact sheet (data v2).

  data/val.npy, data/test.npy      uint8 (256, 4, 128, 128), fixed seeds, disjoint sources (doe.data);
                                   the previous (v1, mostly glyphs/shapes) sets are moved to data/v1/ once
  data/demo/<name>/target_{1..4}.png   8-bit mode-L 128x128 demo quads
  data/preview.png                 32 random training quads (one quad per row-cell, 4 images each)

Usage: .venv/bin/python scripts/make_datasets.py [--m 256]
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from doe.config import N  # noqa: E402
from doe.data import Painter, QuadGenerator, _SS, _draw_shape, _paste, _to_u8  # noqa: E402
from doe.io import save_gray  # noqa: E402

DATA = ROOT / "data"
VAL_SEED, TEST_SEED, PREVIEW_SEED = 1001, 2002, 3003
# seeds of hand-picked (eyeballed) demo images, Painter("test") family methods
DEMO_SEEDS = {
    "halftone_photos": [3, 4, 8, 11],             # photo
    "dark_objects": {"darkscene": [3, 10], "object": [11, 9]},
    "rocket_smoke": [5, 9, 11, 13],               # darkscene
    "photo_text": {"photo": 14, "phototext": 9, "object": 16},
}

BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
SERIF = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)


def text_img(text, path=BOLD, height=0.75, max_w=0.9, invert=False):
    """Centered white text on black, glyph-box height = height*N (width limited to max_w*N)."""
    f = _font(path, 200)
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    bb = d.multiline_textbbox((0, 0), text, font=f, align="center")
    w, h = int(bb[2] - bb[0]) + 8, int(bb[3] - bb[1]) + 8
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).multiline_text((4 - bb[0], 4 - bb[1]), text, font=f, fill=255, align="center")
    im = im.crop(im.getbbox())
    s = min(height * N / im.size[1], max_w * N / im.size[0])
    im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.LANCZOS)
    out = Image.new("L", (N, N), 0)
    out.paste(im, ((N - im.size[0]) // 2, (N - im.size[1]) // 2))
    a = np.asarray(out, dtype=np.uint8)
    return 255 - a if invert else a


def shape_img(kind, r=0.4, filled=True, ang=0.0, seed=0):
    sf = _SS(2)
    _draw_shape(sf, kind, N / 2, N / 2, r * N, ang, 1.0, filled, 4, np.random.default_rng(seed))
    return _to_u8(sf.done())


def family_img(painter, method, seed, **kw):
    return _to_u8(np.clip(getattr(painter, method)(np.random.default_rng(seed), **kw), 0, 1))


def pick(painter, method, seeds, **kw):
    return [family_img(painter, method, s, **kw) for s in seeds]


def text_on(img_u8, text, path=BOLD, height=0.22, y=0.78, value=255):
    """Overlay centered text (with a soft dark shadow) on an image."""
    t = text_img(text, path, height=height, max_w=0.92).astype(np.float32) / 255.0
    ys, xs = np.nonzero(t > 0.05)
    patch = t[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    m = _paste(np.zeros((N, N), np.float32), patch, (N - patch.shape[1]) // 2, int(y * N - patch.shape[0] / 2))
    from scipy.ndimage import gaussian_filter, shift
    sh = gaussian_filter(shift(m, (1.5, 1.5), order=1), 1.2)
    out = img_u8.astype(np.float32) / 255.0 * (1 - 0.8 * sh)
    out = out * (1 - m) + (value / 255.0) * m
    return _to_u8(out)


def demo_quads():
    p = Painter("test")
    gx = np.linspace(0, 1, N, dtype=np.float32)[None, :].repeat(N, 0)
    quads = {
        "digits_1234": [text_img(c) for c in "1234"],
        "letters_MIFI": [text_img(c) for c in "MIFI"],
        "cyrillic_NIYAU": [text_img(c, SERIF) for c in "НИЯУ"],
        "words": [text_img("DOE", height=0.5), text_img("LASER", REGULAR, height=0.4),
                  text_img("МИФИ", height=0.45), text_img("2026", SERIF, height=0.5)],
        "shapes": [shape_img("circle", 0.38), shape_img("triangle", 0.42, seed=1), shape_img("star", 0.45, seed=2),
                   shape_img("ring", 0.4, filled=False)],
        "mixed": [text_img("A", height=0.7), shape_img("star", 0.42, seed=3), _to_u8(gx),
                  family_img(p, "_dead_leaves", 7)],
        "photo_like": [family_img(p, "_landscape", 11), family_img(p, "_still_life", 12),
                       family_img(p, "_portrait", 13), family_img(p, "emoji", 14)],
        "stress": [np.zeros((N, N), np.uint8), np.full((N, N), 255, np.uint8), text_img("7"), text_img("7")],
        # ---- v2: halftone photos / illustrations (test-split sources only) ----
        "halftone_photos": pick(p, "photo", DEMO_SEEDS["halftone_photos"]),
        "dark_objects": pick(p, "darkscene", DEMO_SEEDS["dark_objects"]["darkscene"])
        + pick(p, "object", DEMO_SEEDS["dark_objects"]["object"]),
        "rocket_smoke": pick(p, "darkscene", DEMO_SEEDS["rocket_smoke"]),
        "photo_text": [text_on(family_img(p, "photo", DEMO_SEEDS["photo_text"]["photo"]), "MIFI"),
                       text_img("LASER", height=0.4), family_img(p, "phototext", DEMO_SEEDS["photo_text"]["phototext"]),
                       family_img(p, "object", DEMO_SEEDS["photo_text"]["object"])],
    }
    return quads


def contact_sheet(quads, cols=4, pad=2, gap=8):
    """quads: (M,4,N,N) -> image with `cols` quads per row, separated by gaps."""
    m = len(quads)
    rows = (m + cols - 1) // cols
    qw = 4 * N + 3 * pad
    W = cols * qw + (cols - 1) * gap
    H = rows * N + (rows - 1) * gap
    sheet = np.full((H, W), 60, np.uint8)
    for k, q in enumerate(quads):
        r, c = divmod(k, cols)
        y = r * (N + gap)
        for i in range(4):
            x = c * (qw + gap) + i * (N + pad)
            sheet[y:y + N, x:x + N] = q[i]
    return sheet


def write_fixed_sets(m):
    v1 = DATA / "v1"
    if not v1.exists():  # keep the previous (v1) fixed sets once
        v1.mkdir()
        for split in ("val", "test"):
            if (DATA / f"{split}.npy").exists():
                shutil.copy2(DATA / f"{split}.npy", v1 / f"{split}.npy")
        print("previous val/test copied to data/v1/")
    for split, seed in (("val", VAL_SEED), ("test", TEST_SEED)):
        arr = QuadGenerator(split, seed=seed).sample(m)
        tmp = DATA / f".{split}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, DATA / f"{split}.npy")  # atomic switch for readers
        print(f"data/{split}.npy {arr.shape} {arr.dtype} mean={arr.mean():.1f}")


def write_demos():
    for name, imgs in demo_quads().items():
        for i, im in enumerate(imgs, 1):
            save_gray(DATA / "demo" / name / f"target_{i}.png", np.ascontiguousarray(im, dtype=np.uint8))
        print(f"data/demo/{name}/target_1..4.png")


def write_preview():
    prev = QuadGenerator("train", seed=PREVIEW_SEED).sample(32)
    Image.fromarray(contact_sheet(prev)).save(DATA / ".preview.tmp.png")
    os.replace(DATA / ".preview.tmp.png", DATA / "preview.png")
    print("data/preview.png (32 training quads, 4 per row)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--only", nargs="*", default=["sets", "demos", "preview"], choices=["sets", "demos", "preview"])
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    if "sets" in args.only:
        write_fixed_sets(args.m)
    if "demos" in args.only:
        write_demos()
    if "preview" in args.only:
        write_preview()


if __name__ == "__main__":
    main()
