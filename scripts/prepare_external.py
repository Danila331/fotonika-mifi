"""Preprocess downloaded open datasets into a compact 128x128 grayscale uint8 cache.

Archives in data/external/_downloads/ are treated as DATA: they are never extracted to disk and nothing from them
is executed - JPEG/PNG bytes and .mat annotation files are streamed straight out of the tar/zip archives,
decoded with PIL / scipy.io.loadmat and resized.

Output (data/external/cache/):
  <source>_img.npy   uint8 (M,128,128)  (fmnist: (M,28,28))
  <source>_mask.npy  uint8 (M,128,128)  object masks (object sources only)
  <source>_split.npy uint8 (M,)         0 train / 1 val / 2 test  (split by CLASS, or by official/id split)
  <source>_class.npy int16 (M,)
  manifest.json      class names, split of each class, counts

Sources:
  c101_photo / c101_obj    Caltech-101 (center crop / annotated object crops with contour masks)
  c256_photo / c256_obj    Caltech-256 (center crop, <= CAP_256 per class / auto-masked plain-background objects)
  flowers_photo            Oxford Flowers-102
  bsds_photo               BSDS500 (3 crops per image)
  dtd_tex                  Describable Textures Dataset
  fmnist_obj               Fashion-MNIST (28x28, objects on black)

Usage: .venv/bin/python -I scripts/prepare_external.py [--downloads DIR] [--out DIR] [--workers 8]
"""

import argparse
import gzip
import io
import json
import sys
import tarfile
import zipfile
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
S = 128
CAP_256 = 60
Image.MAX_IMAGE_PIXELS = 60_000_000

# classes forced into TRAIN (aerospace / night-sky / glow content resembling the expected jury images)
PIN_TRAIN = {
    "c101": {"airplanes", "helicopter", "Faces_easy", "Faces", "Motorbikes", "car_side", "watch", "cup"},
    "c256": {"252.car-side-101", "137.mars", "177.saturn", "254.greyhound", "039.chess-board", "251.airplanes-101",
             "254.greyhound", "224.touring-bike", "141.motorbikes-101", "145.motorbikes-101", "246.wine-bottle",
             "253.faces-easy-101", "250.zebra", "038.chimp", "060.duck", "033.cd", "212.teapot", "158.penguin"},
    "flowers": set(),
    "dtd": {"smeared", "marbled", "veined", "swirly"},
}
PIN_TRAIN_SUBSTR = {"c256": ("space-shuttle", "comet", "galaxy", "fireworks", "lightning", "rainbow", "fighter-jet",
                             "airplane", "blimp", "hot-air-balloon", "helicopter", "mars", "saturn", "sunflower",
                             "telescope", "light-house", "car-side", "faces")}
C101_ANNOT = {"Faces": "Faces_2", "Faces_easy": "Faces_3", "Motorbikes": "Motorbikes_16", "airplanes": "Airplanes_Side_2"}


def class_splits(source, classes):
    pinned = set(PIN_TRAIN.get(source, set()))
    for c in classes:
        if any(s in c for s in PIN_TRAIN_SUBSTR.get(source, ())):
            pinned.add(c)
    free = sorted((c for c in classes if c not in pinned), key=lambda c: zlib.crc32(f"{source}:{c}".encode()))
    n_val = max(1, round(0.15 * len(classes)))
    n_test = max(1, round(0.15 * len(classes)))
    out = {c: 0 for c in classes}
    for i, c in enumerate(free):
        out[c] = 1 if i < n_val else (2 if i < n_val + n_test else 0)
    return out


# ------------------------------------------------------------------ image helpers (run in worker processes)

def _open_gray(data, draft=512):
    im = Image.open(io.BytesIO(data))
    if im.format == "JPEG":
        im.draft("L", (draft, draft))
    if im.mode not in ("L", "RGB"):
        im = im.convert("RGB")
    return im.convert("L")


def center_crop(im, s=S):
    w, h = im.size
    k = min(w, h)
    im = im.crop(((w - k) // 2, (h - k) // 2, (w - k) // 2 + k, (h - k) // 2 + k))
    return np.asarray(im.resize((s, s), Image.LANCZOS), dtype=np.uint8)


def random_crops(im, n, seed, s=S):
    rng = np.random.default_rng(seed)
    w, h = im.size
    out = []
    for _ in range(n):
        k = int(min(w, h) * rng.uniform(0.6, 1.0))
        x, y = int(rng.integers(0, w - k + 1)), int(rng.integers(0, h - k + 1))
        out.append(np.asarray(im.crop((x, y, x + k, y + k)).resize((s, s), Image.LANCZOS), dtype=np.uint8))
    return out


def object_crop(img, mask, s=S, margin=0.08):
    """Square crop around the mask bbox (padded with zeros), resized to s. img/mask uint8 arrays."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    k = int(max(y1 - y0, x1 - x0) * (1 + 2 * margin)) + 2
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    H, W = img.shape
    pad = k
    ip = np.pad(img, pad, mode="edge")
    mp = np.pad(mask, pad)
    ty, tx = cy - k // 2 + pad, cx - k // 2 + pad
    ci = Image.fromarray(ip[ty:ty + k, tx:tx + k]).resize((s, s), Image.LANCZOS)
    cm = Image.fromarray(mp[ty:ty + k, tx:tx + k]).resize((s, s), Image.BILINEAR)
    return np.asarray(ci, dtype=np.uint8), np.asarray(cm, dtype=np.uint8)


def auto_mask(a):
    """Foreground mask for images on a plain background, or None."""
    h, w = a.shape
    b = np.concatenate([a[:4].ravel(), a[-4:].ravel(), a[:, :4].ravel(), a[:, -4:].ravel()]).astype(np.int16)
    bg = int(np.median(b))
    if np.mean(np.abs(b - bg) < 14) < 0.92:
        return None
    d = ndimage.gaussian_filter(np.abs(a.astype(np.int16) - bg).astype(np.float32), 1.0)
    m = d > 22
    m = ndimage.binary_closing(m, iterations=2)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.nonzero(sizes >= max(0.15 * sizes.max(), 0.002 * m.size))[0])
    frac = keep.mean()
    if not 0.04 < frac < 0.8:
        return None
    return (keep * 255).astype(np.uint8)


def _job_photo(args):
    data, kind, seed = args
    try:
        im = _open_gray(data)
        if min(im.size) < 48:
            return None
        if kind == "bsds":
            return random_crops(im, 3, seed)
        out = [center_crop(im)]
        if kind == "c256":
            a = np.asarray(im.resize((im.size[0] * 256 // max(im.size), im.size[1] * 256 // max(im.size)),
                                     Image.LANCZOS) if max(im.size) > 256 else im, dtype=np.uint8)
            m = auto_mask(a)
            out.append(object_crop(a, m) if m is not None else None)
        return out
    except Exception:  # noqa: BLE001  (corrupt / unsupported file)
        return None


def _job_c101(args):
    data, annot = args
    try:
        im = _open_gray(data, draft=4096)
        out = [center_crop(im)]
        obj = None
        if annot is not None:
            m = loadmat(io.BytesIO(annot))
            box = np.asarray(m["box_coord"]).ravel().astype(float)  # [y1 y2 x1 x2]
            cont = np.asarray(m["obj_contour"], dtype=float)         # (2,K) x,y relative to box
            pts = [(box[2] + x, box[0] + y) for x, y in cont.T]
            mk = Image.new("L", im.size, 0)
            if len(pts) >= 3:
                ImageDraw.Draw(mk).polygon(pts, fill=255)
                a, ma = np.asarray(im, dtype=np.uint8), np.asarray(mk, dtype=np.uint8)
                if ma.mean() > 255 * 0.01:
                    obj = object_crop(a, ma)
        out.append(obj)
        return out
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ source readers

def iter_tar(fileobj_or_path, suffixes=(".jpg", ".jpeg", ".png")):
    kw = {"fileobj": fileobj_or_path} if not isinstance(fileobj_or_path, (str, Path)) else {"name": str(fileobj_or_path)}
    with tarfile.open(mode="r|*", **kw) as tf:
        for m in tf:
            if m.isfile() and m.name.lower().endswith(suffixes) and "/._" not in m.name:
                f = tf.extractfile(m)
                if f is not None:
                    yield m.name, f.read()


def save(out, name, imgs, split, cls, masks=None):
    np.save(out / f"{name}_img.npy", np.ascontiguousarray(np.stack(imgs).astype(np.uint8)))
    np.save(out / f"{name}_split.npy", np.asarray(split, np.uint8))
    np.save(out / f"{name}_class.npy", np.asarray(cls, np.int16))
    if masks is not None:
        np.save(out / f"{name}_mask.npy", np.ascontiguousarray(np.stack(masks).astype(np.uint8)))
    sp = np.asarray(split)
    print(f"  {name}: {len(imgs)} images (train {np.sum(sp == 0)}, val {np.sum(sp == 1)}, test {np.sum(sp == 2)})",
          flush=True)
    return {"count": len(imgs), "train": int(np.sum(sp == 0)), "val": int(np.sum(sp == 1)),
            "test": int(np.sum(sp == 2))}


def do_c101(dl, out, pool, manifest):
    z = zipfile.ZipFile(dl / "caltech-101.zip")
    names = z.namelist()
    inner_img = next(n for n in names if n.endswith("101_ObjectCategories.tar.gz"))
    inner_ann = next(n for n in names if n.endswith("Annotations.tar"))
    annots = {}
    with z.open(inner_ann) as f:
        for name, data in iter_tar(f, (".mat",)):
            p = Path(name)
            annots[(p.parent.name, p.stem.replace("annotation_", ""))] = data
    jobs, meta = [], []
    with z.open(inner_img) as f:
        for name, data in iter_tar(f):
            p = Path(name)
            c = p.parent.name
            if c == "BACKGROUND_Google":
                continue
            a = annots.get((C101_ANNOT.get(c, c), p.stem.replace("image_", "")))
            jobs.append((data, a))
            meta.append(c)
    classes = sorted(set(meta))
    cs = class_splits("c101", classes)
    ph, ob, obm, sp, cl, osp, ocl = [], [], [], [], [], [], []
    for c, r in zip(meta, pool.imap(_job_c101, jobs, chunksize=16)):
        if r is None:
            continue
        ph.append(r[0]); sp.append(cs[c]); cl.append(classes.index(c))
        if r[1] is not None:
            ob.append(r[1][0]); obm.append(r[1][1]); osp.append(cs[c]); ocl.append(classes.index(c))
    manifest["c101_photo"] = save(out, "c101_photo", ph, sp, cl)
    manifest["c101_obj"] = save(out, "c101_obj", ob, osp, ocl, obm)
    manifest["c101_classes"] = {c: ("train", "val", "test")[cs[c]] for c in classes}


def do_c256(dl, out, pool, manifest):
    per = {}
    jobs, meta = [], []
    for name, data in iter_tar(dl / "256_ObjectCategories.tar"):
        c = Path(name).parent.name
        if per.get(c, 0) >= CAP_256:
            continue
        per[c] = per.get(c, 0) + 1
        jobs.append((data, "c256", 0))
        meta.append(c)
    classes = sorted(set(meta))
    cs = class_splits("c256", classes)
    ph, ob, obm, sp, cl, osp, ocl = [], [], [], [], [], [], []
    for c, r in zip(meta, pool.imap(_job_photo, jobs, chunksize=16)):
        if r is None:
            continue
        ph.append(r[0]); sp.append(cs[c]); cl.append(classes.index(c))
        if r[1] is not None and not c.startswith("257.clutter"):
            ob.append(r[1][0]); obm.append(r[1][1]); osp.append(cs[c]); ocl.append(classes.index(c))
    manifest["c256_photo"] = save(out, "c256_photo", ph, sp, cl)
    manifest["c256_obj"] = save(out, "c256_obj", ob, osp, ocl, obm)
    manifest["c256_classes"] = {c: ("train", "val", "test")[cs[c]] for c in classes}


def do_flowers(dl, out, pool, manifest):
    labels = loadmat(dl / "imagelabels.mat")["labels"].ravel()
    jobs, meta = [], []
    for name, data in iter_tar(dl / "102flowers.tgz"):
        idx = int(Path(name).stem.split("_")[-1]) - 1
        jobs.append((data, "flowers", 0))
        meta.append(str(int(labels[idx])))
    classes = sorted(set(meta), key=int)
    cs = class_splits("flowers", classes)
    ph, sp, cl = [], [], []
    for c, r in zip(meta, pool.imap(_job_photo, jobs, chunksize=16)):
        if r is not None:
            ph.append(r[0]); sp.append(cs[c]); cl.append(classes.index(c))
    manifest["flowers_photo"] = save(out, "flowers_photo", ph, sp, cl)
    manifest["flowers_classes"] = {c: ("train", "val", "test")[cs[c]] for c in classes}


def do_bsds(dl, out, pool, manifest):
    jobs, meta = [], []
    for name, data in iter_tar(dl / "BSR_bsds500.tgz", (".jpg",)):
        p = Path(name)
        if "/images/" not in name or p.parent.name not in ("train", "val", "test"):
            continue
        jobs.append((data, "bsds", int(p.stem)))
        meta.append(p.parent.name)
    ph, sp, cl = [], [], []
    for s, r in zip(meta, pool.imap(_job_photo, jobs, chunksize=8)):
        if r is not None:
            for a in r:
                ph.append(a); sp.append({"train": 0, "val": 1, "test": 2}[s]); cl.append(0)
    manifest["bsds_photo"] = save(out, "bsds_photo", ph, sp, cl)


def do_dtd(dl, out, pool, manifest):
    jobs, meta = [], []
    for name, data in iter_tar(dl / "dtd-r1.0.1.tar.gz", (".jpg",)):
        if "/images/" not in name:
            continue
        jobs.append((data, "dtd", 0))
        meta.append(Path(name).parent.name)
    classes = sorted(set(meta))
    cs = class_splits("dtd", classes)
    ph, sp, cl = [], [], []
    for c, r in zip(meta, pool.imap(_job_photo, jobs, chunksize=16)):
        if r is not None:
            ph.append(r[0]); sp.append(cs[c]); cl.append(classes.index(c))
    manifest["dtd_tex"] = save(out, "dtd_tex", ph, sp, cl)
    manifest["dtd_classes"] = {c: ("train", "val", "test")[cs[c]] for c in classes}


def do_fmnist(dl, out, pool, manifest):
    def idx(path, off):
        with gzip.open(path) as f:
            return np.frombuffer(f.read(), np.uint8, offset=off)
    tr = idx(dl / "fmnist-train-images-idx3-ubyte.gz", 16).reshape(-1, 28, 28)
    te = idx(dl / "fmnist-t10k-images-idx3-ubyte.gz", 16).reshape(-1, 28, 28)
    ltr = idx(dl / "fmnist-train-labels-idx1-ubyte.gz", 8)
    lte = idx(dl / "fmnist-t10k-labels-idx1-ubyte.gz", 8)
    imgs = np.concatenate([tr, te])
    split = np.concatenate([np.zeros(len(tr), np.uint8), 1 + (np.arange(len(te)) % 2).astype(np.uint8)])
    manifest["fmnist_obj"] = save(out, "fmnist_obj", list(imgs), split, np.concatenate([ltr, lte]))


SOURCES = {"fmnist": do_fmnist, "bsds": do_bsds, "flowers": do_flowers, "dtd": do_dtd, "c101": do_c101,
           "c256": do_c256}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", default=str(ROOT / "data" / "external" / "_downloads"))
    ap.add_argument("--out", default=str(ROOT / "data" / "external" / "cache"))
    ap.add_argument("--only", nargs="*", default=list(SOURCES))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dl, out = Path(args.downloads), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    with Pool(args.workers) as pool:
        for s in args.only:
            print(f"[{s}]", flush=True)
            SOURCES[s](dl, out, pool, manifest)
            mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print("done")


if __name__ == "__main__":
    sys.exit(main())
