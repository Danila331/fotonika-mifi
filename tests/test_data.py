import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from doe.config import N
from doe.data import (EXT_DIR, FAMILIES, FAMILY_WEIGHTS, OBJECT_SOURCES, PHOTO_SOURCES, REAL_FAMILIES,
                      SIMPLE_FAMILIES, SPLITS, TEXTURE_SOURCES, ExternalBank, FontBank, QuadGenerator, split_fonts)

DATA = Path(__file__).resolve().parent.parent / "data"
HAVE_EXT = all((EXT_DIR / f"{s}_img.npy").exists() for s in (*PHOTO_SOURCES, *OBJECT_SOURCES, *TEXTURE_SOURCES))
needs_ext = pytest.mark.skipif(not HAVE_EXT, reason="external cache missing: run scripts/prepare_external.py")


@pytest.mark.parametrize("split", SPLITS)
def test_shapes_dtype(split):
    q = QuadGenerator(split, seed=3).sample(8)
    assert q.shape == (8, 4, N, N)
    assert q.dtype == np.uint8
    assert q.flags.c_contiguous


def test_determinism_and_streams():
    a = QuadGenerator("train", seed=7).sample(6)
    b = QuadGenerator("train", seed=7).sample(6)
    assert np.array_equal(a, b)
    c = QuadGenerator("train", seed=8).sample(6)
    d = QuadGenerator("val", seed=7).sample(6)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, d)
    g = QuadGenerator("test", seed=1)  # consecutive calls continue the stream
    assert not np.array_equal(g.sample(2), g.sample(2))


def test_weights():
    assert abs(sum(FAMILY_WEIGHTS.values()) - 1.0) < 1e-6
    real = sum(FAMILY_WEIGHTS[f] for f in REAL_FAMILIES)
    simple = sum(FAMILY_WEIGHTS[f] for f in SIMPLE_FAMILIES)
    assert 0.5 <= real <= 0.65 and 0.18 <= simple <= 0.32


def test_font_splits_disjoint():
    fams = {s: set(split_fonts(s)) for s in SPLITS}
    # macOS has ~100 families; on Linux/Colab only a handful of open fonts exist. val/test .npy are
    # pre-generated on macOS and shipped, so on Linux only the train stream needs real variety.
    need = {"train": 5, "val": 5, "test": 5} if sys.platform == "darwin" else {"train": 5, "val": 1, "test": 1}
    for s in SPLITS:
        assert len(fams[s]) >= need[s], f"{s}: too few font families"
    assert not fams["train"] & fams["val"]
    assert not fams["train"] & fams["test"]
    assert not fams["val"] & fams["test"]
    bank = FontBank.get()
    files = {s: {(f["path"], f["index"]) for k in bank.family_keys[s] for f in bank.families[s][k]} for s in SPLITS}
    assert not files["train"] & files["val"] and not files["train"] & files["test"] and not files["val"] & files["test"]
    for s in SPLITS:
        assert bank.cyr_faces[s], f"{s}: no Cyrillic-capable font"


def test_other_sources_disjoint():
    bank = FontBank.get()
    for attr, key in (("symbols", lambda g: (g["path"], g["cp"])), ("emoji_cands", lambda c: c),
                      ("words", lambda w: w)):
        sets = {s: {key(x) for x in getattr(bank, attr)[s]} for s in SPLITS}
        for s in SPLITS:
            assert sets[s], f"{attr}/{s} empty"
        assert not sets["train"] & sets["val"] and not sets["train"] & sets["test"] and not sets["val"] & sets["test"]


@needs_ext
def test_external_sources_split_by_class():
    ext = ExternalBank.get()
    counts = ext.counts()
    for name in (*PHOTO_SOURCES, *OBJECT_SOURCES, *TEXTURE_SOURCES):
        for s in SPLITS:
            assert counts[name][s] > 0, f"{name}/{s} empty"
        img = np.load(EXT_DIR / f"{name}_img.npy", mmap_mode="r")
        assert img.dtype == np.uint8 and img.shape[1:] in ((N, N), (28, 28))
        sp = np.load(EXT_DIR / f"{name}_split.npy")
        assert len(sp) == len(img)
        if name.startswith(("c101", "c256", "flowers", "dtd")):  # class-level split: no class in two splits
            cl = np.load(EXT_DIR / f"{name}_class.npy")
            cls = {s: set(cl[sp == i].tolist()) for i, s in enumerate(SPLITS)}
            assert not cls["train"] & cls["val"] and not cls["train"] & cls["test"] and not cls["val"] & cls["test"]
        if name in ext.mask:
            assert ext.mask[name].shape == img.shape
    man = json.loads((EXT_DIR / "manifest.json").read_text())
    for src in ("c101", "c256", "flowers", "dtd"):
        assert set(man[f"{src}_classes"].values()) == set(SPLITS)


@pytest.mark.parametrize("split", SPLITS)
def test_every_family_renders(split):
    g = QuadGenerator(split, seed=0)
    for fam in FAMILIES:
        ims = g.sample_family(fam, 3)
        assert ims.shape == (3, N, N) and ims.dtype == np.uint8, fam


@needs_ext
def test_mix_statistics():
    q, fams = QuadGenerator("train", seed=11).sample(256, return_families=True)
    flat = [f for fs in fams for f in fs]
    real = np.mean([f in REAL_FAMILIES for f in flat])
    simple = np.mean([f in SIMPLE_FAMILIES for f in flat])
    assert 0.48 < real < 0.68, real
    assert 0.15 < simple < 0.35, simple
    ims = q.reshape(-1, N, N)
    means = ims.reshape(len(ims), -1).mean(1)
    assert means.min() < 5 and means.max() > 200          # near-empty and near-full images both occur
    gray = np.mean([len(np.unique(im)) > 32 for im in ims])
    assert gray > 0.55                                       # mostly halftone images now
    binary = np.mean([np.isin(im, (0, 255)).mean() > 0.95 for im in ims])
    assert 0.05 < binary < 0.4


def test_generator_speed():
    g = QuadGenerator("train", seed=0)
    g.sample(64)  # warm-up (font / emoji / mmap caches)
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        g.sample(64)
        ts.append(time.perf_counter() - t)
    assert np.median(ts) < 1.0, f"sample(64) took {np.median(ts):.2f}s"


def test_fixed_sets():
    for name in ("val", "test"):
        p = DATA / f"{name}.npy"
        if not p.exists():
            pytest.skip("run scripts/make_datasets.py first")
        a = np.load(p)
        assert a.dtype == np.uint8 and a.shape == (256, 4, N, N)
        # v2 sets are mostly halftone (many gray levels)
        assert np.mean([len(np.unique(im)) > 32 for im in a.reshape(-1, N, N)]) > 0.5
    assert not np.array_equal(np.load(DATA / "val.npy")[:4], np.load(DATA / "test.npy")[:4])
    for name in ("val", "test"):
        old = DATA / "v1" / f"{name}.npy"
        if old.exists():
            assert np.load(old).shape == (256, 4, N, N)


def test_demo_pngs():
    demo = DATA / "demo"
    if not demo.exists():
        pytest.skip("run scripts/make_datasets.py first")
    dirs = sorted(d for d in demo.iterdir() if d.is_dir())
    assert 10 <= len(dirs) <= 14
    for name in ("halftone_photos", "dark_objects", "rocket_smoke", "photo_text"):
        assert (demo / name).is_dir(), name
    for d in dirs:
        for i in range(1, 5):
            im = Image.open(d / f"target_{i}.png")
            assert im.mode == "L" and im.size == (N, N), f"{d.name}/target_{i}.png"
    stress = [np.asarray(Image.open(demo / "stress" / f"target_{i}.png")) for i in range(1, 5)]
    assert stress[0].max() == 0 and stress[1].min() == 255 and np.array_equal(stress[2], stress[3])


def test_torch_dataset():
    from doe.data import QuadDataset
    if QuadDataset is None:
        pytest.skip("torch not available")
    batches = list(QuadDataset("train", seed=0, batch=2, length=3))
    assert len(batches) == 3 and tuple(batches[0].shape) == (2, 4, N, N)


# ------------------------------------------------------------------ portability (Linux / Colab)

def _open_font_files():
    """Open-licence fonts available locally (bundled with Python packages), for a Linux-like simulation."""
    import site
    roots = [Path(p) for p in site.getsitepackages()]
    found = []
    for r in roots:
        for pat in ("**/DejaVu*.ttf", "**/Liberation*.ttf", "**/KaTeX_*.ttf", "**/Noto*.ttf"):
            found += list(r.glob(pat))
    return sorted(set(found))[:40]


def _bank_for(tmp_path, monkeypatch, files):
    import shutil
    from doe import data as D
    fdir = tmp_path / "usr" / "share" / "fonts" / "truetype"
    fdir.mkdir(parents=True)
    for f in files:
        shutil.copy(f, fdir / f.name)
    bank = D.FontBank(dirs=((str(tmp_path / "usr" / "share" / "fonts"), True),), cache_dir=tmp_path / "cache")
    monkeypatch.setattr(D.FontBank, "_instance", bank)
    return bank


def test_font_split_rule_is_name_based():
    from doe.data import font_family_split
    bank = FontBank.get()
    for key, s in bank.split_of_family.items():  # current machine agrees with the pure name rule
        assert font_family_split(key) == s
    # names shared with the macOS val/test families keep their split everywhere
    assert font_family_split("gill") == "val" and font_family_split("tahoma") == "test"
    assert font_family_split("arial") == "train"
    assert font_family_split("dejavu") == "train" and font_family_split("liberation") == "train"
    assert font_family_split("noto") == "train"
    assert font_family_split("freemono") == "val" and font_family_split("freeserif") == "test"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_simulated_linux_open_fonts(tmp_path, monkeypatch):
    files = _open_font_files()
    if len(files) < 2:
        pytest.skip("no open fonts bundled in site-packages")
    bank = _bank_for(tmp_path, monkeypatch, files)
    assert all(str(tmp_path) in f["path"] for f in bank.faces)          # no macOS paths leak in
    fams = {s: set(bank.split_fonts(s)) for s in SPLITS}
    assert fams["train"], "no train fonts found"
    assert not fams["train"] & fams["val"] and not fams["train"] & fams["test"] and not fams["val"] & fams["test"]
    for s in SPLITS:
        g = QuadGenerator(s, seed=0)
        q = g.sample(16)
        assert q.shape == (16, 4, N, N) and q.dtype == np.uint8
        assert abs(sum(g.family_mix().values()) - 1) < 1e-9
        if not bank.emoji_font:
            assert "emoji" not in g.family_mix()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_no_fonts_fallback(tmp_path, monkeypatch):
    bank = _bank_for(tmp_path, monkeypatch, [])
    assert not bank.faces and bank.emoji_font is None
    for s in SPLITS:
        assert bank.split_fonts(s) == []
        g = QuadGenerator(s, seed=1)
        assert {"emoji", "symbol"} <= g.dropped
        mix = g.family_mix()
        assert "emoji" not in mix and "symbol" not in mix and abs(sum(mix.values()) - 1) < 1e-9
        assert g.sample(12).shape == (12, 4, N, N)
        for fam in ("glyph", "word", "logo", "phototext", "emoji", "symbol"):  # direct calls must not crash
            assert g.sample_family(fam, 2).shape == (2, N, N)
    assert "split=" in QuadGenerator("train", 0).describe()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_font_cache_keyed_and_validated(tmp_path):
    from doe import data as D
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    cache = tmp_path / "cache"
    D.FontBank(dirs=((str(a), True),), cache_dir=cache)
    D.FontBank(dirs=((str(b), True),), cache_dir=cache)
    names = sorted(p.name for p in cache.glob("fontbank_*.json"))
    assert len(names) == 2 and all(n.startswith(f"fontbank_{sys.platform}_") for n in names)
    # a cache whose paths do not exist (e.g. copied from another machine) is not reused
    p = cache / names[0]
    d = json.loads(p.read_text())
    d["faces"] = [{"path": "/System/Library/Fonts/Nope.ttf", "index": 0, "family": "Nope", "style": "R",
                   "cyr": False, "key": "nope"}]
    p.write_text(json.dumps(d))
    for dirs in (a, b):
        bank = D.FontBank(dirs=((str(dirs), True),), cache_dir=cache)
        assert not bank.faces


def test_verbose_line(monkeypatch, capsys):
    monkeypatch.setenv("DOE_DATA_VERBOSE", "1")
    monkeypatch.setattr(QuadGenerator, "_logged", False)
    QuadGenerator("train", seed=0)
    QuadGenerator("val", seed=0)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("[doe.data]") and "mix[" in out[0] and "fonts[" in out[0]
