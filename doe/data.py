"""Target generator (v2): real halftone photos / objects from open datasets + procedural content.

API (stable):
  QuadGenerator(split="train"|"val"|"test", seed=0).sample(batch) -> uint8 (batch, 4, N, N)

Real images come from the preprocessed cache data/external/cache/ (scripts/prepare_external.py; sources and
licences in data/SOURCES.md): Caltech-101/256, Oxford Flowers-102, BSDS500, DTD, Fashion-MNIST. If the cache is
missing the real-image families fall back to their synthetic counterparts (with a warning).

Image families (per-image probabilities in FAMILY_WEIGHTS):
  real / halftone (~60%):
    photo       augmented grayscale photo crops (crop/zoom, flip, gamma, contrast, vignette, porthole on black)
    object      masked objects composited on dark backgrounds (+ glow, vignette, metallic tone)
    darkscene   "rocket in smoke" style: object (real or synthetic rocket) on a dark sky, lit smoke, glow, bloom
    phototext   photo + text (caption, photo inside lettering, badge, caption bar)
    realtexture DTD texture crops
    emoji       Apple Color Emoji rendered to grayscale (shaded illustrations)
  simple (~24%): glyph, word, multiglyph, shapes, lineart, logo, symbol, dots
  synthetic (~16%): synphoto (dead leaves / landscape / still life / portrait), texture, gradient, pattern,
    hard (empty, full white, constant gray, single dot, white with a small dark object)

Quad level: ~4% "sequence" quads (same font/size/position: "1 2 3 4", "M I F I", ...), ~30% themed quads (all
four images from one family), the rest independent mixes; occasional duplicated images inside a quad.

Train/val/test separation is by SOURCE: font families are split into disjoint sets (popular fonts pinned to
train); external images are split by class (Caltech, Flowers, DTD) or by the official / id split (BSDS500,
Fashion-MNIST) before any augmentation; dingbat glyphs / emoji / words are hash-split; each split has its own
independent seed stream.
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .config import N

SPLITS = ("train", "val", "test")
_SPLIT_ID = {"train": 0, "val": 1, "test": 2}

# font search dirs: (dir, recursive). macOS dirs are scanned flat (as in data v1/v2), Linux dirs recursively.
MAC_FONT_DIRS = (("/System/Library/Fonts", False), ("/System/Library/Fonts/Supplemental", False),
                 ("/Library/Fonts", False))
LINUX_FONT_DIRS = (("/usr/share/fonts", True), ("/usr/local/share/fonts", True), ("~/.fonts", True),
                   ("~/.local/share/fonts", True))
FONT_DIRS = MAC_FONT_DIRS + LINUX_FONT_DIRS
FONT_DIRS_ENV = "DOE_FONT_DIRS"      # os.pathsep-separated override, searched recursively
VERBOSE_ENV = "DOE_DATA_VERBOSE"     # "1": print one line (fonts found, family mix) per process
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
_BANK_VERSION = 5


def font_dirs():
    env = os.environ.get(FONT_DIRS_ENV)
    if env:
        return tuple((d, True) for d in env.split(os.pathsep) if d)
    return FONT_DIRS

FAMILY_WEIGHTS = {
    # real / halftone
    "photo": 0.28, "object": 0.14, "darkscene": 0.09, "phototext": 0.04, "realtexture": 0.04, "emoji": 0.03,
    # simple
    "glyph": 0.07, "word": 0.05, "multiglyph": 0.01, "shapes": 0.04, "lineart": 0.02, "logo": 0.03,
    "symbol": 0.01, "dots": 0.01,
    # synthetic
    "synphoto": 0.03, "texture": 0.03, "gradient": 0.03, "pattern": 0.03, "hard": 0.02,
}
REAL_FAMILIES = ("photo", "object", "darkscene", "phototext", "realtexture", "emoji")
SIMPLE_FAMILIES = ("glyph", "word", "multiglyph", "shapes", "lineart", "logo", "symbol", "dots")
EXT_DIR = Path(__file__).resolve().parent.parent / "data" / "external" / "cache"
PHOTO_SOURCES = {"c256_photo": 0.35, "c101_photo": 0.2, "flowers_photo": 0.25, "bsds_photo": 0.2}
OBJECT_SOURCES = {"c101_obj": 0.45, "c256_obj": 0.37, "fmnist_obj": 0.18}
TEXTURE_SOURCES = {"dtd_tex": 1.0}
FAMILIES = tuple(FAMILY_WEIGHTS)
_FW = np.array([FAMILY_WEIGHTS[f] for f in FAMILIES], dtype=np.float64)
_FW /= _FW.sum()
_THEME_FAMILIES = tuple(f for f in FAMILIES if f != "hard")
_TW = np.array([FAMILY_WEIGHTS[f] for f in _THEME_FAMILIES], dtype=np.float64)
_TW /= _TW.sum()

# font families pinned to a split (lower-case family key); everything else is split by a stable hash.
# The most common "jury" fonts go to train; a few standard UI fonts are kept for val/test so that those
# splits are not only exotic display fonts.
PINNED_FONT_SPLIT = {
    "arial": "train", "helvetica": "train", "times": "train", "courier": "train", "verdana": "train",
    "georgia": "train", "menlo": "train", "pt": "train", "system": "train", "sf": "train", "impact": "train",
    "comic": "train", "avenir": "train", "futura": "train", "lucida": "train",
    "gill": "val", "palatino": "val", "baskerville": "val",
    "tahoma": "test", "trebuchet": "test", "charter": "test", "geneva": "test",
}
_EXCLUDE_FONT_FILES = (
    "emoji", "braille", "lastresort", "keyboard", "sfcamera", "webdings", "wingdings", "zapfdingbats",
    "ornaments", "symbol", "arial unicode", "stixsiz", "stixint", "stixnonuni", "stixvar", "stixtwomath",
    "applegothic", "coloremoji", "color emoji", "dingbat",
)

# Frozen macOS font-family split (data v1/v2; the shipped val/test sets were rendered with it). Families with
# these names stay in the same split on every platform, so a Linux train stream never uses val/test families.
_FROZEN_FONT_SPLIT = {
    "val": frozenset("ayuthaya baskerville chalkduster didot euphemia gill kohinoor lao monaco palatino plantagenet "
                     "sinhala stsong superclarendon".split()),
    "test": frozenset("bangla charter cochin devanagari geneva grantha khmer microsoft noteworthy sathu savoye "
                      "signpainter tahoma trattatello trebuchet".split()),
    "train": frozenset(
        "academy american andale apple applemyungjo aqua arial athelas avenir big bodoni bradley brush chalkboard "
        "comic copperplate courier din futura galvji georgia gujarati gurmukhi heiti helvetica herculanum hiragino "
        "hoefler impact inaimathi iowan kannada kefa krungthep lucida luminari malayalam marion marker menlo mukta "
        "myanmar new optima oriya papyrus party phosphate pt rockwell seravek sf shree silom skia snell songti stix "
        "sukhumvit system tamil telugu thonburi times verdana ys zapfino".split()),
}
# open (Linux / Colab) font families: the big multi-script families go to train, two FreeFont faces to val/test
OPEN_FONT_SPLIT = {"dejavu": "train", "liberation": "train", "noto": "train", "freesans": "train",
                   "droid": "train", "ubuntu": "train", "cantarell": "train",
                   "freemono": "val", "freeserif": "test"}

# symbol sources: (file name, encoding, code ranges); macOS fonts first (order matters for determinism)
_SYMBOL_SOURCES = (
    ("Webdings.ttf", "symb", ((0x21, 0xFF),)),
    ("Wingdings.ttf", "symb", ((0x21, 0xFF),)),
    ("Wingdings 2.ttf", "symb", ((0x21, 0xFF),)),
    ("Wingdings 3.ttf", "symb", ((0x21, 0xFF),)),
    ("ZapfDingbats.ttf", "unic", ((0x2701, 0x27BF),)),
    ("Apple Symbols.ttf", "unic", ((0x2190, 0x21FF), (0x25A0, 0x25FF), (0x2600, 0x26FF), (0x2700, 0x27BF))),
    ("Bodoni Ornaments.ttf", "unic", ((0x21, 0x7E),)),
    # Linux / open fonts
    ("NotoSansSymbols2-Regular.ttf", "unic", ((0x2190, 0x21FF), (0x25A0, 0x25FF), (0x2600, 0x26FF), (0x2700, 0x27BF),
                                               (0x1F300, 0x1F5FF), (0x1F650, 0x1F67F), (0x1F780, 0x1F7FF))),
    ("NotoSansSymbols-Regular.ttf", "unic", ((0x2190, 0x21FF), (0x2600, 0x26FF), (0x2700, 0x27BF))),
    ("DejaVuSans.ttf", "unic", ((0x2190, 0x21FF), (0x25A0, 0x25FF), (0x2600, 0x26FF), (0x2700, 0x27BF))),
    ("FreeSerif.ttf", "unic", ((0x2600, 0x26FF), (0x2700, 0x27BF))),
)
# colour emoji fonts: (file name, fixed bitmap sizes to try)
_EMOJI_FONTS = (
    ("Apple Color Emoji.ttc", (160, 96, 64)),
    ("NotoColorEmoji.ttf", (109, 136, 128, 96)),
    ("NotoColorEmoji-Regular.ttf", (109, 136, 128, 96)),
    ("Noto Color Emoji.ttf", (109, 136, 128, 96)),
)
_EMOJI_RANGES = ((0x1F300, 0x1F5FF), (0x1F600, 0x1F64F), (0x1F680, 0x1F6FF), (0x1F900, 0x1F9FF),
                 (0x1FA70, 0x1FAFF), (0x2600, 0x26FF))

DIGITS = "0123456789"
LATIN_UP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LATIN_LO = "abcdefghijklmnopqrstuvwxyz"
CYR_UP = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
CYR_LO = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
PUNCT = "#&@?%+=$!*<>/"
WORDS_LAT = (
    "MIFI", "DOE", "LASER", "OPTICS", "LIGHT", "WAVE", "PHASE", "HOLO", "LENS", "BEAM", "PHOTON", "FOCUS",
    "HELLO", "WORLD", "TEST", "NEURO", "NET", "AI", "OK", "YES", "NO", "CAT", "DOG", "SUN", "STAR", "MOON",
    "SKY", "TREE", "HOME", "LOVE", "ART", "CODE", "DATA", "PIXEL", "IMAGE", "Hello", "Test", "Laser",
    "Optics", "Wave", "Light", "Moscow", "MEPhI", "Nuclear", "Atom", "Zoom", "GPU", "CPU", "Python", "torch",
    "Fresnel", "Fourier", "Gabor", "Zernike", "Airy", "Bessel", "Lab", "Team", "Win", "Go", "Run", "Stop",
    "abc", "xyz", "ABC", "XYZ", "Hi", "Wow", "Cool", "Max", "Min", "Top", "Box", "Key", "Map", "Pi", "e",
)
WORDS_CYR = (
    "МИФИ", "НИЯУ", "ЛАЗЕР", "СВЕТ", "ВОЛНА", "ФАЗА", "ОПТИКА", "ПРИВЕТ", "ДОЭ", "МИР", "КОТ", "ДОМ",
    "ЛУЧ", "ЛИНЗА", "ФОТОН", "ФОКУС", "ТЕСТ", "НЕЙРО", "СЕТЬ", "ДА", "НЕТ", "СОЛНЦЕ", "ЗВЕЗДА", "ЛУНА",
    "Привет", "Свет", "Волна", "Лазер", "Москва", "Атом", "Физика", "Наука", "Код", "Данные", "Образ",
)

_TWO_PI = 2.0 * np.pi
_YY, _XX = np.mgrid[0:N, 0:N].astype(np.float32)
_FY = np.fft.fftfreq(N).astype(np.float32)[:, None]
_FX = np.fft.fftfreq(N).astype(np.float32)[None, :]
_FR = np.sqrt(_FX**2 + _FY**2)
_FR[0, 0] = 1.0 / N


def _hsplit(key: str) -> str:
    """Stable 70/15/15 hash split for individual source items (words, glyph code points, emoji)."""
    h = zlib.crc32(key.encode("utf-8")) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def _family_key(name: str) -> str:
    """Coarse design-group key: first word of the family name (so "Times"/"Times New Roman",
    "Helvetica"/"Helvetica Neue", "PT Sans"/"PT Serif", all "Hiragino ..." etc. share a split)."""
    w = name.lstrip(".").replace("-", " ").split()
    k = (w[0] if w else name).lower()
    return "stix" if k.startswith("stix") else k


# ---------------------------------------------------------------------------------------------------------
# Font bank (discovered once, cached to data/cache/, loaded lazily per process)
# ---------------------------------------------------------------------------------------------------------

def _font_files(dirs=None):
    """Font files in the (dir, recursive) list; sorted, de-duplicated."""
    files = set()
    for d, rec in (dirs if dirs is not None else font_dirs()):
        p = Path(os.path.expanduser(str(d)))
        if not p.is_dir():
            continue
        it = p.rglob("*") if rec else p.iterdir()
        try:
            for f in it:
                if f.suffix.lower() in (".ttf", ".ttc", ".otf") and f.is_file():
                    files.add(f)
        except OSError:
            continue
    return sorted(files)


def _render_probe(font, ch, size=64):
    img = Image.new("L", (size, size), 0)
    ImageDraw.Draw(img).text((8, 4), ch, font=font, fill=255)
    return np.asarray(img)


def _glyph_support(f):
    nd = _render_probe(f, "￿")

    def ok(chars):
        ims = [_render_probe(f, c) for c in chars]
        if any(im.sum() < 255 * 8 or np.array_equal(im, nd) for im in ims):
            return False
        return all(not np.array_equal(ims[i], ims[j]) for i in range(len(ims)) for j in range(i + 1, len(ims)))

    latin = ok("AB58gk")
    return latin, latin and ok("ЖЯЛД")


def _probe_face(path, idx):
    f = ImageFont.truetype(path, 40, index=idx)
    fam, sty = f.getname()
    latin, cyr = _glyph_support(f)
    return fam, sty, latin, cyr


def _discover(files):
    faces = []
    for path in files:
        low = path.name.lower()
        if any(x in low for x in _EXCLUDE_FONT_FILES):
            continue
        nmax = 16 if path.suffix.lower() == ".ttc" else 1
        for idx in range(nmax):
            try:
                fam, sty, latin, cyr = _probe_face(str(path), idx)
            except Exception:  # noqa: BLE001  (no more faces / unreadable font)
                break
            if latin and fam:
                faces.append({"path": str(path), "index": idx, "family": fam, "style": sty, "cyr": bool(cyr),
                              "key": _family_key(fam)})
    # symbol glyphs (sources located by file name; order of _SYMBOL_SOURCES is kept)
    by_name = {}
    for f in files:
        by_name.setdefault(f.name, str(f))
    symbols = []
    for fname, enc, ranges in _SYMBOL_SOURCES:
        path = by_name.get(fname)
        if path is None:
            continue
        try:
            f = ImageFont.truetype(path, 40, encoding=enc) if enc == "symb" else ImageFont.truetype(path, 40)
            nd = _render_probe(f, "￿" if enc != "symb" else "\u0001")
        except Exception:  # noqa: BLE001
            continue
        seen = []
        for a, b in ranges:
            for cp in range(a, b + 1):
                try:
                    im = _render_probe(f, chr(cp))
                except Exception:  # noqa: BLE001
                    continue
                if im.sum() < 255 * 20 or np.array_equal(im, nd):
                    continue
                if any(np.array_equal(im, s) for s in seen[-8:]):
                    continue
                seen.append(im)
                symbols.append({"path": path, "enc": enc, "cp": cp})
    # colour emoji font (bitmap fonts need one of their fixed strike sizes)
    emoji = None
    for fname, sizes in _EMOJI_FONTS:
        path = by_name.get(fname)
        if path is None:
            continue
        for size in sizes:
            try:
                f = ImageFont.truetype(path, size)
                im = Image.new("RGBA", (int(size * 1.3) + 20,) * 2, (0, 0, 0, 0))
                ImageDraw.Draw(im).text((10, 10), "\U0001F600", font=f, embedded_color=True)
                if im.getbbox() is not None and np.asarray(im)[..., 3].sum() > 255 * 400:
                    emoji = {"path": path, "size": size}
                    break
            except Exception:  # noqa: BLE001
                continue
        if emoji:
            break
    return {"faces": faces, "symbols": symbols, "emoji": emoji}


def _signature(files, dirs):
    s = "|".join(f"{f}:{f.stat().st_size}" for f in files if f.exists())
    d = "|".join(f"{a}:{int(r)}" for a, r in dirs)
    return f"v{_BANK_VERSION}-{sys.platform}-{zlib.crc32(d.encode()):08x}-{zlib.crc32(s.encode()):08x}"


def font_family_split(key: str) -> str:
    """Split of a font family key - a pure function of the name, identical on every platform:
    pinned names -> frozen macOS assignment (the shipped val/test sets were rendered with it) ->
    pinned open (Linux) families -> stable 70/15/15 crc32 hash of the name."""
    if key in PINNED_FONT_SPLIT:
        return PINNED_FONT_SPLIT[key]
    for s in SPLITS:
        if key in _FROZEN_FONT_SPLIT[s]:
            return s
    if key in OPEN_FONT_SPLIT:
        return OPEN_FONT_SPLIT[key]
    return _hsplit_key(key)


def _hsplit_key(key):
    h = zlib.crc32(key.encode("utf-8")) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


_PSEUDO_FACE = {"path": None, "index": 0, "family": "PIL default", "style": "Regular", "cyr": False,
                "key": "pil-default", "pseudo": True}


def _warn_once(key, msg, _seen=set()):  # noqa: B006
    if key not in _seen:
        _seen.add(key)
        warnings.warn(msg, stacklevel=3)


class FontBank:
    """Discovered fonts / symbol glyphs / emoji split by source, plus per-process render caches.

    Fonts are searched in font_dirs(): macOS system dirs, Linux dirs (/usr/share/fonts, /usr/local/share/fonts,
    ~/.fonts, ~/.local/share/fonts, recursive) or the DOE_FONT_DIRS override. Missing resources never crash:
    a split without fonts uses PIL's built-in font, missing emoji / symbol fonts disable those families.
    """

    _instance = None

    @classmethod
    def get(cls) -> "FontBank":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, dirs=None, cache_dir=CACHE_DIR):
        self.dirs = tuple(dirs) if dirs is not None else font_dirs()
        files = _font_files(self.dirs)
        sig = _signature(files, self.dirs)
        cache = Path(cache_dir) / f"fontbank_{sys.platform}_{sig.split('-')[2]}.json"
        data = None
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                ok = data.get("signature") == sig and all(Path(f["path"]).exists() for f in data["faces"])
                data = data if ok else None
            except Exception:  # noqa: BLE001
                data = None
        if data is None:
            data = _discover(files)
            data["signature"] = sig
            try:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                tmp = cache.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_text(json.dumps(data))
                tmp.replace(cache)
            except OSError:
                pass
        self.faces = data["faces"]
        for f in self.faces:
            f["key"] = _family_key(f["family"])
        self.split_of_family = {f["key"]: font_family_split(f["key"]) for f in self.faces}
        # split -> list of families -> list of faces
        self.families = {s: {} for s in SPLITS}
        for f in self.faces:
            s = self.split_of_family[f["key"]]
            self.families[s].setdefault(f["key"], []).append(f)
        self.family_keys = {s: sorted(self.families[s]) for s in SPLITS}
        self.pseudo = {s: not self.family_keys[s] for s in SPLITS}
        for s in SPLITS:
            if self.pseudo[s]:
                _warn_once(f"nofont-{s}-{sig}", f"doe.data: no fonts for split '{s}' in {self.dirs}; "
                           "text uses PIL's built-in font (font-level split disjointness not guaranteed)")
                self.families[s] = {"pil-default": [dict(_PSEUDO_FACE)]}
                self.family_keys[s] = ["pil-default"]
        self.cyr_faces = {s: [f for k in self.family_keys[s] for f in self.families[s][k] if f["cyr"]]
                          for s in SPLITS}
        self.symbols = {s: [g for g in data["symbols"] if _hsplit(f"sym:{Path(g['path']).name}:{g['cp']}") == s]
                        for s in SPLITS}
        self.emoji_font = data.get("emoji")
        self.emoji_cands = {s: [] for s in SPLITS}
        for a, b in _EMOJI_RANGES:
            for cp in range(a, b + 1):
                self.emoji_cands[_hsplit(f"emoji:{cp}")].append(cp)
        self.words = {s: [w for w in WORDS_LAT + WORDS_CYR if _hsplit(f"word:{w}") == s] for s in SPLITS}
        self._fonts = {}
        self._emoji = {}
        self._emoji_font = None
        self._emoji_nd = None
        self._glyph_cache = {}

    def split_fonts(self, split):
        """Sorted list of real font family keys of a split (the PIL built-in fallback is not listed)."""
        return [] if self.pseudo[split] else list(self.family_keys[split])

    def unavailable_families(self, split):
        """Families that cannot be rendered for a split on this machine (their weight is redistributed)."""
        out = set()
        if not self.emoji_font:
            out.add("emoji")
        if not self.symbols[split]:
            out.add("symbol")
        return out

    def summary(self):
        parts = []
        for s in SPLITS:
            nf = sum(len(v) for v in self.families[s].values())
            fam = "PIL-default" if self.pseudo[s] else f"{len(self.family_keys[s])} fam/{nf} faces"
            parts.append(f"{s}: {fam} (cyr {len(self.cyr_faces[s])}), sym {len(self.symbols[s])}")
        em = f"{Path(self.emoji_font['path']).name}@{self.emoji_font['size']}" if self.emoji_font else "none"
        return f"fonts[{'; '.join(parts)}] emoji={em}"

    def font(self, face, size):
        key = (face["path"], face["index"], size)
        f = self._fonts.get(key)
        if f is None:
            if len(self._fonts) > 4000:
                self._fonts.clear()
            if face["path"] is None:
                f = ImageFont.load_default(size)
            else:
                f = ImageFont.truetype(face["path"], size, index=face["index"])
            self._fonts[key] = f
        return f

    def sym_font(self, path, enc, size):
        key = (path, enc, size)
        f = self._fonts.get(key)
        if f is None:
            f = ImageFont.truetype(path, size, encoding=enc) if enc == "symb" else ImageFont.truetype(path, size)
            self._fonts[key] = f
        return f

    def random_face(self, split, rng, cyr=False):
        if cyr and self.cyr_faces[split]:
            faces = self.cyr_faces[split]
            return faces[int(rng.integers(len(faces)))]
        keys = self.family_keys[split]
        fam = self.families[split][keys[int(rng.integers(len(keys)))]]
        return fam[int(rng.integers(len(fam)))]

    def emoji(self, split, rng):
        """Gray emoji (float32 luminance premultiplied by alpha, alpha) at <=120 px, or None."""
        if not self.emoji_font:
            return None
        cands = self.emoji_cands[split]
        size = int(self.emoji_font["size"])
        canvas = (int(size * 1.3) + 20,) * 2
        for _ in range(30):
            cp = cands[int(rng.integers(len(cands)))]
            if cp in self._emoji:
                v = self._emoji[cp]
                if v is None:
                    continue
                return v
            try:
                if self._emoji_font is None:
                    self._emoji_font = ImageFont.truetype(self.emoji_font["path"], size)
                    nd = Image.new("RGBA", canvas, (0, 0, 0, 0))
                    ImageDraw.Draw(nd).text((10, 10), chr(0x10FFFD), font=self._emoji_font, embedded_color=True)
                    self._emoji_nd = np.asarray(nd)
                im = Image.new("RGBA", canvas, (0, 0, 0, 0))
                ImageDraw.Draw(im).text((10, 10), chr(cp), font=self._emoji_font, embedded_color=True)
                bbox = im.getbbox()
                if bbox is None or np.array_equal(np.asarray(im), self._emoji_nd):
                    self._emoji[cp] = None
                    continue
                im = im.crop(bbox)
                a = np.asarray(im, dtype=np.float32) / 255.0
                if a[..., 3].sum() < 400:
                    self._emoji[cp] = None
                    continue
                s = 120.0 / max(im.size)
                im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.LANCZOS)
                a = np.asarray(im, dtype=np.float32) / 255.0
                lum = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])
                v = (np.clip(lum, 0, 1), np.clip(a[..., 3], 0, 1))
            except Exception:  # noqa: BLE001
                v = None
            self._emoji[cp] = v
            if v is not None:
                return v
        return None


class ExternalBank:
    """Memory-mapped 128x128 grayscale caches of open datasets (data/external/cache), indexed per split."""

    _instance = None

    @classmethod
    def get(cls) -> "ExternalBank":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, root=EXT_DIR):
        self.img, self.mask, self.index = {}, {}, {}
        for name in tuple(PHOTO_SOURCES) + tuple(OBJECT_SOURCES) + tuple(TEXTURE_SOURCES):
            p = Path(root) / f"{name}_img.npy"
            if not p.exists():
                continue
            try:
                self.img[name] = np.load(p, mmap_mode="r")
                sp = np.load(Path(root) / f"{name}_split.npy")
                mp = Path(root) / f"{name}_mask.npy"
                if mp.exists():
                    self.mask[name] = np.load(mp, mmap_mode="r")
            except Exception:  # noqa: BLE001
                self.img.pop(name, None)
                continue
            self.index[name] = {s: np.nonzero(sp == _SPLIT_ID[s])[0] for s in SPLITS}
        self._warned = False

    def available(self, sources, split):
        return {k: w for k, w in sources.items() if k in self.index and len(self.index[k][split]) > 0}

    def pick(self, sources, split, rng):
        """(source name, uint8 image, uint8 mask or None) drawn from the split, or None if unavailable."""
        av = self.available(sources, split)
        if not av:
            if not self._warned:
                import warnings
                warnings.warn(f"external image cache missing in {EXT_DIR}; real-image families fall back to "
                              "synthetic ones (run scripts/prepare_external.py)")
                self._warned = True
            return None
        names = sorted(av)
        w = np.array([av[k] for k in names], np.float64)
        name = names[int(rng.choice(len(names), p=w / w.sum()))]
        idx = self.index[name][split]
        i = int(idx[int(rng.integers(len(idx)))])
        m = self.mask[name][i] if name in self.mask else None
        return name, np.asarray(self.img[name][i]), (None if m is None else np.asarray(m))

    def counts(self):
        return {k: {s: int(len(v[s])) for s in SPLITS} for k, v in self.index.items()}


# ---------------------------------------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------------------------------------

def _norm01(x):
    x = x.astype(np.float32, copy=False)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _fractal_noise(rng, beta):
    """Isotropic 1/f^beta noise (power spectrum ~ f^-beta... amplitude ~ f^-beta/2), normalised to [0,1]."""
    spec = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) * (_FR ** (-beta / 2.0))
    spec[0, 0] = 0
    return _norm01(np.fft.ifft2(spec).real)


def _noise1d(rng, n, beta):
    f = np.fft.rfftfreq(n)
    f[0] = 1.0 / n
    spec = (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)) * f ** (-beta / 2.0)
    spec[0] = 0
    return _norm01(np.fft.irfft(spec, n))


def _gauss(x, sigma):
    return ndimage.gaussian_filter(x, sigma, mode="nearest") if sigma > 0 else x


def _paste(canvas, patch, x0, y0, mode="max", alpha=None):
    """Paste float patch onto canvas at integer (x0, y0) with clipping. mode: max | over (needs alpha)."""
    h, w = patch.shape
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(N, x0 + w), min(N, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return canvas
    p = patch[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    c = canvas[cy0:cy1, cx0:cx1]
    if mode == "max":
        np.maximum(c, p, out=c)
    else:
        a = alpha[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        c *= 1.0 - a
        c += p * a if mode == "over_unpremult" else p
    return canvas


def _blur_big(x, sigma):
    """Gaussian blur; large sigmas are done at 1/4 resolution (fast)."""
    if sigma <= 6:
        return _gauss(x, sigma)
    small = x.reshape(N // 4, 4, N // 4, 4).mean((1, 3))
    small = ndimage.gaussian_filter(small, sigma / 4.0, mode="nearest")
    return ndimage.zoom(small, 4, order=1).astype(np.float32)


def _vignette(rng, strength=None):
    cx, cy = N / 2 + rng.normal(0, 8), N / 2 + rng.normal(0, 8)
    r2 = ((_XX - cx) ** 2 + (_YY - cy) ** 2) / (N * N / 4.0)
    k = rng.uniform(0.3, 0.9) if strength is None else strength
    return np.clip(1.0 - k * r2 ** rng.uniform(0.8, 1.6), 0, 1).astype(np.float32)


def _tone(x, rng, strength=1.0):
    """Random autocontrast / gamma / contrast / brightness for halftone images (float [0,1])."""
    if rng.random() < 0.5:
        lo, hi = np.percentile(x, (rng.uniform(0, 3), rng.uniform(97, 100)))
        if hi - lo > 0.05:
            x = (x - lo) / (hi - lo)
    x = np.clip(x, 0, 1) ** float(np.exp(rng.normal(0, 0.3 * strength)))
    c, b = float(np.exp(rng.normal(0, 0.2 * strength))), float(rng.normal(0, 0.06 * strength))
    return np.clip((x - 0.5) * c + 0.5 + b, 0, 1).astype(np.float32)


def _metal(x, rng):
    """S-curve contrast with highlights - a 'metallic' look."""
    for _ in range(int(rng.integers(1, 3))):
        x = 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0, 1))
    return x.astype(np.float32)


def _crop_resize(a_u8, rng, smin=0.6, size=(N, N), flip=True):
    """Random crop (zoom, aspect jitter) of a uint8 image resized to `size` -> float32 [0,1]."""
    H, W = a_u8.shape
    s = rng.uniform(smin, 1.0) if rng.random() < 0.75 else 1.0
    asp = float(np.exp(rng.normal(0, 0.08)))
    w, h = min(W, s * W * asp), min(H, s * H / asp)
    x0, y0 = rng.uniform(0, W - w), rng.uniform(0, H - h)
    im = Image.fromarray(np.ascontiguousarray(a_u8)).resize(size, Image.BILINEAR, box=(x0, y0, x0 + w, y0 + h))
    out = np.asarray(im, dtype=np.float32) / 255.0
    if flip and rng.random() < 0.5:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _rot(pts, ang):
    c, s = math.cos(ang), math.sin(ang)
    return [(x * c - y * s, x * s + y * c) for x, y in pts]


def _shape_points(kind, rng):
    """Unit-size polygon (radius ~1) for a shape kind; None for kinds drawn differently."""
    t = np.linspace(0, _TWO_PI, 72, endpoint=False)
    if kind == "ellipse":
        e = rng.uniform(0.45, 1.0)
        return list(zip(np.cos(t), e * np.sin(t)))
    if kind == "circle":
        return list(zip(np.cos(t), np.sin(t)))
    if kind == "triangle":
        a = np.array([-90, 30, 150]) * np.pi / 180 + rng.normal(0, 0.15, 3)
        return list(zip(np.cos(a), np.sin(a)))
    if kind == "square":
        return [(-0.75, -0.75), (0.75, -0.75), (0.75, 0.75), (-0.75, 0.75)]
    if kind == "rect":
        a = rng.uniform(0.25, 0.7)
        return [(-1, -a), (1, -a), (1, a), (-1, a)]
    if kind == "diamond":
        return [(0, -1), (0.65, 0), (0, 1), (-0.65, 0)]
    if kind == "polygon":
        k = int(rng.integers(5, 9))
        a = np.linspace(0, _TWO_PI, k, endpoint=False)
        return list(zip(np.cos(a), np.sin(a)))
    if kind == "star":
        k = int(rng.integers(4, 9))
        ri = rng.uniform(0.3, 0.6)
        a = np.linspace(0, _TWO_PI, 2 * k, endpoint=False) - np.pi / 2
        r = np.where(np.arange(2 * k) % 2 == 0, 1.0, ri)
        return list(zip(r * np.cos(a), r * np.sin(a)))
    if kind == "cross":
        w = rng.uniform(0.2, 0.4)
        return [(-w, -1), (w, -1), (w, -w), (1, -w), (1, w), (w, w), (w, 1), (-w, 1), (-w, w), (-1, w), (-1, -w),
                (-w, -w)]
    if kind == "arrow":
        s, hw = rng.uniform(0.15, 0.3), rng.uniform(0.45, 0.7)
        hx = rng.uniform(-0.1, 0.4)
        return [(-1, -s), (hx, -s), (hx, -hw), (1, 0), (hx, hw), (hx, s), (-1, s)]
    if kind == "heart":
        x = 16 * np.sin(t) ** 3
        y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        return list(zip(x / 17.0, (y + 2.5) / 17.0))
    if kind == "bolt":
        return [(-0.2, -1), (0.55, -1), (0.1, -0.15), (0.6, -0.15), (-0.35, 1), (-0.05, 0.1), (-0.5, 0.1)]
    if kind == "trapezoid":
        a = rng.uniform(0.2, 0.7)
        return [(-a, -0.6), (a, -0.6), (1, 0.6), (-1, 0.6)]
    if kind == "blob":
        r = 1 + 0.25 * np.real(sum((rng.normal() + 1j * rng.normal()) * np.exp(1j * k * t) / k for k in range(2, 6)))
        r = np.clip(r, 0.4, 1.4) / 1.3
        return list(zip(r * np.cos(t), r * np.sin(t)))
    return None


_SHAPE_KINDS = ("circle", "ellipse", "triangle", "square", "rect", "diamond", "polygon", "star", "cross", "arrow",
                "heart", "bolt", "trapezoid", "blob", "ring", "crescent", "roundrect", "pie")


class _SS:
    """Supersampled drawing surface (L mode, values 0..255) -> float32 N x N in [0,1]."""

    def __init__(self, ss=2, bg=0):
        self.ss = ss
        self.img = Image.new("L", (N * ss, N * ss), int(bg))
        self.d = ImageDraw.Draw(self.img)

    def done(self):
        im = self.img.reduce(self.ss) if self.ss > 1 else self.img
        return np.asarray(im, dtype=np.float32) / 255.0


def _draw_shape(sf: _SS, kind, cx, cy, r, ang, level, filled, width, rng, bg=0):
    s = sf.ss
    v = int(round(255 * level))
    w = max(1, int(round(width * s)))
    X, Y, R = cx * s, cy * s, r * s
    d = sf.d
    if kind == "ring":
        r2 = R * rng.uniform(0.45, 0.85) if filled else R
        if filled:
            d.ellipse([X - R, Y - R, X + R, Y + R], fill=v)
            d.ellipse([X - r2, Y - r2, X + r2, Y + r2], fill=bg)
        else:
            d.ellipse([X - R, Y - R, X + R, Y + R], outline=v, width=w)
        return
    if kind == "crescent":
        o = R * rng.uniform(0.3, 0.6)
        d.ellipse([X - R, Y - R, X + R, Y + R], fill=v if filled else None, outline=None if filled else v,
                  width=w)
        ox, oy = X + o * math.cos(ang), Y + o * math.sin(ang)
        d.ellipse([ox - R, oy - R, ox + R, oy + R], fill=bg)
        return
    if kind == "roundrect":
        a = rng.uniform(0.4, 1.0)
        box = [X - R, Y - R * a, X + R, Y + R * a]
        rad = R * a * rng.uniform(0.15, 0.9)
        if filled:
            d.rounded_rectangle(box, radius=rad, fill=v)
        else:
            d.rounded_rectangle(box, radius=rad, outline=v, width=w)
        return
    if kind == "pie":
        a0 = rng.uniform(0, 360)
        a1 = a0 + rng.uniform(60, 330)
        if filled:
            d.pieslice([X - R, Y - R, X + R, Y + R], a0, a1, fill=v)
        else:
            d.arc([X - R, Y - R, X + R, Y + R], a0, a1, fill=v, width=w)
        return
    pts = _shape_points(kind, rng)
    pts = [(X + px * R, Y + py * R) for px, py in _rot(pts, ang)]
    if filled:
        d.polygon(pts, fill=v)
    else:
        d.line(pts + [pts[0], pts[1]], fill=v, width=w, joint="curve")


# ---------------------------------------------------------------------------------------------------------
# Painter: one method per family, each takes its own rng
# ---------------------------------------------------------------------------------------------------------

class Painter:
    def __init__(self, split: str):
        self.split = split
        self.bank = FontBank.get()
        self.ext = ExternalBank.get()

    # ---------------- text helpers ----------------
    def _render_text(self, text, face, rng, size=96, outline=False, sym=None):
        font = self.bank.sym_font(*sym, size) if sym is not None else self.bank.font(face, size)
        spacing = int(size * rng.uniform(0.05, 0.3))
        sw = int(max(2, size * rng.uniform(0.03, 0.07))) if outline else 0
        tmp = ImageDraw.Draw(Image.new("L", (1, 1)))
        bb = tmp.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center", stroke_width=sw)
        bb = (math.floor(bb[0]), math.floor(bb[1]), math.ceil(bb[2]), math.ceil(bb[3]))
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if w <= 0 or h <= 0:
            return None
        pad = 4 + sw
        img = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
        dr = ImageDraw.Draw(img)
        if outline:
            dr.multiline_text((pad - bb[0], pad - bb[1]), text, font=font, fill=0, spacing=spacing, align="center",
                              stroke_width=sw, stroke_fill=255)
        else:
            dr.multiline_text((pad - bb[0], pad - bb[1]), text, font=font, fill=255, spacing=spacing,
                              align="center")
        return img

    @staticmethod
    def _random_angle(rng, p_zero=0.6, small=10.0, p_big=0.1, big=180.0):
        u = rng.random()
        if u < p_zero:
            return 0.0
        if u < 1 - p_big:
            return float(rng.normal(0, small))
        return float(rng.uniform(-big, big))

    def _fit(self, img, rng, target_h, max_w=None, angle=0.0, aspect=1.0, binarize=False):
        """Rotate / scale a hi-res L mask so its height ~ target_h px (and width <= max_w); float32 patch."""
        if angle:
            img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=0)
        bb = img.getbbox()
        if bb is None:
            return None
        img = img.crop(bb)
        w, h = img.size
        max_w = max_w or N - 2
        sc = min(target_h / h, max_w / (w * aspect), (N - 2) / h)
        nw, nh = max(1, round(w * sc * aspect)), max(1, round(h * sc))
        img = img.resize((nw, nh), Image.LANCZOS if sc < 1 else Image.BICUBIC)
        a = np.clip(np.asarray(img, dtype=np.float32) / 255.0, 0, 1)
        if binarize:
            a = (a > 0.5).astype(np.float32)
        return a

    @staticmethod
    def _position(rng, w, h, center_p=0.55, jitter=6.0, allow_crop=0.05):
        if rng.random() < center_p:
            x0 = (N - w) / 2 + rng.normal(0, jitter)
            y0 = (N - h) / 2 + rng.normal(0, jitter)
        else:
            x0 = rng.uniform(min(0, N - w), max(0, N - w))
            y0 = rng.uniform(min(0, N - h), max(0, N - h))
        if rng.random() < allow_crop:
            x0 += rng.normal(0, 0.25 * w)
            y0 += rng.normal(0, 0.25 * h)
        return int(round(x0)), int(round(y0))

    def _text_layer(self, rng, text, face, target_h, max_w=None, angle=0.0, outline=False, center_p=0.55,
                    sym=None, binarize=None, aspect=None):
        img = self._render_text(text, face, rng, outline=outline, sym=sym)
        if img is None:
            return np.zeros((N, N), np.float32)
        if aspect is None:
            aspect = float(np.exp(rng.normal(0, 0.12))) if rng.random() < 0.3 else 1.0
        if binarize is None:
            binarize = rng.random() < 0.3
        a = self._fit(img, rng, target_h, max_w, angle, aspect, binarize)
        canvas = np.zeros((N, N), np.float32)
        if a is None:
            return canvas
        x0, y0 = self._position(rng, a.shape[1], a.shape[0], center_p)
        return _paste(canvas, a, x0, y0)

    def _fill_texture(self, rng, mask):
        """Fill a mask with a flat level (mostly), a gradient or a noise texture."""
        u = rng.random()
        if u < 0.82:
            lvl = 1.0 if rng.random() < 0.7 else rng.uniform(0.35, 1.0)
            return mask * lvl
        if u < 0.93:
            g = self._gradient_raw(rng)
            return mask * (0.3 + 0.7 * g)
        return mask * (0.35 + 0.65 * _fractal_noise(rng, rng.uniform(1.5, 3.0)))

    def random_char(self, rng, cyr_ok):
        u = rng.random()
        if u < 0.40:
            return DIGITS[int(rng.integers(10))], False
        if u < 0.68:
            return LATIN_UP[int(rng.integers(26))], False
        if u < 0.78:
            return LATIN_LO[int(rng.integers(26))], False
        if u < 0.95 and cyr_ok:
            s = CYR_UP if rng.random() < 0.8 else CYR_LO
            return s[int(rng.integers(len(s)))], True
        return PUNCT[int(rng.integers(len(PUNCT)))], False

    # ---------------- families ----------------
    def glyph(self, rng, face=None, ch=None):
        cyr = rng.random() < 0.18
        if face is None:
            face = self.bank.random_face(self.split, rng, cyr=cyr)
        if ch is None:
            ch, _ = self.random_char(rng, face["cyr"])
        th = rng.uniform(0.35, 0.92) * N if rng.random() < 0.8 else rng.uniform(0.12, 0.35) * N
        ang = self._random_angle(rng)
        m = self._text_layer(rng, ch, face, th, angle=ang, outline=rng.random() < 0.08)
        return self._fill_texture(rng, m)

    def random_text(self, rng, cyr_ok):
        u = rng.random()
        words = self.bank.words[self.split]
        if u < 0.35 and words:
            for _ in range(10):
                w = words[int(rng.integers(len(words)))]
                if cyr_ok or not any(c in CYR_UP + CYR_LO for c in w):
                    return w
        if u < 0.6:
            k = int(rng.integers(2, 7))
            if cyr_ok and rng.random() < 0.3:
                alpha = CYR_UP if rng.random() < 0.7 else CYR_LO
            else:
                alpha = [LATIN_UP, LATIN_LO, LATIN_UP + DIGITS, DIGITS][int(rng.integers(4))]
            return "".join(alpha[int(rng.integers(len(alpha)))] for _ in range(k))
        if u < 0.78:
            v = rng.random()
            if v < 0.5:
                return str(int(rng.integers(0, 10 ** int(rng.integers(1, 5)))))
            if v < 0.8:
                return f"{rng.uniform(0, 100):.{int(rng.integers(1, 3))}f}"
            return f"{int(rng.integers(1, 31)):02d}.{int(rng.integers(1, 13)):02d}"
        if u < 0.9:
            a = self.random_text(rng, cyr_ok) if rng.random() < 0.5 else str(int(rng.integers(0, 1000)))
            b = "".join(LATIN_UP[int(rng.integers(26))] for _ in range(int(rng.integers(2, 5))))
            return a[:6] + "\n" + b
        k = int(rng.integers(2, 5))
        src = DIGITS if rng.random() < 0.5 else LATIN_UP
        return " ".join(src[int(rng.integers(len(src)))] for _ in range(k))

    def word(self, rng, face=None):
        cyr = rng.random() < 0.2
        if face is None:
            face = self.bank.random_face(self.split, rng, cyr=cyr)
        text = self.random_text(rng, face["cyr"])
        max_w = rng.uniform(0.5, 0.97) * N
        th = rng.uniform(0.2, 0.75) * N
        ang = self._random_angle(rng, p_zero=0.7, small=8.0, p_big=0.05, big=90.0)
        m = self._text_layer(rng, text, face, th, max_w=max_w, angle=ang, outline=rng.random() < 0.06,
                             center_p=0.6)
        return self._fill_texture(rng, m)

    def multiglyph(self, rng, face=None):
        same = rng.random() < 0.5
        if face is None:
            face = self.bank.random_face(self.split, rng)
        out = np.zeros((N, N), np.float32)
        for _ in range(int(rng.integers(2, 7))):
            f = face if same else self.bank.random_face(self.split, rng)
            ch, _ = self.random_char(rng, f["cyr"])
            m = self._text_layer(rng, ch, f, rng.uniform(0.15, 0.5) * N, angle=self._random_angle(rng, 0.5, 20),
                                 center_p=0.0)
            np.maximum(out, m * (1.0 if rng.random() < 0.6 else rng.uniform(0.3, 1.0)), out=out)
        return out

    def shapes(self, rng):
        sf = _SS(2 if rng.random() < 0.8 else 1)
        if rng.random() < 0.2:
            n = 1
        else:
            n = int(rng.choice([1, 2, 2, 3, 3, 4, 5, 6, 8]))
        for i in range(n):
            kind = _SHAPE_KINDS[int(rng.integers(len(_SHAPE_KINDS)))]
            if n == 1:
                r = rng.uniform(0.2, 0.47) * N
                cx, cy = N / 2 + rng.normal(0, 8), N / 2 + rng.normal(0, 8)
            else:
                r = rng.uniform(0.07, 0.3) * N
                cx, cy = rng.uniform(r * 0.6, N - r * 0.6), rng.uniform(r * 0.6, N - r * 0.6)
            level = 1.0 if rng.random() < 0.55 else rng.uniform(0.25, 1.0)
            filled = rng.random() < 0.65
            width = rng.choice([1, 1.5, 2, 3, 4, 6])
            _draw_shape(sf, kind, cx, cy, r, rng.uniform(0, _TWO_PI) if rng.random() < 0.7 else 0.0, level, filled,
                        width, rng)
        return sf.done()

    def lineart(self, rng):
        sf = _SS(2 if rng.random() < 0.7 else 1)
        s = sf.ss
        d = sf.d
        kinds = int(rng.integers(1, 4))
        for _ in range(kinds):
            k = rng.integers(8)
            w = max(1, int(round(rng.choice([1, 1, 1.5, 2, 2, 3, 4]) * s)))
            v = 255 if rng.random() < 0.7 else int(rng.integers(90, 256))
            if k == 0:  # random polyline
                n = int(rng.integers(3, 12))
                pts = [tuple(p) for p in rng.uniform(8, N - 8, (n, 2)) * s]
                d.line(pts, fill=v, width=w, joint="curve")
            elif k == 1:  # cubic Bezier curves
                for _ in range(int(rng.integers(1, 4))):
                    P = rng.uniform(4, N - 4, (4, 2)) * s
                    t = np.linspace(0, 1, 80)[:, None]
                    B = ((1 - t) ** 3) * P[0] + 3 * ((1 - t) ** 2) * t * P[1] + 3 * (1 - t) * t * t * P[2] + t ** 3 * P[3]
                    d.line([tuple(p) for p in B], fill=v, width=w, joint="curve")
            elif k == 2:  # spiral
                turns = rng.uniform(2, 6)
                t = np.linspace(0, turns * _TWO_PI, 400)
                rmax = rng.uniform(0.25, 0.48) * N
                r = rmax * t / t[-1]
                cx, cy = N / 2 + rng.normal(0, 6), N / 2 + rng.normal(0, 6)
                d.line([((cx + a * math.cos(b)) * s, (cy + a * math.sin(b)) * s) for a, b in zip(r, t)], fill=v,
                       width=w)
            elif k == 3:  # sine waves
                for _ in range(int(rng.integers(1, 5))):
                    x = np.linspace(0, N, 200)
                    y = rng.uniform(15, N - 15) + rng.uniform(4, 25) * np.sin(x * rng.uniform(0.03, 0.25) + rng.uniform(0, 6))
                    d.line([(a * s, b * s) for a, b in zip(x, y)], fill=v, width=w)
            elif k == 4:  # outlines of shapes (concentric or scattered)
                for _ in range(int(rng.integers(1, 5))):
                    kind = _SHAPE_KINDS[int(rng.integers(len(_SHAPE_KINDS)))]
                    _draw_shape(sf, kind, rng.uniform(25, N - 25), rng.uniform(25, N - 25), rng.uniform(10, 50),
                                rng.uniform(0, _TWO_PI), v / 255, False, w / s, rng)
            elif k == 5:  # straight segments
                for _ in range(int(rng.integers(3, 15))):
                    p = rng.uniform(0, N, 4) * s
                    d.line([(p[0], p[1]), (p[2], p[3])], fill=v, width=w)
            elif k == 6:  # house / stick drawing
                cx, cy, r = rng.uniform(40, 88), rng.uniform(45, 85), rng.uniform(20, 38)
                pts = [(cx - r, cy), (cx, cy - r), (cx + r, cy), (cx + r, cy + r), (cx - r, cy + r), (cx - r, cy)]
                d.line([(a * s, b * s) for a, b in pts], fill=v, width=w, joint="curve")
                d.rectangle([(cx - r * 0.2) * s, (cy + r * 0.4) * s, (cx + r * 0.2) * s, (cy + r) * s], outline=v,
                            width=w)
            else:  # zigzag
                n = int(rng.integers(4, 14))
                x = np.linspace(rng.uniform(0, 30), rng.uniform(98, N), n)
                y0 = rng.uniform(20, N - 20)
                amp = rng.uniform(5, 30)
                y = y0 + amp * (np.arange(n) % 2 * 2 - 1)
                d.line([(a * s, b * s) for a, b in zip(x, y)], fill=v, width=w)
        return sf.done()

    def logo(self, rng):
        t = rng.integers(5)
        face = self.bank.random_face(self.split, rng, cyr=rng.random() < 0.15)
        sf = _SS(2)
        cx, cy = N / 2 + rng.normal(0, 5), N / 2 + rng.normal(0, 5)
        R = rng.uniform(0.3, 0.47) * N
        lvl = 1.0 if rng.random() < 0.7 else rng.uniform(0.4, 1.0)
        badge = ("circle", "square", "polygon", "roundrect", "diamond", "ring", "star")[int(rng.integers(7))]
        if t <= 1:  # badge with cut-out (or contrasting) text
            _draw_shape(sf, badge if badge != "ring" else "circle", cx, cy, R, 0.0 if rng.random() < 0.6 else rng.uniform(0, 6.3),
                        lvl, True, 2, rng)
            base = sf.done()
            k = int(rng.integers(1, 4))
            txt = "".join(LATIN_UP[int(rng.integers(26))] for _ in range(k)) if rng.random() < 0.7 else \
                self.random_text(rng, face["cyr"])[:4]
            m = self._render_text(txt, face, rng)
            a = self._fit(m, rng, R * rng.uniform(0.6, 1.1), max_w=R * 1.3) if m is not None else None
            out = base
            if a is not None:
                tm = _paste(np.zeros((N, N), np.float32), a, int(cx - a.shape[1] / 2), int(cy - a.shape[0] / 2))
                tv = 0.0 if rng.random() < 0.75 else rng.uniform(0, 0.5) * lvl
                out = base * (1 - tm) + tv * tm
            if rng.random() < 0.3:
                ring = _SS(2)
                rr = R * rng.uniform(1.05, 1.15)
                ring.d.ellipse([(cx - rr) * 2, (cy - rr) * 2, (cx + rr) * 2, (cy + rr) * 2], outline=int(255 * lvl),
                               width=int(rng.integers(2, 6)))
                out = np.maximum(out, ring.done())
            return out
        if t == 2:  # outline badge + text inside
            _draw_shape(sf, badge, cx, cy, R, 0.0, lvl, False, rng.uniform(1.5, 5), rng)
            base = sf.done()
            txt = self.random_text(rng, face["cyr"])[:5]
            m = self._render_text(txt.replace("\n", ""), face, rng)
            a = self._fit(m, rng, R * 0.7, max_w=R * 1.3) if m is not None else None
            if a is not None:
                base = _paste(base, a * lvl, int(cx - a.shape[1] / 2), int(cy - a.shape[0] / 2))
            return base
        if t == 3:  # icon (shape or dingbat) above a word
            out = np.zeros((N, N), np.float32)
            top = N * rng.uniform(0.38, 0.6)
            syms = self.bank.symbols[self.split]
            if syms and rng.random() < 0.5:
                g = syms[int(rng.integers(len(syms)))]
                m = self._render_text(chr(g["cp"]), None, rng, sym=(g["path"], g["enc"]))
                a = self._fit(m, rng, top - 8) if m is not None else None
                if a is not None:
                    _paste(out, a, int(N / 2 - a.shape[1] / 2), int(4 + (top - 4 - a.shape[0]) / 2))
            else:
                kind = _SHAPE_KINDS[int(rng.integers(len(_SHAPE_KINDS)))]
                _draw_shape(sf, kind, N / 2, top / 2 + 2, top / 2 - 4, 0.0, 1.0, rng.random() < 0.7, 3, rng)
                out = sf.done()
            txt = self.random_text(rng, face["cyr"]).replace("\n", " ")[:7]
            m = self._render_text(txt, face, rng)
            a = self._fit(m, rng, (N - top) * 0.75, max_w=N * 0.92) if m is not None else None
            if a is not None:
                _paste(out, a, int(N / 2 - a.shape[1] / 2), int(top + (N - top - a.shape[0]) / 2))
            return out * lvl
        if t == 4:  # monogram: two overlapping letters / concentric shapes
            if rng.random() < 0.5:
                out = np.zeros((N, N), np.float32)
                for j in range(2):
                    ch = LATIN_UP[int(rng.integers(26))]
                    m = self._render_text(ch, face, rng)
                    a = self._fit(m, rng, N * rng.uniform(0.5, 0.85)) if m is not None else None
                    if a is not None:
                        x0 = int(N / 2 - a.shape[1] / 2 + (j - 0.5) * a.shape[1] * rng.uniform(0.2, 0.6))
                        _paste(out, a * (1.0 if j == 0 else rng.uniform(0.5, 1.0)), x0, int(N / 2 - a.shape[0] / 2))
                return out
            r = R
            for j in range(int(rng.integers(2, 5))):
                kind = ("circle", "square", "polygon", "diamond", "star")[int(rng.integers(5))]
                _draw_shape(sf, kind, cx, cy, r, 0.0, 1.0 if j % 2 == 0 else 0.0, True, 2, rng)
                r *= rng.uniform(0.5, 0.8)
            return sf.done() * lvl
        return self.shapes(rng)

    def symbol(self, rng):
        syms = self.bank.symbols[self.split]
        if not syms:
            return self.shapes(rng)
        out = np.zeros((N, N), np.float32)
        k = 1 if rng.random() < 0.75 else int(rng.integers(2, 5))
        for _ in range(k):
            g = syms[int(rng.integers(len(syms)))]
            th = rng.uniform(0.4, 0.92) * N if k == 1 else rng.uniform(0.2, 0.5) * N
            m = self._text_layer(rng, chr(g["cp"]), None, th, angle=self._random_angle(rng, 0.75, 10, 0.05),
                                 center_p=0.7 if k == 1 else 0.0, sym=(g["path"], g["enc"]))
            np.maximum(out, m, out=out)
        return self._fill_texture(rng, out)

    def emoji(self, rng):
        out = np.zeros((N, N), np.float32) + (0.0 if rng.random() < 0.8 else rng.uniform(0, 0.5))
        k = 1 if rng.random() < 0.8 else int(rng.integers(2, 4))
        for _ in range(k):
            e = self.bank.emoji(self.split, rng)
            if e is None:
                return self.shapes(rng)
            lum, alpha = e
            sz = rng.uniform(0.45, 0.98) * N if k == 1 else rng.uniform(0.25, 0.55) * N
            sc = sz / max(lum.shape)
            nh, nw = max(1, round(lum.shape[0] * sc)), max(1, round(lum.shape[1] * sc))
            L = np.asarray(Image.fromarray(lum).resize((nw, nh), Image.BILINEAR))
            A = np.asarray(Image.fromarray(alpha).resize((nw, nh), Image.BILINEAR))
            x0, y0 = self._position(rng, nw, nh, center_p=0.7 if k == 1 else 0.0)
            _paste(out, L * A, x0, y0, mode="over", alpha=A)
        return np.clip(out, 0, 1)

    def _gradient_raw(self, rng):
        u = rng.random()
        if u < 0.4:
            a = rng.uniform(0, _TWO_PI)
            g = _XX * math.cos(a) + _YY * math.sin(a)
        elif u < 0.7:
            cx, cy = rng.uniform(0, N, 2)
            g = -np.sqrt((_XX - cx) ** 2 + (_YY - cy) ** 2)
        elif u < 0.85:
            cx, cy = rng.uniform(N * 0.2, N * 0.8, 2)
            g = np.mod(np.arctan2(_YY - cy, _XX - cx) + rng.uniform(0, _TWO_PI), _TWO_PI)
        else:
            g = _fractal_noise(rng, rng.uniform(3.5, 4.5))
        g = _norm01(g)
        if rng.random() < 0.3:
            g = g ** rng.uniform(0.4, 2.5)
        return g

    def gradient(self, rng):
        u = rng.random()
        if u < 0.45:
            return self._gradient_raw(rng)
        if u < 0.7:  # Gaussian blobs
            out = np.zeros((N, N), np.float32)
            for _ in range(int(rng.integers(1, 9))):
                cx, cy = rng.uniform(0, N, 2)
                sx, sy = rng.uniform(4, 35, 2)
                a = rng.uniform(0.3, 1.0)
                out += a * np.exp(-((_XX - cx) ** 2 / (2 * sx * sx) + (_YY - cy) ** 2 / (2 * sy * sy)))
            return out / max(1.0, float(out.max())) if rng.random() < 0.5 else _norm01(out)
        if u < 0.85:  # gray wedge / step bars
            k = int(rng.integers(3, 17))
            levels = np.linspace(0, 1, k)
            if rng.random() < 0.3:
                levels = rng.permutation(levels)
            idx = np.minimum((_XX if rng.random() < 0.5 else _YY) * k / N, k - 1).astype(int)
            g = levels[idx].astype(np.float32)
            return g if rng.random() < 0.8 else g.T
        # gradient restricted to a shape
        sf = _SS(2)
        _draw_shape(sf, _SHAPE_KINDS[int(rng.integers(len(_SHAPE_KINDS)))], N / 2 + rng.normal(0, 8),
                    N / 2 + rng.normal(0, 8), rng.uniform(0.25, 0.47) * N, rng.uniform(0, 6.3), 1.0, True, 2, rng)
        return sf.done() * (0.15 + 0.85 * self._gradient_raw(rng))

    def texture(self, rng):
        u = rng.random()
        if u < 0.3:
            return _fractal_noise(rng, rng.uniform(1.2, 3.2))
        if u < 0.5:
            n = _fractal_noise(rng, rng.uniform(1.8, 3.5))
            return (n > np.quantile(n, rng.uniform(0.3, 0.85))).astype(np.float32)
        if u < 0.6:
            n = _fractal_noise(rng, rng.uniform(2.5, 3.5))
            return (0.5 + 0.5 * np.sin(n * rng.uniform(10, 40))).astype(np.float32)
        if u < 0.7:
            n = _fractal_noise(rng, rng.uniform(1.5, 3.0))
            return 1.0 - np.abs(2 * n - 1) ** rng.uniform(0.3, 1.0)
        if u < 0.85:
            return _norm01(_gauss(rng.standard_normal((N, N)).astype(np.float32), rng.uniform(0.7, 6)))
        # Voronoi cells
        k = int(rng.integers(6, 50))
        pts = rng.uniform(0, N, (k, 2)).astype(np.float32)
        d = (_XX[None] - pts[:, 0, None, None]) ** 2 + (_YY[None] - pts[:, 1, None, None]) ** 2
        lab = d.argmin(0)
        vals = rng.uniform(0, 1, k).astype(np.float32)
        out = vals[lab]
        if rng.random() < 0.4:  # cell borders
            ds = np.sort(d, axis=0)
            out = (np.sqrt(ds[1]) - np.sqrt(ds[0]) < rng.uniform(1, 3)).astype(np.float32)
        return out

    def synphoto(self, rng):
        u = rng.random()
        if u < 0.3:
            out = self._dead_leaves(rng)
        elif u < 0.55:
            out = self._landscape(rng)
        elif u < 0.8:
            out = self._still_life(rng)
        else:
            out = self._portrait(rng)
        if rng.random() < 0.5:
            out = out + rng.uniform(0.01, 0.05) * (_fractal_noise(rng, rng.uniform(0.5, 2.0)) - 0.5)
        return np.clip(out, 0, 1)

    def _dead_leaves(self, rng):
        sf = _SS(1, bg=int(rng.integers(0, 256)))
        rmin, rmax = rng.uniform(1.5, 4), rng.uniform(20, 60)
        for _ in range(int(rng.integers(60, 260))):
            uu = rng.random()
            r = (rmin ** -2 - uu * (rmin ** -2 - rmax ** -2)) ** -0.5
            x, y = rng.uniform(-r, N + r, 2)
            v = int(rng.integers(0, 256))
            if rng.random() < 0.8:
                sf.d.ellipse([x - r, y - r, x + r, y + r], fill=v)
            else:
                sf.d.rectangle([x - r, y - r * rng.uniform(0.3, 1), x + r, y + r], fill=v)
        return _gauss(sf.done(), rng.uniform(0, 1.2))

    def _landscape(self, rng):
        top, hor = rng.uniform(0.5, 1.0), rng.uniform(0.2, 0.8)
        sky = top + (hor - top) * (_YY / N)
        if rng.random() < 0.6:  # sun / moon
            cx, cy, r = rng.uniform(15, N - 15), rng.uniform(10, N * 0.45), rng.uniform(5, 18)
            sky = np.maximum(sky, np.clip((r - np.sqrt((_XX - cx) ** 2 + (_YY - cy) ** 2)) / 2.0, 0, 1)
                             * rng.uniform(0.85, 1.0))
        out = sky.astype(np.float32)
        nl = int(rng.integers(1, 4))
        for i in range(nl):
            base = N * (0.35 + 0.18 * i + rng.uniform(-0.05, 0.08))
            prof = base - rng.uniform(10, 45) * _noise1d(rng, N, rng.uniform(1.5, 3.0))
            lvl = rng.uniform(0.05, 0.6) * (1 - 0.2 * i)
            m = (_YY > prof[None, :]).astype(np.float32)
            m = _gauss(m, 0.7)
            tex = lvl * (0.75 + 0.5 * _fractal_noise(rng, 2.0))
            out = out * (1 - m) + tex * m
        return _gauss(out, rng.uniform(0, 1.0))

    def _shaded_ellipse(self, rng, out, cx, cy, a, b, alb, light, soft):
        u = (_XX - cx) / a
        v = (_YY - cy) / b
        r2 = u * u + v * v
        inside = np.clip((1 - r2) * min(a, b) / (2 * soft), 0, 1)
        z = np.sqrt(np.clip(1 - r2, 0, 1))
        sh = np.clip(light[0] * u + light[1] * v + light[2] * z, 0, 1)
        val = alb * (0.25 + 0.75 * sh)
        return out * (1 - inside) + val * inside

    def _still_life(self, rng):
        out = (rng.uniform(0.05, 0.6) + rng.uniform(-0.3, 0.3) * (self._gradient_raw(rng) - 0.5)).astype(np.float32)
        out = out + 0.08 * (_fractal_noise(rng, 2.5) - 0.5)
        L = rng.normal(0, 1, 3)
        L[2] = abs(L[2]) + 0.8
        L /= np.linalg.norm(L)
        for _ in range(int(rng.integers(1, 6))):
            cx, cy = rng.uniform(15, N - 15, 2)
            a, b = rng.uniform(8, 40), rng.uniform(8, 40)
            if rng.random() < 0.5:  # drop shadow
                sh = np.exp(-(((_XX - cx - 6 * L[0] * -1) / a) ** 2 + ((_YY - cy + 6 * L[1]) / b) ** 2) * 1.5)
                out = out * (1 - 0.4 * sh)
            if rng.random() < 0.7:
                out = self._shaded_ellipse(rng, out, cx, cy, a, b, rng.uniform(0.3, 1.0), L, rng.uniform(0.5, 2.5))
            else:  # box with linear shading
                m = ((np.abs(_XX - cx) < a) & (np.abs(_YY - cy) < b)).astype(np.float32)
                m = _gauss(m, rng.uniform(0.5, 1.5))
                g = rng.uniform(0.3, 1.0) * (0.6 + 0.4 * _norm01(_XX * L[0] + _YY * L[1]))
                out = out * (1 - m) + g * m
        return _gauss(out, rng.uniform(0.3, 1.2))

    def _portrait(self, rng):
        bg = (rng.uniform(0, 0.7) + 0.3 * (self._gradient_raw(rng) - 0.5)).astype(np.float32)
        L = np.array([rng.normal(0, 0.6), rng.normal(-0.3, 0.4), 1.0])
        L /= np.linalg.norm(L)
        cx, cy = N / 2 + rng.normal(0, 8), N * rng.uniform(0.38, 0.5)
        a, b = rng.uniform(24, 36), rng.uniform(32, 46)
        out = self._shaded_ellipse(rng, bg, cx, N + 10, a * 2.2, b * 1.2, rng.uniform(0.2, 0.9), L, 2)  # shoulders
        hair = rng.uniform(0.0, 1.0)
        out = self._shaded_ellipse(rng, out, cx, cy - b * 0.15, a * 1.12, b * 1.05, hair, L, 2)
        skin = rng.uniform(0.4, 1.0)
        out = self._shaded_ellipse(rng, out, cx, cy + b * 0.08, a, b * 0.95, skin, L, 1.5)
        eo, ey = a * rng.uniform(0.3, 0.45), cy - b * 0.05
        for sgn in (-1, 1):
            out = self._shaded_ellipse(rng, out, cx + sgn * eo, ey, a * 0.14, b * 0.08, skin * 0.15, L, 0.8)
        mouth = (np.abs(((_XX - cx) / (a * 0.35)) ** 2 + ((_YY - cy - b * 0.5) / (b * 0.08)) ** 2 - 1) < 0.5)
        out = np.where(mouth & (_YY > cy + b * 0.48), out * 0.5, out)
        return _gauss(out.astype(np.float32), rng.uniform(0.5, 1.5))

    def pattern(self, rng):
        u = rng.random()
        ang = rng.uniform(0, np.pi) if rng.random() < 0.4 else 0.0
        X = _XX * math.cos(ang) + _YY * math.sin(ang)
        Y = -_XX * math.sin(ang) + _YY * math.cos(ang)
        p = float(rng.choice([4, 6, 8, 10, 12, 16, 20, 24, 32, 42, 64]))
        if u < 0.16:  # checkerboard
            out = ((np.floor(X / p) + np.floor(Y / p)) % 2).astype(np.float32)
        elif u < 0.32:  # stripes
            duty = rng.uniform(0.2, 0.8)
            out = (np.mod(X, p) < duty * p).astype(np.float32)
        elif u < 0.42:  # sinusoidal grating
            out = 0.5 + 0.5 * np.cos(_TWO_PI * X / p + rng.uniform(0, 6.3))
            if rng.random() < 0.3:
                out = out * (0.5 + 0.5 * np.cos(_TWO_PI * Y / float(rng.choice([8, 16, 32]))))
        elif u < 0.52:  # concentric rings
            cx, cy = N / 2 + rng.normal(0, 10, 2)
            r = np.sqrt((_XX - cx) ** 2 + (_YY - cy) ** 2)
            out = (np.mod(r, p) < p * rng.uniform(0.3, 0.7)).astype(np.float32) if rng.random() < 0.7 else \
                0.5 + 0.5 * np.cos(_TWO_PI * r / p)
        elif u < 0.6:  # Siemens star
            cx, cy = N / 2 + rng.normal(0, 5, 2)
            k = int(rng.integers(4, 24))
            th = np.arctan2(_YY - cy, _XX - cx)
            out = (np.mod(np.floor(th * k / np.pi), 2)).astype(np.float32)
            out *= np.sqrt((_XX - cx) ** 2 + (_YY - cy) ** 2) < rng.uniform(0.3, 0.5) * N
        elif u < 0.75:  # QR-like module grid
            out = self._qr(rng)
        elif u < 0.85:  # dot lattice
            r = p * rng.uniform(0.15, 0.45)
            dx = np.mod(X, p) - p / 2
            dy = np.mod(Y, p) - p / 2
            out = np.clip(r + 0.5 - np.sqrt(dx * dx + dy * dy), 0, 1).astype(np.float32)
        elif u < 0.93:  # grid lines
            w = rng.uniform(1, max(1.5, p / 4))
            out = ((np.mod(X, p) < w) | (np.mod(Y, p) < w)).astype(np.float32)
        else:  # bar target (groups of 3 bars, decreasing width)
            sf = _SS(1)
            x = 6.0
            w = rng.uniform(5, 9)
            while x < N - 6 and w >= 1:
                for j in range(3):
                    sf.d.rectangle([x, 10 + rng.uniform(0, 10), x + w - 1, N - 10 - rng.uniform(0, 10)], fill=255)
                    x += 2 * w
                x += w
                w *= 0.7
            out = sf.done()
            if rng.random() < 0.5:
                out = out.T.copy()
        out = out.astype(np.float32)
        if rng.random() < 0.35:  # window it into a shape / region
            sf = _SS(2)
            _draw_shape(sf, ("circle", "square", "roundrect", "polygon", "star", "heart")[int(rng.integers(6))],
                        N / 2 + rng.normal(0, 10), N / 2 + rng.normal(0, 10), rng.uniform(0.25, 0.46) * N,
                        rng.uniform(0, 6.3), 1.0, True, 2, rng)
            out = out * sf.done()
        return out

    def _qr(self, rng):
        m = int(rng.choice([11, 15, 21, 25, 29, 33, 41]))
        px = max(1, int((N * rng.uniform(0.6, 0.95)) // m))
        bits = (rng.random((m, m)) < rng.uniform(0.35, 0.65)).astype(np.float32)
        if m >= 21:
            fp = np.zeros((7, 7), np.float32)
            fp[:, :] = 1
            fp[1:6, 1:6] = 0
            fp[2:5, 2:5] = 1
            for (r0, c0) in ((0, 0), (0, m - 7), (m - 7, 0)):
                bits[max(0, r0 - 1):r0 + 8, max(0, c0 - 1):c0 + 8] = 0
                bits[r0:r0 + 7, c0:c0 + 7] = fp
        img = np.kron(bits, np.ones((px, px), np.float32))
        out = np.zeros((N, N), np.float32)
        s = img.shape[0]
        x0 = int((N - s) / 2 + rng.normal(0, 3))
        y0 = int((N - s) / 2 + rng.normal(0, 3))
        return _paste(out, img, x0, y0)

    def dots(self, rng):
        u = rng.random()
        out = np.zeros((N, N), np.float32)
        if u < 0.45:  # small disks
            sf = _SS(2)
            for _ in range(int(rng.integers(1, 41))):
                r = rng.uniform(0.6, 4.5) * 2
                x, y = rng.uniform(4, N - 4, 2) * 2
                sf.d.ellipse([x - r, y - r, x + r, y + r], fill=int(rng.integers(100, 256)) if rng.random() < 0.4 else 255)
            return sf.done()
        if u < 0.7:  # Gaussian spots
            for _ in range(int(rng.integers(1, 12))):
                cx, cy = rng.uniform(5, N - 5, 2)
                s = rng.uniform(0.8, 6)
                out = np.maximum(out, rng.uniform(0.4, 1) * np.exp(-((_XX - cx) ** 2 + (_YY - cy) ** 2) / (2 * s * s)))
            return out
        if u < 0.85:  # single pixels
            k = int(rng.integers(1, 60))
            ys, xs = rng.integers(2, N - 2, (2, k))
            out[ys, xs] = rng.uniform(0.5, 1.0, k) if rng.random() < 0.4 else 1.0
            return out
        # sparse lattice
        p = int(rng.integers(8, 33))
        o = rng.integers(0, p, 2)
        out[o[0]::p, o[1]::p] = 1.0
        if rng.random() < 0.5:
            out = np.minimum(1.0, _gauss(out, rng.uniform(0.5, 1.5)) * 8)
        return out

    def hard(self, rng):
        u = rng.random()
        if u < 0.25:
            return np.zeros((N, N), np.float32)
        if u < 0.45:
            return np.ones((N, N), np.float32)
        if u < 0.57:
            return np.full((N, N), rng.uniform(0.1, 0.9), np.float32)
        if u < 0.77:  # nearly empty: a single dot / faint speck
            out = np.zeros((N, N), np.float32)
            y, x = rng.integers(2, N - 2, 2)
            r = int(rng.integers(0, 3))
            out[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = 1.0 if rng.random() < 0.7 else rng.uniform(0.05, 0.3)
            return out
        # full white with a small dark object
        sub = (self.glyph if rng.random() < 0.5 else self.dots)(rng)
        return 1.0 - sub * rng.uniform(0.6, 1.0)

    # ---------------- real-image families ----------------
    def _real_photo(self, rng, smin=0.6):
        r = self.ext.pick(PHOTO_SOURCES, self.split, rng)
        return None if r is None else _crop_resize(r[1], rng, smin)

    def _soft_window(self, rng, r_lo=0.3, r_hi=0.48):
        sf = _SS(2)
        _draw_shape(sf, ("circle", "ellipse", "roundrect", "square", "circle")[int(rng.integers(5))],
                    N / 2 + rng.normal(0, 5), N / 2 + rng.normal(0, 5), rng.uniform(r_lo, r_hi) * N,
                    0.0 if rng.random() < 0.7 else rng.uniform(0, 6.3), 1.0, True, 2, rng)
        return _gauss(sf.done(), rng.uniform(0.5, 4.0))

    def photo(self, rng):
        x = self._real_photo(rng)
        if x is None:
            return self.synphoto(rng)
        x = _tone(x, rng)
        u = rng.random()
        if u < 0.12:  # porthole: photo in a soft window on black
            x = x * self._soft_window(rng)
        elif u < 0.35:
            x = x * _vignette(rng)
        elif u < 0.45:  # low-key
            x = x * rng.uniform(0.35, 0.7)
        v = rng.random()
        if v < 0.1:
            x = _gauss(x, rng.uniform(0.4, 1.2))
        elif v < 0.2:
            x = x + rng.uniform(0.3, 1.0) * (x - _gauss(x, 1.0))
        return np.clip(x, 0, 1)

    def _dark_background(self, rng):
        u = rng.random()
        if u < 0.45:
            return np.zeros((N, N), np.float32)
        if u < 0.75:
            lo, hi = rng.uniform(0, 0.05), rng.uniform(0.05, 0.3)
            return (lo + (hi - lo) * self._gradient_raw(rng)).astype(np.float32)
        if u < 0.9:
            return (_fractal_noise(rng, rng.uniform(2.2, 3.5)) ** 2 * rng.uniform(0.1, 0.35)).astype(np.float32)
        if u < 0.96:
            return np.full((N, N), rng.uniform(0.1, 0.4), np.float32)
        return np.full((N, N), rng.uniform(0.85, 1.0), np.float32)

    def _object_layer(self, rng, size_frac=None, center=None, sources=None):
        """Real masked object on a transparent canvas: (image, alpha) float32 N x N, or None."""
        r = self.ext.pick(sources or OBJECT_SOURCES, self.split, rng)
        if r is None:
            return None
        name, im, m = r
        if m is None:
            m = np.clip(im.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
        sz = max(8, int((size_frac or rng.uniform(0.45, 0.98)) * N))
        I, M = Image.fromarray(np.ascontiguousarray(im)), Image.fromarray(np.ascontiguousarray(m))
        if rng.random() < 0.3:
            ang = float(rng.normal(0, 12))
            I, M = I.rotate(ang, Image.BILINEAR), M.rotate(ang, Image.BILINEAR)
        rs = Image.BICUBIC if im.shape[0] < sz else Image.BILINEAR
        obj = np.asarray(I.resize((sz, sz), rs), dtype=np.float32) / 255.0
        alpha = np.asarray(M.resize((sz, sz), Image.BILINEAR), dtype=np.float32) / 255.0
        if rng.random() < 0.5:
            obj, alpha = obj[:, ::-1], alpha[:, ::-1]
        alpha = _gauss(np.ascontiguousarray(alpha), rng.uniform(0.3, 1.2))
        obj = _tone(np.ascontiguousarray(obj), rng, 0.7)
        if rng.random() < 0.3:
            obj = _metal(obj, rng)
        cx, cy = center if center is not None else (N / 2 + rng.normal(0, 10), N / 2 + rng.normal(0, 10))
        O, A = np.zeros((N, N), np.float32), np.zeros((N, N), np.float32)
        x0, y0 = int(round(cx - sz / 2)), int(round(cy - sz / 2))
        _paste(O, obj, x0, y0)
        _paste(A, np.clip(alpha, 0, 1), x0, y0)
        return O, A

    def object(self, rng):
        lay = self._object_layer(rng)
        if lay is None:
            return self.emoji(rng) if rng.random() < 0.5 else self.shapes(rng)
        O, A = lay
        out = self._dark_background(rng)
        if rng.random() < 0.3:  # glow / halo behind the object
            out = out + _blur_big(A, rng.uniform(3, 14)) * rng.uniform(0.2, 0.7)
        out = out * (1 - A) + O * A
        if rng.random() < 0.4:
            out = out * _vignette(rng, rng.uniform(0.2, 0.6))
        return np.clip(out, 0, 1)

    def _rocket(self, rng, cx, top, size):
        """Synthetic rocket / vehicle with cylindrical metallic shading: (image, alpha)."""
        w = size * rng.uniform(0.12, 0.24)
        nose = size * rng.uniform(0.18, 0.35)
        bottom = top + size
        sf = _SS(2)
        s = 2
        t = np.linspace(0, 1, 24)
        prof = (w / 2) * t ** rng.uniform(0.4, 0.8)
        left = [((cx - p) * s, (top + nose * tt) * s) for p, tt in zip(prof, t)]
        right = [((cx + p) * s, (top + nose * tt) * s) for p, tt in zip(prof[::-1], t[::-1])]
        body = left + [((cx - w / 2) * s, bottom * s), ((cx + w / 2) * s, bottom * s)] + right
        sf.d.polygon(body, fill=255)
        fw, fh = w * rng.uniform(0.5, 1.1), size * rng.uniform(0.15, 0.3)
        for sg in (-1, 1):
            sf.d.polygon([((cx + sg * w / 2) * s, (bottom - fh) * s), ((cx + sg * (w / 2 + fw)) * s, bottom * s),
                          ((cx + sg * w / 2) * s, bottom * s)], fill=255)
        A = sf.done()
        u = np.clip((_XX - cx) / (w / 2 + 1e-3), -1.6, 1.6)
        base = 0.25 + 0.65 * np.sqrt(np.clip(1 - (u / 1.6) ** 2, 0, 1))
        u0 = rng.uniform(-0.6, 0.6)
        spec = np.exp(-((u - u0) / rng.uniform(0.1, 0.25)) ** 2) * rng.uniform(0.3, 0.7)
        O = np.clip(base * rng.uniform(0.6, 1.0) + spec, 0, 1).astype(np.float32)
        if rng.random() < 0.6:  # dark bands
            for _ in range(int(rng.integers(1, 4))):
                yb = rng.uniform(top + nose, bottom - 4)
                O = np.where(np.abs(_YY - yb) < rng.uniform(1, 3), O * 0.35, O)
        ang = float(rng.normal(0, 15)) if rng.random() < 0.6 else 0.0
        if ang:
            c = (cx, top + size / 2)
            O = np.asarray(Image.fromarray(O, "F").rotate(ang, Image.BILINEAR, center=c), dtype=np.float32)
            A = np.asarray(Image.fromarray(A, "F").rotate(ang, Image.BILINEAR, center=c), dtype=np.float32)
        return O, A

    def darkscene(self, rng):
        """Object on a dark sky with lit smoke / clouds, glow and bloom ("rocket in smoke")."""
        top, bot = rng.uniform(0, 0.06), rng.uniform(0.03, 0.25)
        out = (top + (bot - top) * (_YY / N) ** rng.uniform(0.7, 2.0)).astype(np.float32)
        if rng.random() < 0.4:  # stars
            k = int(rng.integers(10, 80))
            ys, xs = rng.integers(0, N, (2, k))
            st = np.zeros((N, N), np.float32)
            st[ys, xs] = rng.uniform(0.2, 1.0, k)
            out = np.maximum(out, _gauss(st, 0.5) * 2.0)
        mode = rng.random()
        size = rng.uniform(0.35, 0.75) * N
        cx = N / 2 + rng.normal(0, 12)
        obj_top = rng.uniform(0.04, 0.3) * N
        O = A = None
        if mode < 0.5:
            lay = self._object_layer(rng, size_frac=size / N, center=(cx, obj_top + size / 2),
                                     sources={"c101_obj": 0.5, "c256_obj": 0.45, "fmnist_obj": 0.05})
            if lay is not None:
                O, A = lay
        if O is None and mode < 0.85:
            O, A = self._rocket(rng, cx, obj_top, size)
        # glow source: bottom of the object (or a random point)
        if A is not None and A.max() > 0.5:
            ys, xs = np.nonzero(A > 0.5)
            gy = float(ys.max())
            gx = float(xs[ys >= gy - 3].mean())
        else:
            gx, gy = cx, rng.uniform(0.3, 0.7) * N
        # smoke: plume below the glow point, a cloud bank and / or a cloud around the glow
        n = _fractal_noise(rng, rng.uniform(2.6, 3.6))
        dy = _YY - gy
        width = rng.uniform(3, 9) + np.clip(dy, 0, None) * rng.uniform(0.5, 1.5)
        drift = rng.normal(0, 0.25)
        plume = np.exp(-((_XX - gx - drift * dy) ** 2) / (2 * width ** 2)) * _sigmoid(dy / 4.0)
        smoke_m = plume * (rng.random() < 0.85)
        if rng.random() < 0.75:
            yb = N * rng.uniform(0.55, 0.92) + 30 * (n - 0.5)
            smoke_m = np.maximum(smoke_m, _sigmoid((_YY - yb) / rng.uniform(3, 10)))
        if O is None or rng.random() < 0.2:  # cloud / explosion around the glow point
            sg = rng.uniform(12, 40)
            smoke_m = np.maximum(smoke_m, np.exp(-((_XX - gx) ** 2 + (_YY - gy) ** 2) / (2 * sg * sg)))
        smoke_m = np.clip(smoke_m * (0.4 + 1.0 * n), 0, 1)
        light = np.exp(-np.sqrt((_XX - gx) ** 2 + (_YY - gy) ** 2) / rng.uniform(20, 70))
        smoke = (0.2 + 0.8 * n ** rng.uniform(0.7, 1.5)) * (rng.uniform(0.15, 0.45) + rng.uniform(0.5, 1.0) * light)
        out = out * (1 - smoke_m) + smoke * smoke_m
        if rng.random() < 0.85:  # flame / glow
            sx, sy = rng.uniform(1.5, 6), rng.uniform(4, 16)
            flame = np.exp(-((_XX - gx) ** 2 / (2 * sx * sx) + (_YY - gy - sy * 0.8) ** 2 / (2 * sy * sy)))
            out = np.maximum(out, flame.astype(np.float32) * rng.uniform(0.8, 1.0))
        if O is not None:
            out = out * (1 - A) + O * A
        bright = np.clip(out - rng.uniform(0.6, 0.85), 0, 1)
        out = out + _blur_big(bright, rng.uniform(3, 12)) * rng.uniform(0.5, 2.0)
        if rng.random() < 0.6:
            out = out * _vignette(rng, rng.uniform(0.2, 0.7))
        return np.clip(out, 0, 1)

    def phototext(self, rng):
        x = self._real_photo(rng)
        if x is None:
            x = self.synphoto(rng)
        x = _tone(x, rng)
        face = self.bank.random_face(self.split, rng, cyr=rng.random() < 0.2)
        text = self.random_text(rng, face["cyr"])
        u = rng.random()
        if u < 0.35:  # caption over the photo
            m = self._text_layer(rng, text, face, rng.uniform(0.15, 0.4) * N, max_w=rng.uniform(0.6, 0.95) * N,
                                 center_p=0.4, binarize=False)
            col = 1.0 if rng.random() < 0.7 else 0.0
            if rng.random() < 0.6:
                sh = _gauss(np.roll(m, (1, 1), (0, 1)), 1.0)
                x = x * (1 - 0.7 * sh) if col > 0.5 else np.clip(x + 0.7 * sh, 0, 1)
            return x * (1 - m) + col * m
        if u < 0.6:  # photo seen through big letters
            t = text.replace("\n", "")[: int(rng.integers(1, 5))] or "A"
            m = self._text_layer(rng, t, face, rng.uniform(0.5, 0.92) * N, max_w=0.95 * N, center_p=0.8,
                                 binarize=False)
            return x * m + (1 - m) * (0.0 if rng.random() < 0.8 else rng.uniform(0, 0.3))
        if u < 0.8:  # caption bar
            h = int(rng.uniform(0.2, 0.35) * N)
            at_top = rng.random() < 0.3
            bar_v = 0.0 if rng.random() < 0.7 else 1.0
            img = self._render_text(text.replace("\n", " ")[:10], face, rng)
            a = self._fit(img, rng, h * 0.7, max_w=N * 0.92) if img is not None else None
            out = x.copy()
            ys = slice(0, h) if at_top else slice(N - h, N)
            out[ys] = bar_v
            if a is not None:
                y0 = (0 if at_top else N - h) + (h - a.shape[0]) // 2
                m = _paste(np.zeros((N, N), np.float32), a, (N - a.shape[1]) // 2, y0)
                out = out * (1 - m) + (1 - bar_v) * m
            return out
        # photo in a round badge + word below
        r = rng.uniform(0.25, 0.33) * N
        cy = N * 0.38
        circ = np.clip(r + 0.5 - np.sqrt((_XX - N / 2) ** 2 + (_YY - cy) ** 2), 0, 1)
        out = x * circ
        img = self._render_text(text.replace("\n", " ")[:8], face, rng)
        a = self._fit(img, rng, N - (cy + r) - 6, max_w=N * 0.9) if img is not None else None
        if a is not None:
            _paste(out, a, (N - a.shape[1]) // 2, int(cy + r + 3))
        return out

    def realtexture(self, rng):
        r = self.ext.pick(TEXTURE_SOURCES, self.split, rng)
        if r is None:
            return self.texture(rng)
        x = _tone(_crop_resize(r[1], rng, smin=0.4), rng)
        if rng.random() < 0.2:
            x = x * self._soft_window(rng, 0.25, 0.46)
        return x

    # ---------------- post-processing ----------------
    _INV_P = {"glyph": 0.18, "word": 0.18, "multiglyph": 0.12, "shapes": 0.15, "lineart": 0.12, "logo": 0.15,
              "symbol": 0.15, "emoji": 0.05, "gradient": 0.1, "texture": 0.1, "synphoto": 0.05, "pattern": 0.1,
              "photo": 0.03, "object": 0.02, "darkscene": 0.0, "phototext": 0.03, "realtexture": 0.06,
              "dots": 0.05, "hard": 0.0}

    def post(self, img, family, rng):
        img = np.clip(img, 0, 1).astype(np.float32, copy=False)
        if family == "hard":
            return img
        if rng.random() < self._INV_P[family]:
            img = 1.0 - img
        if rng.random() < 0.15:
            img = img ** rng.uniform(0.6, 1.7)
        if rng.random() < 0.2:  # contrast compression / gray background
            lo = rng.uniform(0, 0.3)
            hi = rng.uniform(max(lo + 0.3, 0.6), 1.0)
            img = lo + (hi - lo) * img
        if rng.random() < 0.1:
            img = _gauss(img, rng.uniform(0.4, 1.5))
        if rng.random() < 0.06:
            img = img + rng.normal(0, rng.uniform(0.01, 0.05), img.shape).astype(np.float32)
        if rng.random() < 0.04 and family not in REAL_FAMILIES:
            k = int(rng.integers(2, 7))
            img = np.round(np.clip(img, 0, 1) * (k - 1)) / (k - 1)
        return np.clip(img, 0, 1)

    def image(self, family, rng, **kw):
        img = getattr(self, family)(rng, **kw)
        return self.post(img, family, rng)


def _to_u8(img):
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------------------------------------

class QuadGenerator:
    """Random quadruples of 8-bit targets.

    split: "train" | "val" | "test" - disjoint font families / glyphs / emoji / words and independent seed streams.
    sample(batch) -> uint8 (batch, 4, N, N);  sample(batch, return_families=True) -> (quads, [[family]*4]*batch).
    """

    P_SEQUENCE = 0.04
    P_THEMED = 0.30
    P_DUPLICATE = 0.04
    P_ALL_SAME = 0.007

    _logged = False

    def __init__(self, split: str = "train", seed: int = 0, verbose=None):
        if split not in _SPLIT_ID:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.split = split
        self.seed = int(seed)
        ss = np.random.SeedSequence(entropy=[self.seed, 0x0D0E_DA7A], spawn_key=(_SPLIT_ID[split],))
        self.rng = np.random.default_rng(ss)
        self.painter = Painter(split)
        # families that cannot be rendered here (no emoji / symbol font) get weight 0, the rest is renormalised
        self.dropped = self.painter.bank.unavailable_families(split)
        if self.dropped:
            _warn_once(f"drop-{split}-{sorted(self.dropped)}",
                       f"doe.data: families {sorted(self.dropped)} unavailable for split '{split}' on this "
                       "machine; their weight is redistributed")
            fw = np.array([0.0 if f in self.dropped else FAMILY_WEIGHTS[f] for f in FAMILIES])
            tw = np.array([0.0 if f in self.dropped else FAMILY_WEIGHTS[f] for f in _THEME_FAMILIES])
            self._fw, self._tw = fw / fw.sum(), tw / tw.sum()
        else:
            self._fw, self._tw = _FW, _TW
        if verbose is None:
            verbose = os.environ.get(VERBOSE_ENV, "") not in ("", "0") and not QuadGenerator._logged
        if verbose:
            QuadGenerator._logged = True
            print(self.describe(), flush=True)

    def family_mix(self) -> dict:
        """Effective per-image family probabilities of independent draws (before quad-level themes)."""
        return {f: float(w) for f, w in zip(FAMILIES, self._fw) if w > 0}

    def describe(self) -> str:
        """One line: platform, fonts / emoji / symbols found, external cache, effective family mix."""
        ext = self.painter.ext.counts()
        ext_s = ", ".join(f"{k} {v[self.split]}" for k, v in ext.items()) or "MISSING (synthetic fallback)"
        mix = " ".join(f"{f}={w:.3f}" for f, w in self.family_mix().items())
        return (f"[doe.data] split={self.split} platform={sys.platform} {self.painter.bank.summary()} "
                f"dropped={sorted(self.dropped) or '-'} external[{ext_s}] mix[{mix}]")

    # one image of a given family, with its own child rng
    def _img(self, family, rng, **kw):
        return self.painter.image(family, np.random.default_rng(rng.integers(2**63)), **kw)

    def _sequence_quad(self, rng):
        bank = self.painter.bank
        u = rng.random()
        cyr = False
        if u < 0.35:
            st = int(rng.integers(0, 7))
            chars = [DIGITS[(st + i) % 10] for i in range(4)]
        elif u < 0.5:
            chars = [DIGITS[int(rng.integers(10))] for _ in range(4)]
        elif u < 0.65:
            st = int(rng.integers(0, 23))
            chars = [LATIN_UP[st + i] for i in range(4)]
        elif u < 0.8:
            cands = [w for w in bank.words[self.split] if len(w) >= 4 and "\n" not in w] or ["MIFI"]
            w = cands[int(rng.integers(len(cands)))]
            st = int(rng.integers(0, len(w) - 3))
            chars = list(w[st:st + 4])
            cyr = any(c in CYR_UP + CYR_LO for c in chars)
        elif u < 0.9:
            cyr = True
            chars = [CYR_UP[int(rng.integers(len(CYR_UP)))] for _ in range(4)]
        else:
            src = LATIN_UP + LATIN_LO + DIGITS
            chars = [src[int(rng.integers(len(src)))] for _ in range(4)]
        face = bank.random_face(self.split, rng, cyr=cyr)
        if cyr and not face["cyr"]:
            chars = [LATIN_UP[int(rng.integers(26))] for _ in range(4)]
        th = rng.uniform(0.45, 0.92) * N
        same_pos = rng.random() < 0.75
        ang = self.painter._random_angle(rng, 0.8, 5, 0.0)
        inv = rng.random() < 0.15
        fixed_bin = rng.random() < 0.3
        out = []
        for ch in chars:
            r = np.random.default_rng(rng.integers(2**63))
            img = self.painter._render_text(ch, face, r)
            a = self.painter._fit(img, r, th, angle=ang, binarize=fixed_bin) if img is not None else None
            canvas = np.zeros((N, N), np.float32)
            if a is not None:
                if same_pos:
                    x0, y0 = int((N - a.shape[1]) / 2), int((N - a.shape[0]) / 2)
                else:
                    x0, y0 = self.painter._position(r, a.shape[1], a.shape[0])
                _paste(canvas, a, x0, y0)
            out.append(1.0 - canvas if inv else canvas)
        return out, ["glyph"] * 4

    def _quad(self, rng):
        u = rng.random()
        if u < self.P_SEQUENCE:
            imgs, fams = self._sequence_quad(rng)
        elif u < self.P_SEQUENCE + self.P_THEMED:
            f = _THEME_FAMILIES[int(rng.choice(len(_THEME_FAMILIES), p=self._tw))]
            kw = {}
            if f in ("glyph", "word", "multiglyph") and rng.random() < 0.5:
                kw["face"] = self.painter.bank.random_face(self.split, rng, cyr=rng.random() < 0.2)
            imgs = [self._img(f, rng, **kw) for _ in range(4)]
            fams = [f] * 4
        else:
            fams = [FAMILIES[int(i)] for i in rng.choice(len(FAMILIES), size=4, p=self._fw)]
            imgs = [self._img(f, rng) for f in fams]
        v = rng.random()
        if v < self.P_ALL_SAME:
            imgs = [imgs[0]] * 4
            fams = [fams[0]] * 4
        elif v < self.P_ALL_SAME + self.P_DUPLICATE:
            i, j = rng.choice(4, 2, replace=False)
            imgs[j] = imgs[i]
            fams[j] = fams[i]
        return np.stack([_to_u8(im) for im in imgs]), fams

    def sample(self, batch: int, return_families: bool = False):
        quads, fams = [], []
        for _ in range(int(batch)):
            q, f = self._quad(np.random.default_rng(self.rng.integers(2**63)))
            quads.append(q)
            fams.append(f)
        out = np.stack(quads) if quads else np.zeros((0, 4, N, N), np.uint8)
        return (out, fams) if return_families else out

    def sample_family(self, family: str, count: int) -> np.ndarray:
        """`count` single images (uint8 (count, N, N)) of one family - for previews / debugging."""
        return np.stack([_to_u8(self._img(family, self.rng)) for _ in range(count)])


def split_fonts(split: str) -> list:
    """Sorted font family keys used by a split (disjoint across splits)."""
    return FontBank.get().split_fonts(split)


try:  # optional torch wrapper: parallel generation in DataLoader worker processes
    import torch
    from torch.utils.data import IterableDataset, get_worker_info

    class QuadDataset(IterableDataset):
        """Infinite stream of uint8 tensor batches (B,4,N,N). Use with DataLoader(ds, batch_size=None,
        num_workers=k, persistent_workers=True); each worker gets an independent seed stream."""

        def __init__(self, split="train", seed=0, batch=64, length=None):
            self.split, self.seed, self.batch, self.length = split, seed, batch, length

        def __iter__(self):
            wi = get_worker_info()
            wid = 0 if wi is None else wi.id
            nw = 1 if wi is None else wi.num_workers
            gen = QuadGenerator(self.split, seed=self.seed * 1000 + wid)
            n = None if self.length is None else len(range(wid, self.length, nw))
            i = 0
            while n is None or i < n:
                yield torch.from_numpy(gen.sample(self.batch))
                i += 1
except ImportError:  # pragma: no cover
    QuadDataset = None
