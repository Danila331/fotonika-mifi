"""PNG input/output and the 8-bit phase codec (slide 6: 0 -> -π, 255 -> +π)."""

import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from .config import N


def load_gray(path, size: int = N, strict: bool = False) -> np.ndarray:
    """Load an 8-bit single-channel PNG as uint8 (size,size).

    strict=True rejects anything that is not exactly the spec; otherwise RGB/alpha/16-bit/other sizes
    are converted with a warning so the pipeline never silently crashes on a slightly wrong file.
    """
    img = Image.open(path)
    problems = []
    if img.mode != "L":
        problems.append(f"mode {img.mode}, expected L")
        if img.mode in ("I;16", "I;16B", "I;16L", "I"):
            a = np.asarray(img, dtype=np.float64)
            a = a / 257.0 if a.max() > 255 else a
            img = Image.fromarray(np.clip(np.round(a), 0, 255).astype(np.uint8))
        else:
            if img.mode in ("RGBA", "LA", "P", "PA"):
                img = img.convert("RGBA")
                bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
                img = Image.alpha_composite(bg, img)
            img = img.convert("L")
    if img.size != (size, size):
        problems.append(f"size {img.size}, expected {(size, size)}")
        img = img.resize((size, size), Image.LANCZOS)
    if problems:
        msg = f"{path}: " + "; ".join(problems)
        if strict:
            raise ValueError(msg)
        warnings.warn(msg + " (converted)")
    return np.asarray(img, dtype=np.uint8).copy()


def save_gray(path, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    if arr.dtype != np.uint8 or arr.ndim != 2:
        raise ValueError(f"expected 2-D uint8, got {arr.dtype} {arr.shape}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def wrap_phase(phase):
    return np.mod(np.asarray(phase, dtype=np.float64) + np.pi, 2 * np.pi) - np.pi


def phase_to_u8(phase) -> np.ndarray:
    p = np.asarray(phase, dtype=np.float64)
    p = np.where(np.abs(p) <= np.pi, p, wrap_phase(p))  # keep +π -> 255 as on slide 6
    q = np.round((p + np.pi) / (2 * np.pi) * 255.0)
    return np.clip(q, 0, 255).astype(np.uint8)


def u8_to_phase(q) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) / 255.0 * 2 * np.pi - np.pi
