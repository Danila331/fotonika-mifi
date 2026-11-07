"""Inference API for the neural DOE synthesiser (ML agent), see CONTRACTS.md.

    from doe.infer import NeuralDOE
    net = NeuralDOE.load()                       # checkpoints/best.pt, MPS if available else CPU
    phase = net.predict(targets_u8)              # (4,N,N) uint8 -> (N,N) float in [-π, π]; ONE forward pass
    phase = net.predict_refined(targets_u8, 50)  # network init + 50 explicit Adam steps (separate mode)

predict_batch / predict_refined_batch accept (B,4,N,N) and return (B,N,N).
"""

import dataclasses
import time
from pathlib import Path

import numpy as np
import torch

from .losses import LossConfig, per_sample_loss
from .model import build_model

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "checkpoints" / "best.pt"


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class NeuralDOE:
    def __init__(self, model, device, meta=None):
        self.model = model.to(device).eval()
        self.device = device
        self.meta = meta or {}
        lc = dict(self.meta.get("loss_cfg") or {})
        lc["quant"] = False
        self.loss_cfg = LossConfig(**lc)

    @classmethod
    def load(cls, path=None, device=None) -> "NeuralDOE":
        path = Path(path) if path is not None else DEFAULT_CKPT
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        device = torch.device(device) if device is not None else _default_device()
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(ck["model_cfg"])
        model.load_state_dict(ck["state_dict"])
        meta = {k: v for k, v in ck.items() if k != "state_dict"}
        meta["path"] = str(path)
        return cls(model, device, meta)

    # ------------------------------------------------------------------ helpers
    def _to_tensor(self, targets_u8) -> torch.Tensor:
        t = np.asarray(targets_u8)
        if t.ndim != 4 or t.shape[1] != 4:
            raise ValueError(f"expected (B,4,N,N) targets, got {t.shape}")
        return torch.tensor(t.astype(np.float32) / 255.0, device=self.device)

    # ------------------------------------------------------------------ pure network
    @torch.no_grad()
    def predict_batch(self, targets_u8) -> np.ndarray:
        x = self._to_tensor(targets_u8)
        return self.model(x).float().cpu().numpy().astype(np.float64)

    def predict(self, targets_u8) -> np.ndarray:
        """(4,N,N) uint8 -> (N,N) phase in [-π, π]. Single forward pass, no iterations."""
        t = np.asarray(targets_u8)
        if t.ndim != 3:
            raise ValueError(f"expected (4,N,N) targets, got {t.shape}")
        return self.predict_batch(t[None])[0]

    # ------------------------------------------------------------------ network + explicit refinement
    def predict_refined_batch(self, targets_u8, steps: int = 50, lr: float = 0.05,
                              xtalk: float | None = None, alpha: float | None = None,
                              beta: float | None = None) -> np.ndarray:
        """Network initialisation followed by `steps` Adam steps on the phase pixels, same jury loss
        (each sample optimised independently). This is an iterative mode, reported separately.
        xtalk: crosstalk-penalty weight used during refinement (None = the weight the model was trained with).
        alpha / beta: weights of the windowed / global SSIM terms in the refinement loss (None = as trained).
        A weaker penalty than in training (0.25 vs 1.0) trades some depth separation for cleaner images."""
        x = self._to_tensor(targets_u8)
        with torch.no_grad():
            phase0 = self.model(x).float()
        if steps <= 0:
            return phase0.cpu().numpy().astype(np.float64)
        fresnel = self.model.fresnel
        kw = {k: float(v) for k, v in (("xtalk", xtalk), ("alpha", alpha), ("beta", beta)) if v is not None}
        cfg = dataclasses.replace(self.loss_cfg, quant=False, **kw)
        ph = phase0.clone().requires_grad_(True)
        opt = torch.optim.Adam([ph], lr=lr)
        for _ in range(steps):
            loss, _ = per_sample_loss(ph, x, fresnel, cfg)
            opt.zero_grad(set_to_none=True)
            loss.sum().backward()
            opt.step()
        out = torch.remainder(ph.detach() + np.pi, 2 * np.pi) - np.pi
        return out.cpu().numpy().astype(np.float64)

    def predict_refined(self, targets_u8, steps: int = 50, lr: float = 0.05,
                        xtalk: float | None = None, alpha: float | None = None,
                        beta: float | None = None) -> np.ndarray:
        t = np.asarray(targets_u8)
        if t.ndim != 3:
            raise ValueError(f"expected (4,N,N) targets, got {t.shape}")
        return self.predict_refined_batch(t[None], steps, lr, xtalk, alpha, beta)[0]

    # ------------------------------------------------------------------ timing
    def benchmark(self, targets_u8, repeats: int = 30) -> float:
        """Median wall-clock ms of predict() (includes host<->device transfer)."""
        for _ in range(3):
            self.predict(targets_u8)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            self.predict(targets_u8)
            ts.append((time.perf_counter() - t0) * 1000)
        return float(np.median(ts))
