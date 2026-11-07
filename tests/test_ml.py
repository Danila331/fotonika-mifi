import math
from pathlib import Path

import numpy as np
import pytest
import torch

from doe.config import N
from doe.losses import LossConfig, jury_loss, quantize_phase, quantize_ste
from doe.io import phase_to_u8, u8_to_phase
from doe.model import build_model

ROOT = Path(__file__).resolve().parent.parent
SMALL = dict(width=8, mults=(1, 2), blocks=1, attn=True)


@pytest.mark.parametrize("cfg", [
    dict(kind="image", inputs="physics"),
    dict(kind="image", inputs="raw"),
    dict(kind="doe", inputs="physics"),
    dict(kind="doe", inputs="raw"),
    dict(kind="doe", inputs="physics", residual=True),
    dict(kind="unroll", inputs="physics", unroll=2, unroll_width=8, unroll_mults=(1, 2)),
    dict(kind="doe", inputs="physics", unroll=1, unroll_width=8, unroll_mults=(1, 2)),
])
def test_model_shape_range_and_grad(cfg):
    torch.manual_seed(0)
    m = build_model({**SMALL, **cfg})
    x = torch.rand(2, 4, N, N)
    phases = m(x, return_all=True)
    assert len(phases) == 1 + cfg.get("unroll", 0)
    ph = phases[-1]
    assert ph.shape == (2, N, N)
    assert torch.isfinite(ph).all()
    assert ph.abs().max() <= math.pi + 1e-5
    loss, _ = jury_loss(ph, x, m.fresnel, LossConfig(quant=True))
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads if g is not None)
    assert sum(float(g.abs().sum()) for g in grads if g is not None) > 0


def test_quantize_matches_codec():
    p = torch.rand(1000, dtype=torch.float64) * 2 * math.pi - math.pi
    q = quantize_phase(p).numpy()
    ref = u8_to_phase(phase_to_u8(p.numpy()))
    assert np.allclose(q, ref, atol=1e-6)
    pg = p.clone().requires_grad_(True)
    quantize_ste(pg).sum().backward()
    assert torch.allclose(pg.grad, torch.ones_like(pg))


@pytest.mark.skipif(not (ROOT / "checkpoints" / "best.pt").exists(), reason="no trained checkpoint yet")
def test_infer_checkpoint_deterministic():
    from doe.infer import NeuralDOE
    net = NeuralDOE.load(device="cpu")
    rng = np.random.default_rng(0)
    t = rng.integers(0, 256, (4, N, N)).astype(np.uint8)
    a = net.predict(t)
    b = net.predict(t)
    assert a.shape == (N, N) and np.isfinite(a).all()
    assert np.abs(a).max() <= math.pi + 1e-5
    assert np.array_equal(a, b)
    r = net.predict_refined(t, steps=3)
    assert r.shape == (N, N) and np.isfinite(r).all() and np.abs(r).max() <= math.pi + 1e-5


def test_amplitude_loss_matches_official_metrics_and_handles_empty():
    """Peak-normalised amplitude PCC/SSIM in the loss == doe.metrics.evaluate (up to 8-bit rounding)."""
    from doe.losses import LossConfig, per_sample_loss
    from doe.metrics import evaluate
    from doe.physics import Fresnel
    rng = np.random.default_rng(1)
    t = rng.integers(0, 200, (1, 4, N, N)).astype(np.uint8)
    t[0, 3] = 0  # all-zero target plane: PCC undefined, must not produce NaN
    ph = rng.uniform(-math.pi, math.pi, (N, N))
    fr = Fresnel().double()
    x = torch.tensor(t / 255.0)
    loss, m = per_sample_loss(torch.tensor(ph)[None], x, fr, LossConfig(norm="peak"))
    assert torch.isfinite(loss).all()
    rep = evaluate(phase_to_u8(ph), t[0])
    for d in range(3):
        assert float(m["pcc"][0, d]) == pytest.approx(rep["planes"][d]["pcc"], abs=0.02)
        assert float(m["ssim_global"][0, d]) == pytest.approx(rep["planes"][d]["ssim_global"], abs=0.02)
    assert not bool(m["pcc_valid"][0, 3])


def test_model_brightness_invariant():
    torch.manual_seed(0)
    m = build_model({**SMALL, "kind": "doe", "inputs": "physics"}).eval()
    x = torch.rand(1, 4, N, N)
    with torch.no_grad():
        assert torch.allclose(m(x), m(x * 0.5), atol=1e-5)


def test_crosstalk_penalty_behaviour():
    import torch
    from doe.losses import crosstalk_penalty
    g = torch.Generator().manual_seed(0)
    t = torch.rand(2, 4, 32, 32, generator=g)
    blend = t.mean(1, keepdim=True).expand_as(t)
    assert float(crosstalk_penalty(t, t).max()) < 1e-3            # perfect separation: no penalty
    assert float(crosstalk_penalty(blend, t).min()) > 0.2          # same blend everywhere: penalised
    same = t[:, :1].expand_as(t)                                    # identical targets: nothing to separate
    assert float(crosstalk_penalty(same, same).max()) < 1e-6
    x = blend.clone().requires_grad_(True)
    crosstalk_penalty(x, t).sum().backward()
    assert torch.isfinite(x.grad).all()
