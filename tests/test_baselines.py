import numpy as np
import torch

from doe.baselines import adam, adam_batch, backprop_init, gs, gs_batch, jury_loss, random_init, ssim_local_fft
from doe.config import N
from doe.io import phase_to_u8
from doe.metrics import evaluate
from doe.metrics_torch import ssim_local

torch.set_num_threads(4)


def _targets():
    t = np.zeros((4, N, N), np.uint8)
    t[0, 30:60, 30:90] = 255           # bar
    t[1, 40:100, 50:70] = 200          # vertical bar
    yy, xx = np.mgrid[:N, :N]
    t[2][(yy - 64) ** 2 + (xx - 64) ** 2 < 30**2] = 255  # disc
    t[3, 80:110, 20:110] = 150         # wide bar
    return t


def _pcc(phase, t):
    return evaluate(phase_to_u8(phase), t)["summary"]["pcc"]["mean"]


def test_gs_improves_over_random():
    t = _targets()
    p_rand = random_init(t[None], seed=1)[0].numpy()
    p_gs = gs(t, iters=20)
    assert p_gs.shape == (N, N) and np.isfinite(p_gs).all()
    assert _pcc(p_gs, t) > _pcc(p_rand, t) + 0.2
    p_seq = gs(t, iters=5, variant="sequential", init="random")
    assert p_seq.shape == (N, N) and np.isfinite(p_seq).all()


def test_adam_improves_over_init():
    t = _targets()
    p0 = backprop_init(t[None], seed=0)[0].numpy()
    p = adam(t, steps=40, init_phase=p0, lr=0.1, loss="pcc")
    assert p.shape == (N, N) and np.isfinite(p).all()
    assert np.abs(p).max() <= np.pi + 1e-6
    assert _pcc(p, t) > _pcc(p0, t) + 0.05
    # default (jury) loss decreases on the 8-bit phase
    from doe.physics import Fresnel
    prop, t01 = Fresnel(), torch.tensor(t[None] / 255.0, dtype=torch.float32)
    jl = lambda ph: float(jury_loss(prop.intensity(torch.tensor(phase_to_u8(ph) / 255.0 * 2 * np.pi - np.pi,
                                                                  dtype=torch.float32)[None]), t01))
    assert jl(adam(t, steps=40, init_phase=p0, lr=0.1)) < jl(p0) - 0.1


def test_batched_shapes_and_snapshots():
    t = np.stack([_targets(), _targets()[::-1].copy()])
    out, snaps = gs_batch(t, 3, record_at=[1, 3])
    assert out.shape == (2, N, N) and set(snaps) == {1, 3}
    out = adam_batch(t, 3)
    assert out.shape == (2, N, N) and np.isfinite(out).all()


def test_jury_loss_and_fast_ssim():
    g = torch.Generator().manual_seed(0)
    a = torch.rand(2, 4, N, N, generator=g)
    b = torch.rand(2, 4, N, N, generator=g)
    assert torch.allclose(ssim_local_fft(a, b), ssim_local(a, b), atol=1e-5)
    i = a.clone().requires_grad_(True)
    loss = jury_loss(i, b)
    loss.backward()
    loss = loss.detach()
    assert loss.ndim == 0 and torch.isfinite(loss) and torch.isfinite(i.grad).all()
    # amplitude rule: |K| ∝ img, i.e. intensity ∝ img² at any scale is (near) perfect
    assert float(jury_loss(b.pow(2) * 3.0, b)) < 0.05 < float(loss)
