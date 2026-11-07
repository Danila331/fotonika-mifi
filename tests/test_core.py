import numpy as np
import pytest
import torch

from doe.config import DEPTHS, DX, N, WAVELENGTH, critical_distance, output_pitch
from doe.io import load_gray, phase_to_u8, save_gray, u8_to_phase
from doe.metrics import compare, evaluate, pcc, ssim_global, ssim_skimage
from doe.metrics_torch import pcc as tpcc, ssim_global as tssim_global, ssim_local as tssim_local
from doe.physics import Fresnel, backpropagate_np, intensity_np, propagate_np

rng = np.random.default_rng(0)


def test_optics_numbers():
    assert abs(critical_distance() - 24.06e-3) < 0.01e-3
    assert abs(output_pitch(DEPTHS[0]) - 10.02e-6) < 0.01e-6
    assert all(z >= critical_distance() for z in DEPTHS)


def test_energy_conserved_every_depth():
    phase = rng.uniform(-np.pi, np.pi, (N, N))
    e = intensity_np(phase).sum(axis=(-2, -1))
    assert np.allclose(e, N * N, rtol=1e-10)


def test_backpropagation_inverts_forward():
    phase = rng.uniform(-np.pi, np.pi, (N, N))
    fields = propagate_np(phase)
    for i, z in enumerate(DEPTHS):
        assert np.allclose(backpropagate_np(fields[i], z), np.exp(1j * phase), atol=1e-10)


def test_lens_focuses_on_axis():
    u = (np.arange(N) - N // 2) * DX
    r2 = u[:, None] ** 2 + u[None, :] ** 2
    for i, z in enumerate(DEPTHS):
        inten = intensity_np(-np.pi * r2 / (WAVELENGTH * z))
        assert np.unravel_index(inten[i].argmax(), inten[i].shape) == (N // 2, N // 2)
        assert inten[i].max() / inten[i].sum() > 0.99


def test_torch_matches_numpy():
    phase = rng.uniform(-np.pi, np.pi, (2, N, N))
    model = Fresnel()
    t = model(torch.tensor(phase, dtype=torch.float32)).detach().numpy()
    for b in range(2):
        ref = propagate_np(phase[b])
        assert np.abs(t[b] - ref).max() < 1e-3 * np.abs(ref).max()
    back = model.backward_field(model(torch.tensor(phase, dtype=torch.float32)))
    assert torch.allclose(back[:, 2], torch.exp(1j * torch.tensor(phase, dtype=torch.float32)), atol=1e-4)


def test_phase_codec():
    q = np.arange(256, dtype=np.uint8)
    assert np.array_equal(phase_to_u8(u8_to_phase(q)), q)
    assert u8_to_phase(0) == -np.pi and u8_to_phase(255) == np.pi
    phase = rng.uniform(-10, 10, (N, N))
    err = np.angle(np.exp(1j * (u8_to_phase(phase_to_u8(phase)) - phase)))
    assert np.abs(err).max() <= np.pi / 255 + 1e-9


def test_png_roundtrip_and_conversion(tmp_path):
    a = rng.integers(0, 256, (N, N), dtype=np.uint8)
    save_gray(tmp_path / "a.png", a)
    assert np.array_equal(load_gray(tmp_path / "a.png", strict=True), a)
    from PIL import Image
    Image.new("RGB", (64, 64), (255, 0, 0)).save(tmp_path / "rgb.png")
    with pytest.raises(ValueError):
        load_gray(tmp_path / "rgb.png", strict=True)
    with pytest.warns(UserWarning):
        b = load_gray(tmp_path / "rgb.png")
    assert b.shape == (N, N) and b.dtype == np.uint8


def test_metrics_edge_cases():
    a = rng.integers(0, 256, (N, N)).astype(np.uint8)
    m = compare(a, a)
    assert abs(m["pcc"] - 1) < 1e-12 and abs(m["ssim_global"] - 1) < 1e-12 and abs(m["ssim_skimage"] - 1) < 1e-9
    assert pcc(a, np.zeros_like(a)) is None
    assert pcc(a, 255 - a) == pytest.approx(-1)
    assert ssim_global(a, a) == pytest.approx(1)


def test_torch_metrics_match_numpy():
    a = rng.uniform(0, 1, (N, N))
    b = np.clip(a + rng.normal(0, 0.2, (N, N)), 0, 1)
    ta, tb = torch.tensor(a), torch.tensor(b)
    assert float(tpcc(ta, tb)) == pytest.approx(pcc(a, b), abs=1e-9)
    assert float(tssim_global(ta, tb)) == pytest.approx(ssim_global(a * 255, b * 255), abs=1e-9)
    from skimage.metrics import structural_similarity
    ref = structural_similarity(a, b, data_range=1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    assert float(tssim_local(ta, tb)) == pytest.approx(ref, abs=2e-3)


def test_evaluate_from_png_bytes():
    phase = phase_to_u8(rng.uniform(-np.pi, np.pi, (N, N)))
    targets = rng.integers(0, 256, (4, N, N)).astype(np.uint8)
    rep = evaluate(phase, targets)
    assert rep["recon"].shape == (4, N, N) and rep["recon"].dtype == np.uint8
    assert len(rep["planes"]) == 4 and rep["summary"]["pcc"]["valid"] == 4


def test_expert_rules_amplitude_and_max_normalisation():
    """E1: compare amplitudes |K|; E2: every image normalised to max = 1 before the metrics."""
    phase_u8 = phase_to_u8(rng.uniform(-np.pi, np.pi, (N, N)))
    amp = np.sqrt(intensity_np(u8_to_phase(phase_u8)))
    targets = (amp / amp.max(axis=(-2, -1), keepdims=True) * 200).round().astype(np.uint8)  # max 200, not 255
    rep = evaluate(phase_u8, targets)
    assert rep["quantity"] == "amplitude" and rep["policy"] == "peak"
    assert np.abs(rep["recon"].astype(int) - (amp / amp.max(axis=(-2, -1), keepdims=True) * 255).round()).max() <= 1
    for p in rep["planes"]:  # target scaled to max 200 still matches perfectly after max-normalisation
        assert p["pcc"] > 0.999 and p["ssim_global"] > 0.999 and p["ssim_skimage"] > 0.99
    assert evaluate(phase_u8, targets, quantity="intensity")["summary"]["pcc"]["mean"] < 0.99


def test_crosstalk_metric_blend_vs_separated():
    """A reconstruction that shows the same blend in every plane must score ~0 selectivity."""
    from doe.metrics import crosstalk
    t = rng.integers(0, 256, (4, N, N)).astype(np.uint8)
    blend = np.repeat(t.astype(float).mean(0, keepdims=True), 4, 0).round().astype(np.uint8)
    assert abs(crosstalk(t, blend)["selectivity"]) < 0.05
    sep = crosstalk(t, t)
    assert sep["selectivity"] > 0.9 and sep["own_best"] == 4
