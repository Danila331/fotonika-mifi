import numpy as np

from doe.single_plane import single_plane_gs_components, single_plane_superposition


def test_single_plane_components_have_one_phase_per_depth():
    targets = np.zeros((2, 4, 128, 128), dtype=np.uint8)
    targets[..., 32:96, 32:96] = 255
    components = single_plane_gs_components(targets, iters=2)
    assert components.shape == (2, 4, 128, 128)
    assert np.isfinite(components).all()
    assert np.abs(components).max() <= np.pi


def test_superposition_is_phase_of_complex_component_sum():
    targets = np.random.default_rng(4).integers(0, 256, (1, 4, 128, 128), dtype=np.uint8)
    components = single_plane_gs_components(targets, iters=1)
    expected = np.angle(np.exp(1j * components).sum(axis=1))
    actual = single_plane_superposition(components)
    assert np.allclose(actual, expected, atol=1e-12)
