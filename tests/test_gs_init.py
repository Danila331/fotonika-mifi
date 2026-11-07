import numpy as np

from doe.baselines import gs_batch


def test_gs_batch_accepts_phase_array_initialization():
    rng = np.random.default_rng(31)
    targets = rng.integers(0, 256, (1, 4, 128, 128), dtype=np.uint8)
    initial = rng.uniform(-np.pi, np.pi, (1, 128, 128))
    phase = gs_batch(targets, iters=2, init=initial, device="cpu")
    assert phase.shape == (1, 128, 128)
    assert np.isfinite(phase).all()
    assert np.abs(phase).max() <= np.pi
