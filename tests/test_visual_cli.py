import numpy as np
import pytest

import predict
import synth


def test_visual_method_is_registered():
    assert "nn_visual" in synth.METHODS
    assert synth.DEFAULT_STEPS["nn_visual"] == 300
    assert "nn_visual" in synth.METHOD_RU


def test_visual_conflicts_with_other_modes():
    with pytest.raises(SystemExit):
        predict.main(["--visual", "--quality"])
    with pytest.raises(SystemExit):
        predict.main(["--visual", "--steps", "2"])


def test_visual_zero_steps_returns_valid_phase_and_metadata():
    rng = np.random.default_rng(22)
    targets = rng.integers(0, 256, (4, 128, 128), dtype=np.uint8)
    engine = synth.Synthesiser("nn_visual", steps=0, device="cpu")
    phase = engine(targets)
    assert phase.shape == (128, 128) and np.isfinite(phase).all()
    assert engine.info["visual"]["baseline_steps"] == 100
    assert engine.info["visual"]["ms_l1_weight"] == 0.3
    assert engine.info["visual"]["excess_tv_weight"] == 0.1
