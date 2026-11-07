import numpy as np
import pytest

import predict
import synth


def test_fastgs_is_registered_with_100_steps():
    assert "nn_fastgs" in synth.METHODS
    assert synth.DEFAULT_STEPS["nn_fastgs"] == 100
    assert "nn_fastgs" in synth.METHOD_RU


def test_fastgs_uses_channels_last_and_returns_valid_phase():
    rng = np.random.default_rng(41)
    targets = rng.integers(0, 256, (4, 128, 128), dtype=np.uint8)
    engine = synth.Synthesiser("nn_fastgs", steps=2, device="cpu")
    phase = engine(targets)
    assert phase.shape == (128, 128) and np.isfinite(phase).all()
    assert np.abs(phase).max() <= np.pi
    assert engine.info["inference_layout"] == "channels_last"
    assert engine.info["fastgs"]["steps"] == 2
    assert engine.info["fastgs"]["teacher"] == "stage0_finetuned"


def test_fastgs_conflicts_with_other_modes():
    with pytest.raises(SystemExit):
        predict.main(["--fastgs", "--quality"])
    with pytest.raises(SystemExit):
        predict.main(["--fastgs", "--visual"])


def test_docker_environment_selects_fastgs_without_explicit_flag(monkeypatch):
    monkeypatch.setenv("DOE_DEFAULT_MODE", "fastgs")
    assert predict.default_method() == "nn_fastgs"
    monkeypatch.delenv("DOE_DEFAULT_MODE")
    assert predict.default_method() == "nn"
