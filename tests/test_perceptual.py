from types import SimpleNamespace

import numpy as np
import torch

from doe.distill import validate_teacher_arrays
from doe.io import phase_to_u8, u8_to_phase
from doe.losses import LossConfig
from doe.perceptual import refine_perceptual
from doe.physics import Fresnel
from doe.quality import acceptable, exported_metrics


def _net():
    return SimpleNamespace(device=torch.device("cpu"), loss_cfg=LossConfig(),
                           model=SimpleNamespace(fresnel=Fresnel()),
                           _to_tensor=lambda a: torch.tensor(a / 255.0, dtype=torch.float32))


def test_zero_step_perceptual_returns_quantized_baseline():
    rng = np.random.default_rng(19)
    targets = rng.integers(0, 256, (1, 4, 128, 128), dtype=np.uint8)
    baseline = rng.uniform(-np.pi, np.pi, (1, 128, 128))
    phase, choices = refine_perceptual(_net(), targets, baseline, steps=0)
    assert np.array_equal(phase, u8_to_phase(phase_to_u8(baseline)))
    assert choices[0]["step"] == 0


def test_perceptual_candidates_preserve_guardrails_and_valid_phase():
    rng = np.random.default_rng(20)
    targets = rng.integers(0, 256, (1, 4, 128, 128), dtype=np.uint8)
    baseline = rng.uniform(-np.pi, np.pi, (1, 128, 128))
    phase, _ = refine_perceptual(_net(), targets, baseline, steps=2, check_every=1)
    assert phase.shape == baseline.shape and np.isfinite(phase).all()
    assert acceptable(exported_metrics(phase[0], targets[0]), exported_metrics(baseline[0], targets[0]))
    validate_teacher_arrays(targets, phase.astype(np.float32))
