import math

import pytest
import torch

from doe.distill import circular_phasor_loss


def test_identical_phases_have_zero_loss_and_batch_shape():
    phase = torch.randn((3, 128, 128))
    loss = circular_phasor_loss(phase, phase.clone())
    assert loss.shape == (3,)
    assert torch.allclose(loss, torch.zeros(3), atol=1e-7)


def test_phasor_loss_is_periodic_at_wrapped_boundary():
    student = torch.tensor([[-math.pi + 0.05]])
    teacher = torch.tensor([[math.pi - 0.05]])
    assert circular_phasor_loss(student, teacher).item() == pytest.approx(1 - math.cos(0.1), abs=1e-7)


def test_opposite_phasors_have_maximum_loss():
    student = torch.zeros((2, 4, 4))
    teacher = torch.full_like(student, math.pi)
    assert torch.allclose(circular_phasor_loss(student, teacher), torch.full((2,), 2.0), atol=1e-7)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        circular_phasor_loss(torch.zeros((2, 4, 4)), torch.zeros((1, 4, 4)))
