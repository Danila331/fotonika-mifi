import torch

from doe.dc_phase import reconstruction_phase_loss, signal_mask


def test_constant_reconstruction_phase_has_zero_regularization():
    field = torch.ones((1, 4, 8, 8), dtype=torch.complex64)
    target = torch.ones((1, 4, 8, 8))
    loss = reconstruction_phase_loss(field, target)
    assert torch.allclose(loss, torch.zeros(1), atol=1e-6)


def test_alternating_reconstruction_phase_is_penalized():
    phase = torch.zeros((1, 1, 8, 8))
    phase[..., ::2, :] = torch.pi / 2
    field = torch.polar(torch.ones_like(phase), phase)
    target = torch.ones_like(phase)
    loss = reconstruction_phase_loss(field, target)
    assert loss.item() > 1.0


def test_zero_signal_mask_disables_phase_regularization():
    phase = torch.randn((1, 4, 8, 8))
    field = torch.polar(torch.ones_like(phase), phase)
    target = torch.ones_like(phase)
    mask = torch.zeros_like(target)
    loss = reconstruction_phase_loss(field, target, mask=mask)
    assert torch.allclose(loss, torch.zeros(1), atol=1e-6)


def test_signal_mask_has_batch_and_depth_dimensions():
    target = torch.zeros((2, 4, 8, 8))
    target[:, :, 2:6, 2:6] = 1
    mask = signal_mask(target, threshold=0.1, dilation=1)
    assert mask.shape == target.shape
    assert mask.min() >= 0 and mask.max() <= 1
    assert mask[0, 0, 0, 0] == 0
    assert mask[0, 0, 1, 2] == 1
