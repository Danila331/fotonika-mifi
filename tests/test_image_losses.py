import torch

from doe.image_losses import excess_tv_loss, multiscale_l1_loss


def test_multiscale_l1_is_zero_for_identical_images():
    target = torch.rand((2, 4, 16, 16))
    assert torch.allclose(multiscale_l1_loss(target, target), torch.zeros(2), atol=1e-7)


def test_multiscale_l1_is_positive_for_blurred_reconstruction():
    target = torch.rand((1, 4, 16, 16))
    reconstruction = torch.nn.functional.avg_pool2d(
        target.reshape(4, 1, 16, 16), 3, stride=1, padding=1).reshape(1, 4, 16, 16)
    assert multiscale_l1_loss(reconstruction, target).item() > 0


def test_excess_tv_penalizes_only_roughness_above_target():
    flat = torch.zeros((1, 4, 16, 16))
    checker = torch.zeros((1, 4, 16, 16))
    checker[..., ::2, :] = 1
    assert excess_tv_loss(checker, flat).item() > 0
    assert torch.allclose(excess_tv_loss(flat, checker), torch.zeros(1))
    assert torch.allclose(excess_tv_loss(checker, checker), torch.zeros(1))
