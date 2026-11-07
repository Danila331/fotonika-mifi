import torch

from doe.model import NeuralDOENet
from doe.orap import make_shared_phase_trainable


def test_shared_phase_becomes_the_only_trainable_parameter():
    model = NeuralDOENet(kind="image", inputs="physics", width=8, mults=(1, 2), blocks=1,
                         attn=False, unroll=1, unroll_width=8, unroll_mults=(1, 2))
    phase = make_shared_phase_trainable(model)
    assert isinstance(model.rand_phase, torch.nn.Parameter)
    assert phase is model.rand_phase
    assert phase.shape == (4, 128, 128)
    assert phase.requires_grad
    assert all(not p.requires_grad for name, p in model.named_parameters() if name != "rand_phase")
    assert "rand_phase" in dict(model.named_parameters())
    assert "rand_phase" not in model._buffers
