"""Optimized random-phase (ORAP) utilities for shared reconstruction-plane masks."""
import torch


def make_shared_phase_trainable(model: torch.nn.Module) -> torch.nn.Parameter:
    """Freeze the network and expose its four shared image-plane phases for ORAP training."""
    if not hasattr(model, "rand_phase"):
        raise ValueError("model has no shared rand_phase")
    initial = model.rand_phase.detach().clone() if isinstance(model.rand_phase, torch.Tensor) else None
    if initial is None or initial.ndim != 3:
        raise ValueError("rand_phase must be a (D,H,W) tensor")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model._buffers.pop("rand_phase", None)
    phase = torch.nn.Parameter(initial)
    model.rand_phase = phase
    return phase
