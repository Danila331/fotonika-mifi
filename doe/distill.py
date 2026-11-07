"""Losses and validation for teacher-to-student DOE distillation."""
import numpy as np
import torch


def circular_phasor_loss(student_phase: torch.Tensor, teacher_phase: torch.Tensor) -> torch.Tensor:
    """Periodic phase discrepancy, averaged per sample without a wrap discontinuity."""
    if student_phase.shape != teacher_phase.shape or student_phase.ndim < 2:
        raise ValueError("student_phase and teacher_phase must share shape (B,...)")
    return (1 - torch.cos(student_phase - teacher_phase)).flatten(1).mean(1)


def validate_teacher_arrays(targets: np.ndarray, phases: np.ndarray) -> None:
    """Validate the persisted teacher contract before training or resuming."""
    if targets.ndim != 4 or targets.shape[1:] != (4, 128, 128):
        raise ValueError("targets shape must be (B,4,128,128)")
    if phases.shape != (len(targets), 128, 128):
        raise ValueError("phases shape must match targets as (B,128,128)")
    if targets.dtype != np.uint8:
        raise ValueError("targets must be uint8")
    if phases.dtype != np.float32:
        raise ValueError("phases must be float32")
    if not np.isfinite(phases).all():
        raise ValueError("phases must be finite")
    if np.abs(phases).max(initial=0) > np.pi + 1e-6:
        raise ValueError("phases must lie in [-pi, pi]")


def build_student():
    """Build the compact two-stage student without changing the production model."""
    from .model import NeuralDOENet

    return NeuralDOENet(kind="image", inputs="physics", residual=False,
                        width=16, mults=(1, 2, 4, 8), blocks=1, attn=True,
                        unroll=2, unroll_width=16, unroll_mults=(1, 2, 4),
                        seed=1234, quantity="amplitude", bp_phase="random")


def truncate_stages(model, count):
    """Keep a prefix of learned refinement stages for teacher-compression experiments."""
    total = len(model.stages)
    if count < 0 or count > total:
        raise ValueError(f"stage count must be in [0, {total}]")
    model.stages = torch.nn.ModuleList(list(model.stages)[:count])
    model.stage_mix = torch.nn.Parameter(model.stage_mix.detach()[:count].clone(),
                                         requires_grad=model.stage_mix.requires_grad)
    model.cfg["unroll"] = count
    return model
