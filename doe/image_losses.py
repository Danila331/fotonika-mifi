"""Image-domain losses for exploratory visual-quality refinement."""
import torch
import torch.nn.functional as F


def _flatten_planes(x):
    b, d, h, w = x.shape
    return x.reshape(b * d, 1, h, w)


def multiscale_l1_loss(rec: torch.Tensor, target: torch.Tensor,
                       scales=(1, 2, 4, 8)) -> torch.Tensor:
    """Mean L1 over image scales, returned independently for each sample."""
    if rec.shape != target.shape or rec.ndim != 4:
        raise ValueError("rec and target must have matching (B,D,H,W) shapes")
    losses = []
    for scale in scales:
        if scale == 1:
            r, t = rec, target
        elif rec.shape[-1] % scale == 0 and rec.shape[-2] % scale == 0:
            r = F.avg_pool2d(_flatten_planes(rec), scale).reshape_as(rec[..., ::scale, ::scale])
            t = F.avg_pool2d(_flatten_planes(target), scale).reshape_as(target[..., ::scale, ::scale])
        else:
            continue
        losses.append((r - t).abs().mean((-2, -1)))
    return torch.stack(losses, -1).mean(-1).mean(-1)


def _tv(x):
    vertical = (x[..., 1:, :] - x[..., :-1, :]).abs().mean((-2, -1))
    horizontal = (x[..., :, 1:] - x[..., :, :-1]).abs().mean((-2, -1))
    return vertical + horizontal


def excess_tv_loss(rec: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize reconstruction roughness only above the target's own roughness."""
    if rec.shape != target.shape or rec.ndim != 4:
        raise ValueError("rec and target must have matching (B,D,H,W) shapes")
    excess = F.relu(_tv(rec) - _tv(target))
    return excess.mean(-1)
