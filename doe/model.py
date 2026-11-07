"""Neural phase-DOE synthesis models (ML agent).

All models map targets (B,4,N,N) in [0,1] to a phase (B,N,N) in [-π, π] in a single forward pass.
The Fresnel propagator is part of the model (fixed, non-trainable), so the non-local image <-> DOE
mapping is done analytically and the CNNs only have to solve local problems.

Variants (`kind`):
  "doe"    - CNN acts in the DOE plane. Input: targets and/or physics features (Re/Im of each
             plane's back-propagated target amplitude with a fixed random phase and of their sum).
             Output head: 2 channels (a, b) -> phase = atan2(b, a); optional residual on angle(sum).
  "image"  - DeepCGH-style. CNN acts in the image planes: it predicts a complex field for each of the
             4 planes (8 channels); they are back-propagated with the exact inverse Fresnel operator,
             summed in the DOE plane and the phase of the sum is taken.
  "unroll" - "image" followed by K learned refinement stages (learned Gerchberg-Saxton): propagate the
             current phase, feed [target amplitude, |field|, field phasor] to a small CNN that proposes new
             complex plane fields, back-propagate, sum with the current DOE field, take the phase.
             (`unroll=K` adds the same stages after the "doe" first stage as well.)
Targets are amplitudes (expert rule E1, `quantity="amplitude"`) normalised to max 1 (E2). `bp_phase`
selects the image-plane phase of the back-projection: "random" (diffuse), "zero" or "quad<a>" (smooth).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import N
from .physics import Fresnel


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, c // 4) if c >= 8 else 1, c)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.n1, self.c1 = _gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = _gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    """Global self-attention over the bottleneck tokens (global mixing for a local CNN)."""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.norm = _gn(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        t = self.norm(x).flatten(2).transpose(1, 2)
        t, _ = self.attn(t, t, t, need_weights=False)
        return x + t.transpose(1, 2).reshape(b, c, h, w)


class UNet(nn.Module):
    def __init__(self, cin: int, cout: int, width: int = 32, mults=(1, 2, 4, 8), blocks: int = 2,
                 attn: bool = True):
        super().__init__()
        chs = [width * m for m in mults]
        self.stem = nn.Conv2d(cin, chs[0], 3, padding=1)
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        c = chs[0]
        skips = []
        for i, ch in enumerate(chs):
            level = nn.ModuleList()
            for _ in range(blocks):
                level.append(ResBlock(c, ch))
                c = ch
            self.down.append(level)
            skips.append(c)
            self.pool.append(nn.Conv2d(c, c, 3, stride=2, padding=1) if i < len(chs) - 1 else nn.Identity())
        self.mid1 = ResBlock(c, c)
        self.mid_attn = Attention(c) if attn else nn.Identity()
        self.mid2 = ResBlock(c, c)
        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i, ch in reversed(list(enumerate(chs))):
            level = nn.ModuleList()
            for j in range(blocks):
                level.append(ResBlock(c + (skips[i] if j == 0 else 0), ch))
                c = ch
            self.up.append(level)
            self.upsample.append(nn.Upsample(scale_factor=2, mode="nearest") if i > 0 else nn.Identity())
        self.out_norm = _gn(c)
        self.out = nn.Conv2d(c, cout, 3, padding=1)

    def forward(self, x):
        h = self.stem(x)
        hs = []
        for level, pool in zip(self.down, self.pool):
            for blk in level:
                h = blk(h)
            hs.append(h)
            h = pool(h)
        h = self.mid2(self.mid_attn(self.mid1(h)))
        for level, ups in zip(self.up, self.upsample):
            h = torch.cat([h, hs.pop()], 1)
            for blk in level:
                h = blk(h)
            h = ups(h)
        return self.out(F.silu(self.out_norm(h)))


class NeuralDOENet(nn.Module):
    def __init__(self, kind: str = "image", inputs: str = "physics", residual: bool = False, width: int = 32,
                 mults=(1, 2, 4, 8), blocks: int = 2, attn: bool = True, unroll: int = 0,
                 unroll_width: int = 24, unroll_mults=(1, 2, 4), seed: int = 1234,
                 quantity: str = "amplitude", bp_phase: str = "random"):
        super().__init__()
        self.cfg = dict(kind=kind, inputs=inputs, residual=residual, width=width, mults=tuple(mults),
                        blocks=blocks, attn=attn, unroll=unroll, unroll_width=unroll_width,
                        unroll_mults=tuple(unroll_mults), seed=seed, quantity=quantity, bp_phase=bp_phase)
        self.kind, self.inputs, self.residual = kind, inputs, residual
        self.fresnel = Fresnel()
        self.D = len(self.fresnel.depths)
        self.quantity = quantity
        # image-plane phase used for the back-projection features / bias ("diffuse" vs "smooth" holography)
        if bp_phase == "random":
            g = torch.Generator().manual_seed(seed)
            ph = torch.rand(self.D, N, N, generator=g) * 2 * math.pi
        elif bp_phase == "zero":
            ph = torch.zeros(self.D, N, N)
        elif bp_phase.startswith("quad"):  # quadratic phase pi*a*r^2/N (pixels), a = float suffix, default 1
            a = float(bp_phase[4:] or 1.0)
            x = torch.arange(N, dtype=torch.float32) - N // 2
            ph = (math.pi * a * (x[:, None] ** 2 + x[None, :] ** 2) / N).expand(self.D, N, N).clone()
        else:
            raise ValueError(bp_phase)
        self.register_buffer("rand_phase", ph)
        D = self.D
        if kind == "doe":
            cin = {"raw": D, "physics": 2 * D + 2 + D, "physics_only": 2 * D + 2}[inputs]
            self.net = UNet(cin, 2, width, mults, blocks, attn)
        elif kind in ("image", "unroll"):
            cin = {"raw": D, "physics": D + 2 * D}[inputs]
            self.net = UNet(cin, 2 * D, width, mults, blocks, attn)
            nn.init.zeros_(self.net.out.weight)  # start exactly at the random-phase back-projection
            nn.init.zeros_(self.net.out.bias)
        else:
            raise ValueError(kind)
        self.stages = nn.ModuleList(
            [UNet(4 * D, 2 * D, unroll_width, unroll_mults, 1, False) for _ in range(unroll)])
        # weight (softplus) of the current DOE field in each refinement stage, init ~1
        self.stage_mix = nn.Parameter(torch.full((unroll,), 0.5413))
        for st in self.stages:  # start each refinement stage as plain damped GS (zero learned correction)
            nn.init.zeros_(st.out.weight)
            nn.init.zeros_(st.out.bias)

    # ---------------- CNN execution (optional bf16 autocast; physics always stays fp32) ----------------
    cnn_autocast: bool = False

    def _cnn(self, net, x):
        if not self.cnn_autocast:
            return net(x)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            y = net(x)
        return y.float()

    # ---------------- physics helpers ----------------
    def backprop_sum(self, amp):
        """Back-propagate amp*exp(i*rand_phase) per plane. Returns per-plane DOE fields (B,D,N,N)."""
        f = torch.polar(amp, self.rand_phase.expand_as(amp))
        return self.fresnel.backward_field(f)

    @staticmethod
    def _cplx_feats(f, scale):
        return torch.cat([f.real / scale, f.imag / scale], 1)

    def _phase_from_plane_fields(self, out, extra=None):
        """out (B,2D,N,N) -> complex plane fields -> DOE field sum (+extra) -> phase (B,N,N)."""
        D = self.D
        fields = torch.complex(out[:, :D], out[:, D:])
        h = self.fresnel.backward_field(fields).sum(1)
        if extra is not None:
            h = h + extra
        return torch.angle(h), h

    # ---------------- forward ----------------
    def forward(self, target01, return_all: bool = False):
        # E2: the jury divides every target by its own max -> make the model invariant to target brightness
        m = target01.flatten(-2).amax(-1)[..., None, None]
        target01 = torch.where(m > 0, target01 / m.clamp_min(1e-12), target01)
        # E1: targets are AMPLITUDES (legacy intensity models back-propagate sqrt(target))
        amp = target01.clamp_min(0) if self.quantity == "amplitude" else target01.clamp_min(0).sqrt()
        # equal energy per plane (so a bright full-frame target does not dominate the superposition)
        amp = amp / (amp.pow(2).mean((-2, -1), keepdim=True).sqrt() + 1e-3)
        D = self.D
        if self.kind == "doe":
            feats = []
            hb = None
            if self.inputs in ("physics", "physics_only"):
                fb = self.backprop_sum(amp)
                hb = fb.sum(1)
                s = hb.abs().pow(2).mean((-2, -1), keepdim=True).sqrt().unsqueeze(1) + 1e-6
                feats += [self._cplx_feats(fb, s), self._cplx_feats(hb.unsqueeze(1), s)]
            if self.inputs in ("physics", "raw"):
                feats.append(target01 * 2 - 1)
            out = self._cnn(self.net, torch.cat(feats, 1))
            phase = torch.atan2(out[:, 1], out[:, 0])
            if self.residual and hb is not None:
                phase = torch.angle(torch.polar(torch.ones_like(phase), torch.angle(hb) + phase))
        else:
            # image-domain (DeepCGH-style) first stage
            x = target01 * 2 - 1
            ph0 = self.rand_phase.expand_as(amp)
            if self.inputs == "physics":
                x = torch.cat([x, amp * torch.cos(ph0), amp * torch.sin(ph0)], 1)
            out = self._cnn(self.net, x)
            # bias toward the classic back-projection: add sqrt(target)*exp(i*rand) to the predicted fields
            out = out + torch.cat([amp * torch.cos(ph0), amp * torch.sin(ph0)], 1)
            phase, h = self._phase_from_plane_fields(out)
        phases = [phase]
        # learned refinement stages (any first-stage kind)
        for k, stage in enumerate(self.stages):
            g = self.fresnel(phase)  # (B,D,N,N) complex, unitary FFT
            mag = g.abs()
            s = mag.pow(2).mean((-2, -1), keepdim=True).sqrt() + 1e-6
            ta = amp / (amp.pow(2).mean((-2, -1), keepdim=True).sqrt() + 1e-6)
            unit = g / (mag + 1e-6)
            feats = torch.cat([ta, mag / s, unit.real, unit.imag], 1)
            delta = self._cnn(stage, feats)
            # GS proposal: target amplitude with the current phase, plus a learned complex correction
            prop = torch.cat([ta * unit.real, ta * unit.imag], 1) + delta
            # keep a (learned) fraction of the current DOE field: damped update, avoids GS oscillation
            cur = torch.polar(torch.ones_like(phase), phase) * F.softplus(self.stage_mix[k])
            phase, h = self._phase_from_plane_fields(prop, extra=cur)
            phases.append(phase)
        return phases if return_all else phase


def build_model(cfg: dict) -> NeuralDOENet:
    return NeuralDOENet(**cfg)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
