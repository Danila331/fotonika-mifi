"""Single-FFT Fresnel propagation (slide 2), NumPy reference and differentiable Torch version.

K(x,y) = exp(jkz)/(jλz) · exp(jπ(x²+y²)/λz) · FFT[ H(u,v) · exp(jπ(u²+v²)/λz) ]

The prefactor and the output chirp have unit modulus (up to the constant 1/λz), so they do not
change the reconstructed intensity after per-image brightness normalisation. We therefore work
with the "reduced field" G_z = FFTc[H · Q_z], Q_z = exp(jπ(u²+v²)/λz), using a centred, unitary
(norm="ortho") FFT: Parseval holds, |G_z|² sums to the DOE energy at every depth.

Coordinates are centred: u = (n - N/2)·dx, so pixel N/2 is the optical axis. FFT is the forward
DFT (negative exponent), which is what the Fresnel kernel exp(+jk/2z (x-u)²) produces.
"""

import numpy as np
import torch

from .config import DEPTHS, DX, N, WAVELENGTH


def coords(n: int = N, dx: float = DX) -> np.ndarray:
    return (np.arange(n) - n // 2) * dx


def chirp_np(z: float, n: int = N, dx: float = DX, wavelength: float = WAVELENGTH) -> np.ndarray:
    u = coords(n, dx)
    r2 = u[:, None] ** 2 + u[None, :] ** 2
    return np.exp(1j * np.pi / (wavelength * z) * r2)


def _fftc(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def _ifftc(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def propagate_np(phase: np.ndarray, depths=DEPTHS) -> np.ndarray:
    """phase (N,N) radians -> reduced complex fields (D,N,N), complex128."""
    h = np.exp(1j * np.asarray(phase, dtype=np.float64))
    return np.stack([_fftc(h * chirp_np(z, h.shape[-1])) for z in depths])


def intensity_np(phase: np.ndarray, depths=DEPTHS) -> np.ndarray:
    return np.abs(propagate_np(phase, depths)) ** 2


def backpropagate_np(field: np.ndarray, z: float) -> np.ndarray:
    """Exact inverse of one plane of propagate_np: reduced field at z -> complex DOE field."""
    return _ifftc(field) * np.conj(chirp_np(z, field.shape[-1]))


class Fresnel(torch.nn.Module):
    """Differentiable multi-depth propagator. forward: phase (B,N,N) -> complex field (B,D,N,N)."""

    def __init__(self, depths=DEPTHS, n: int = N, dx: float = DX, wavelength: float = WAVELENGTH):
        super().__init__()
        self.depths = tuple(depths)
        q = np.stack([chirp_np(z, n, dx, wavelength) for z in self.depths])
        self.register_buffer("chirp_re", torch.tensor(q.real, dtype=torch.float32))
        self.register_buffer("chirp_im", torch.tensor(q.imag, dtype=torch.float32))

    @property
    def chirp(self) -> torch.Tensor:
        return torch.complex(self.chirp_re, self.chirp_im)

    @staticmethod
    def fftc(x):
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))

    @staticmethod
    def ifftc(x):
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        h = torch.complex(torch.cos(phase), torch.sin(phase))
        return self.fftc(h.unsqueeze(-3) * self.chirp)

    def intensity(self, phase: torch.Tensor) -> torch.Tensor:
        f = self.forward(phase)
        return f.real**2 + f.imag**2

    def backward_field(self, field: torch.Tensor) -> torch.Tensor:
        """Per-depth exact inverse: (B,D,N,N) reduced fields -> (B,D,N,N) DOE-plane fields."""
        return self.ifftc(field) * torch.conj(self.chirp)
