"""Optical parameters of the case (slide 5). All lengths in metres."""

N = 128                     # DOE and image size, pixels
DX = 10e-6                  # SLM pixel pitch
WAVELENGTH = 532e-9
DEPTHS = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)  # image i is reconstructed at DEPTHS[i]


def output_pitch(z: float) -> float:
    """Pixel pitch in the reconstruction plane of the single-FFT Fresnel transform."""
    return WAVELENGTH * z / (N * DX)


def critical_distance() -> float:
    """z at which the DOE-plane chirp hits Nyquist at the aperture edge (~24.06 mm)."""
    return N * DX**2 / WAVELENGTH
