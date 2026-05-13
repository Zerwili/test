"""
高斯波包初始化
自然单位: ℏ = m = 1
"""

import numpy as np
from config import SimParams


def gaussian_wavepacket(x: np.ndarray, p: SimParams) -> np.ndarray:
    psi = (
        (2.0 * np.pi * p.sigma ** 2) ** (-0.25)
        * np.exp(-(x - p.x0) ** 2 / (4.0 * p.sigma ** 2))
        * np.exp(1j * p.k0 * x)
    )
    norm = np.sqrt(np.trapezoid(np.abs(psi) ** 2, x))
    return psi / norm
