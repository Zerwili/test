"""
三种势垒定义 + 复吸收势 (CAP)
自然单位: ℏ = m = 1
"""

import numpy as np
from config import SimParams


def square_barrier(x: np.ndarray, p: SimParams) -> np.ndarray:
    return np.where((x >= p.barrier_a) & (x <= p.barrier_b), p.V0, 0.0)


def eckart_barrier(x: np.ndarray, p: SimParams) -> np.ndarray:
    return p.V0 / np.cosh(p.alpha * (x - p.x_c)) ** 2


def double_barrier(x: np.ndarray, p: SimParams) -> np.ndarray:
    V = np.zeros_like(x)
    V[(x >= p.b1_a) & (x <= p.b1_b)] = p.V0
    V[(x >= p.b2_a) & (x <= p.b2_b)] = p.V0
    return V


def complex_absorbing_potential(x: np.ndarray, p: SimParams) -> np.ndarray:
    V_cap = np.zeros_like(x, dtype=np.complex128)
    L = p.cap_width
    eta = p.cap_strength
    n = p.cap_power

    left_mask = x < (p.x_min + L)
    right_mask = x > (p.x_max - L)

    V_cap[left_mask] = -1j * eta * ((p.x_min + L - x[left_mask]) / L) ** n
    V_cap[right_mask] = -1j * eta * ((x[right_mask] - (p.x_max - L)) / L) ** n

    return V_cap


POTENTIALS = {
    "square": square_barrier,
    "eckart": eckart_barrier,
    "double": double_barrier,
}


def get_potential(x: np.ndarray, p: SimParams, name: str, with_cap: bool = True) -> np.ndarray:
    V_real = POTENTIALS[name](x, p)
    if with_cap:
        V = V_real.astype(np.complex128) + complex_absorbing_potential(x, p)
    else:
        V = V_real.astype(np.complex128)
    return V
